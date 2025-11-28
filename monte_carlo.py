import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go


def compute_portfolio_features(portfolio_series,window=21):
    # computes portfolio features
    
    df = portfolio_series.copy()
    df = df.rename(columns={"Portfolio_Value": "value",
                            "Portfolio_Return": "ret"})
    
    # desired features: vol, vol_of_vol, rolling_autocorr, skew, kurtosis, 
    # SMA50 momentum, RSI-14
    df["vol"] = df["ret"].rolling(window).std()

    df["vol_of_vol"] = df["vol"].rolling(window).std()

    def rolling_autocorr(x):
        return x.autocorr(lag=1)
    df["rolling_autocorr"] = df["ret"].rolling(window).apply(rolling_autocorr, raw=False)

    def realized_skew(x):
        return ((x - x.mean())**3).mean() / (x.std()**3 + 1e-9)
    df["skew"] = df["ret"].rolling(window).apply(realized_skew, raw=False)
    
    def realized_kurt(x):
        return ((x - x.mean())**4).mean() / (x.std()**4 + 1e-9)
    df["kurtosis"] = df["ret"].rolling(window).apply(realized_kurt, raw=False)

    df["SMA50"] = df["value"].rolling(window=50).mean()
    df["price_rel_sma50"] = df["value"] / df["SMA50"] - 1

    delta = df["value"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    RS = avg_gain / (avg_loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + RS))

    feature_cols = [
        "vol",
        "vol_of_vol",
        "rolling_autocorr",
        "skew",
        "kurtosis",
        "price_rel_sma50",
        "RSI_14",
        "ret"
    ]

    features = df[feature_cols].dropna()

    return df, features, feature_cols

def build_portfolio_hmm(features, n_states=3):
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.values)
    scaler.feature_names = list(features.columns)


    model = GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        n_iter=1000,
        random_state=42
    )

    model.fit(scaled)
    hidden_states = model.predict(scaled)

    features = features.copy()
    features["regime"] = hidden_states

    return model, scaler, features

def merge_regimes_back(full_df, features):
    out = full_df.join(features["regime"], how="inner")
    out["regime"] = out["regime"].astype(int)
    return out

def summarize_portfolio_regimes(full_df):
    rows = []
    for state in full_df["regime"].unique():
        r = full_df[full_df["regime"] == state]["ret"]
        rows.append({
            "regime": state,
            "mean_return": r.mean(),
            "volatility": r.std(),
            "skewness": r.skew(),
            "kurtosis": r.kurtosis()
        })
    return pd.DataFrame(rows)


# the monte carlo part


def extract_hmm_return_params(model, features, scaler):
    
    feature_columns = list(features.columns)
    ret_idx = feature_columns.index("ret")

 
    scaled_means = model.means_[:, ret_idx]
    scaled_vars  = model.covars_[:, ret_idx, ret_idx]



    orig_mean = scaler.mean_[ret_idx]
    orig_std  = scaler.scale_[ret_idx]
    return_means = scaled_means * orig_std + orig_mean
    return_stds  = np.sqrt(scaled_vars) * orig_std

    return return_means, return_stds, model.transmat_

def simulate_single_path(
    start_value,
    days,
    model,
    scaler,
    features,
    market_conditionals,
    market_influence=0.7
):

    # Extract HMM parameters
    return_means, return_stds, transition_matrix = extract_hmm_return_params(model, features,scaler)

    
    scaled_features = scaler.transform(features[scaler.feature_names].values)
    current_state = model.predict(scaled_features)[-1]

    values = [start_value]
    state = current_state

    for _ in range(days):
        
        current_market_state = features["regime_market"].iloc[-1]

# Pure portfolio transition
        p_port = transition_matrix[state]

# Market-adjusted distribution
        p_mkt = market_conditionals.loc[current_market_state].values

# Blend
        p_next = market_influence * p_mkt + (1 - market_influence) * p_port
        p_next = p_next / p_next.sum()

        state = np.random.choice(
            a=np.arange(len(return_means)),
            p=p_next
        )

        ret = np.random.normal(return_means[state], return_stds[state])
        ret = np.clip(ret, -0.3, 0.3)

        values.append(values[-1] * (1 + ret))

    return values

def simulate_many_paths(
    start_value,
    days,
    model,
    scaler,
    features,
    market_conditionals,
    n_paths=500
):
    
    all_paths = []

    for _ in range(n_paths):
        path = simulate_single_path(
            start_value=start_value,
            days=days,
            model=model,
            scaler=scaler,
            features=features,
            market_conditionals=market_conditionals
        )
        all_paths.append(path)

    return np.array(all_paths)

def plot_monte_carlo_paths(mc_paths, dates=None):
   
    n_sims, n_steps = mc_paths.shape

    if dates is None:
        dates = np.arange(n_steps)

    fig = go.Figure()

    # --- Add all simulated paths ---
    for i in range(n_sims):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=mc_paths[i],
                mode="lines",
                line=dict(width=1),
                opacity=0.6,
                showlegend=False
            )
        )

    # --- Add the mean projected path ---
    mean_path = mc_paths.mean(axis=0)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=mean_path,
            mode="lines",
            line=dict(color="red", width=4),
            name="Mean Projection"
        )
    )


    fig.update_layout(
        title="Portfolio Projections via Monte Carlo Simulation (Next 1 Year)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly",
        height=600,
    )

    fig.show()



def run_portfolio_monte_carlo(
    start_value,
    days,
    model,
    scaler,
    features,
    n_paths=500,
    plot=True
):
    
    market_conditionals = (
    features.groupby("regime_market")["regime"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
    )


    mc_paths = simulate_many_paths(
        start_value=start_value,
        days=days,
        model=model,
        scaler=scaler,
        features=features,
        market_conditionals=market_conditionals,
        n_paths=n_paths)

    if plot:
        plot_monte_carlo_paths(mc_paths)
    return mc_paths

