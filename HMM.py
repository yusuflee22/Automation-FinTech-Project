import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

data = yf.download('SPY', start='2018-01-01', interval='1d', progress=False)


data = data[['Close', 'Volume']]
data.columns = ['price', 'Volume']  # or data = data.xs('SPY', axis=1, level='Ticker')

data['ret'] = np.log(data['price'] / data['price'].shift(1))

window = 21
data['vol'] = data['ret'].rolling(window).std()
data['vol_of_vol'] = data['vol'].rolling(window).std()

def rolling_autocorr(x):
    return x.autocorr(lag=1)

data['rolling_autocorr'] = data['ret'].rolling(window).apply(rolling_autocorr, raw=False)

def realized_skew(x):
    return ((x - x.mean())**3).mean() / (x.std()**3 + 1e-9)

def realized_kurtosis(x):
    return ((x - x.mean())**4).mean() / (x.std()**4 + 1e-9)

data['skew'] = data['ret'].rolling(window).apply(realized_skew, raw=False)
data['kurtosis'] = data['ret'].rolling(window).apply(realized_kurtosis, raw=False)

#CREATE BINARY VOLUME SHOCK FEATURE: Z_SCORE > 2 IS 95TH PERCENTILE
volume_avg = data['Volume'].rolling(window).mean()
volume_std = data['Volume'].rolling(window).std()
data['volume_zscore'] = (data['Volume'] - volume_avg) / (volume_std + 1e-9)
data['volume_shock'] = (data['volume_zscore'] > 2).astype(int)

data['SMA50'] = data['price'].rolling(window=50).mean()
data['price_rel_sma50'] = data['price'] / data['SMA50'] - 1

#RSI: OVER 70 INDICATES OVERBOUGHT, BELOW 30 OVERSOLD
delta = data['price'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
RS = avg_gain / avg_loss
data['RSI_14'] = 100 - (100 / (1 + RS))

feature_columns = [
    'vol',
    'vol_of_vol',
    'rolling_autocorr',
    'skew',
    'kurtosis',
    'volume_shock',
    'price_rel_sma50',
    'RSI_14',
    'ret'
]
features = data[feature_columns].dropna()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features.values)

n_states = 3
model = GaussianHMM(
    n_components = n_states, 
    covariance_type='full', 
    n_iter=1000, 
    random_state=42)

model.fit(scaled_features)
hidden_states = model.predict(scaled_features)

features['regime'] = hidden_states
data = data.join(features['regime'], how='inner')
data['regime'] = data['regime'].astype(int)


def get_market_regime_stats():
    state_stats = []
    for i in range(n_states):
        state_data = data[data['regime'] == i]['ret']
        state_stats.append({
            'state': i,
            'mean_return': state_data.mean(),
            'volatility': state_data.std(),
            'skewness': state_data.skew(),
            'kurtosis': state_data.kurtosis()
        })
    state_stats_df = pd.DataFrame(state_stats)
    plot_df = data.dropna(subset=['price', 'regime']).copy()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['price'],
        mode='lines', name='SPY Price',
        line=dict(color='lightgray', width=1)
    ))

    # Add Colored Dots for Regimes
    colors = ['red', 'green', 'blue', 'orange']
    for state in range(n_states):
        state_mask = plot_df['regime'] == state
        subset = plot_df[state_mask]
        
        fig.add_trace(go.Scatter(
            x=subset.index, y=subset['price'],
            mode='markers', name=f'Regime {state}',
            marker=dict(color=colors[state % len(colors)], size=4)
        ))

    fig.update_layout(
        title='SPY Market Regimes (Historical)',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        height=500
    )

    return fig, state_stats_df


if __name__ == "__main__":
    fig, stats = get_market_regime_stats()
    print(stats)
    fig.show()


