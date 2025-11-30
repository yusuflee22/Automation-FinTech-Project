import pandas as pd
from datetime import datetime
import plotly.graph_objs as go
import numpy as np

from data_pipeline import (
    load_positions,
    fetch_prices,
    clean_price_data,
    build_portfolio_series,
)
from performance_vs_benchmarks import (
    fetch_and_normalize_benchmark_prices,
    compute_performance_comparison,
    plot_performance
)

from monte_carlo import (
    compute_portfolio_features,
    build_portfolio_hmm,
    merge_regimes_back,
    run_portfolio_monte_carlo,
    extract_hmm_return_params
)

from HMM import data as market_data

from HMM import get_market_regime_stats
from ret_vol_pe import analyze_risk_return_pe
from performance_vs_benchmarks import run_demo
from portfolio_corr import analyze_correlations



def analyze_portfolio(csv_path):
    positions = load_positions(csv_path)

    tickers = positions["ticker"].tolist()

    raw_prices = fetch_prices(tickers)
    clean_prices = clean_price_data(raw_prices)

    portfolio_series = build_portfolio_series(positions, clean_prices)


    full_df, features, feature_cols = compute_portfolio_features(portfolio_series)

    # Ensure market_data has a datetime index and a `regime` column
    market_regimes = market_data[['regime']].rename(columns={'regime': 'regime_market'})

    # Align market regimes to portfolio feature dates
    features = features.merge(market_regimes, left_index=True, right_index=True, how='left')
    features = features.dropna()

    print("\nFeature sample:")
    print(features.head())


    model, scaler, features_w_states = build_portfolio_hmm(features)
    features["regime"] = features_w_states["regime"]

    print("\nState counts:")
    print(features["regime"].value_counts())

    regime_df = merge_regimes_back(full_df, features_w_states)

    start_value = portfolio_series["Portfolio_Value"].iloc[-1]

    sim_paths = run_portfolio_monte_carlo(
        start_value=start_value,
        days=252,
        model=model,
        scaler=scaler,
        features=features,
        n_paths=200,
        plot=False
    )

    print("\nSimulation complete.")


    print("\nRaw return sample:", full_df["ret"].head())
    print("Feature return sample:", features["ret"].head())

    return_means, return_stds, _ = extract_hmm_return_params(model, features, scaler)
    print("\nHMM return means:", return_means)
    print("HMM return stds:", return_stds)


    # Portfolio correlation
    fig_corr, sectors = analyze_correlations(csv_path)
    fig_mc = go.Figure()
    days_range = np.arange(sim_paths.shape[1])
    for i in range(min(50, sim_paths.shape[0])):
        fig_mc.add_trace(go.Scatter(
            x=days_range, 
            y=sim_paths[i],
            mode='lines', 
            line=dict(width=1),
            showlegend=False
        ))
    mean_path = sim_paths.mean(axis=0)
    
    fig_mc.add_trace(go.Scatter(
        x=days_range, 
        y=mean_path,
        mode='lines', 
        name='Mean Projection',
        line=dict(width=3, color='blue')
    ))
    
    fig_mc.update_layout(
        title="Monte Carlo Projection (1 Year Horizon)", 
        template="plotly_white",
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value ($)",
        hovermode="x"
    )
    return_means, _, _ = extract_hmm_return_params(model, features, scaler)
    
    # Calculate expected end value
    expected_end_val = mean_path[-1]
    roi = (expected_end_val - start_value) / start_value

    metrics = {
        "Start Value": f"${start_value:,.2f}",
        "Current Regime": str(features["regime"].iloc[-1]),
        "Est. Annual Return": f"{roi:.2%}",
        "Sectors Detected": ", ".join(set(sectors.values()))
    }


    print("\n=== Running Portfolio vs Benchmarks ===\n")
    fig_bench = run_demo(csv_path)

    print("\n=== Running HMM Regime Analysis ===\n")
    fig_hmm, hmm_stats = get_market_regime_stats()

    print("\n=== Running Return/Vol/PE Analysis ===\n")
    fig_fundamental, fund_df = analyze_risk_return_pe(csv_path)

    print("\nAll tasks complete.")

    return fig_mc, fig_corr, fig_bench, fig_fundamental, fig_hmm, metrics


if __name__ == "__main__":
    print("\n--- Starting Local Analysis ---")
    try:
        # Expecting 5 figures and 1 dict
        # We unpack them into variables so we can show them
        mc, corr, bench, hmm, fund, mets = analyze_portfolio("sample_portfolio.csv")
        
        print("\nMetrics Calculated:", mets)
        print("\nDisplaying all 5 charts...")
        
        mc.show()
        corr.show()
        bench.show()
        hmm.show()
        fund.show()

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")