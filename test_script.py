import pandas as pd
from datetime import datetime
import plotly.graph_objs as go

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
import HMM
from HMM import data as market_data




csv_path = "sample_portfolio.csv"
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
    plot=True
)

print("\nSimulation complete.")


print("\nRaw return sample:", full_df["ret"].head())
print("Feature return sample:", features["ret"].head())

return_means, return_stds, _ = extract_hmm_return_params(model, features, scaler)
print("\nHMM return means:", return_means)
print("HMM return stds:", return_stds)

from HMM import main as hmm_main
from ret_vol_pe import main as metrics_main
from performance_vs_benchmarks import run_demo as run_benchmarks

print("\n=== Running Portfolio vs Benchmarks ===\n")
run_benchmarks()

print("\n=== Running HMM Regime Analysis ===\n")
hmm_main()

print("\n=== Running Return/Vol/PE Analysis ===\n")
metrics_main()

print("\nAll tasks complete.")
