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
from HMM import data

csv_path = "test_user_input.csv"
positions = load_positions(csv_path)

tickers = positions["ticker"].tolist()
raw_prices = fetch_prices(tickers)
clean_prices = clean_price_data(raw_prices)

portfolio_series = build_portfolio_series(positions, clean_prices)
default_benchmarks = ["SPY","VOO","QQQ","DIA"]
start_date = portfolio_series.index[0]


full_df, features, feature_cols = compute_portfolio_features(portfolio_series)
market_regimes = data[['regime']].rename(columns={'regime': 'regime_market'})
features = features.merge(market_regimes, left_index=True, right_index=True, how='left')

features = features.dropna()



print("Feature sample:")
print(features.head())


model, scaler, features_w_states = build_portfolio_hmm(features)
features["regime"] = features_w_states["regime"]

print("State counts:")
print(features_w_states["regime"].value_counts())


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

print("Simulation complete.")


print("Raw return sample:", full_df["ret"].head())
print("Feature return sample:", features["ret"].head())

return_means, return_stds, _ = extract_hmm_return_params(model, features, scaler)
print("HMM return means:", return_means)
print("HMM return stds:", return_stds)
