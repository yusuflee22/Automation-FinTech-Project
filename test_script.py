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

csv_path = "test_user_input.csv"
positions = load_positions(csv_path)

tickers = positions["ticker"].tolist()
raw_prices = fetch_prices(tickers)
clean_prices = clean_price_data(raw_prices)

portfolio_series = build_portfolio_series(positions, clean_prices)
default_benchmarks = ["SPY","VOO","QQQ","DIA"]
start_date = portfolio_series.index[0]


benchmark_df = fetch_and_normalize_benchmark_prices(start_date,default_benchmarks)
combined_performance_comparison = compute_performance_comparison(portfolio_series, start_date, default_benchmarks)

plot_performance(combined_performance_comparison).show()

