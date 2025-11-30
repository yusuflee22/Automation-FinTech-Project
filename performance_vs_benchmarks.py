import pandas as pd
import pandas as pd
from datetime import datetime
import yfinance as yf
from datetime import timedelta
import plotly.graph_objs as go
from data_pipeline import (
    load_positions,
    fetch_prices,
    clean_price_data,
    build_portfolio_series
)


default_benchmarks = ["SPY","VOO","QQQ","DIA"]

def fetch_and_normalize_benchmark_prices(start_date,tickers=default_benchmarks):
   # fetches the benchmark yfinance data, normalizes them all to start at 100%

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=datetime.now(),
        auto_adjust=True,
        progress=False
    )

    # MultiIndex if >1 ticker
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"].copy()
    else:
        closes = data[["Close"]].rename(columns={"Close": tickers[0]})

    closes.dropna(how="all")

    normalized = closes/closes.iloc[0] * 100

    return normalized


def compute_performance_comparison(portfolio_series, start_date, benchmarks=default_benchmarks):


    portfolio_trimmed = portfolio_series[portfolio_series.index >= start_date]
    
    # normalize portfolio to 100
    portfolio_norm = portfolio_trimmed / portfolio_trimmed.iloc[0] * 100
  

    # fetch + normalize benchmarks
    bench_norm = fetch_and_normalize_benchmark_prices(
        start_date=start_date,
        tickers=benchmarks
    )

    combined = portfolio_norm.join(bench_norm, how="inner")

    return combined


def plot_performance(perf_df, title="Portfolio vs Benchmarks"):
  
    fig = go.Figure()

    for col in perf_df.columns:
        fig.add_trace(
            go.Scatter(
                x=perf_df.index,
                y=perf_df[col],
                mode="lines",
                name=col
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Dates",
        yaxis_title="Performance Index based on percentage of initial (start = 100)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(title="Series"),
        height=600
    )

    return fig


def run_demo():
   # wrapper function

    print("Running Portfolio vs Benchmark comparison...")

    # Load positions
    positions = load_positions(csv_path="sample_portfolio.csv")

    # Fetch price data for portfolio assets
    tickers = positions["ticker"].tolist()
    raw_prices = fetch_prices(tickers)

    # Clean it (drop NaNs, align, etc.)
    clean_prices = clean_price_data(raw_prices)

    # Build portfolio time series
    portfolio_series = build_portfolio_series(
        position_df=positions,
        clean_prices=clean_prices
    )

    # Choose a start date
    start_date = portfolio_series.index[0]

    perf_df = compute_performance_comparison(
        portfolio_series=portfolio_series,
        start_date=start_date
    )

    # Plot
    fig = plot_performance(perf_df)
    fig.show()

if __name__ == "__main__":
    run_demo()

