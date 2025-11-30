import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
from data_pipeline import (
    load_positions
)

tickers = load_positions('sample_portfolio.csv')['ticker'].tolist()


def calculate_vol(prices, window=252):
    log_returns = np.log(prices / prices.shift(1))
    rolling_std = log_returns.rolling(window=window).std()
    annualized_vol = rolling_std * np.sqrt(252)
    return annualized_vol

def calculate_annual_ret(prices, lookback_days=252):
    """
    Approximate 1Y return using the latest price vs. the price
    `lookback_days` trading sessions earlier. Returns NaN if there
    isn't enough history yet.
    """
    if len(prices) <= lookback_days:
        return np.nan

    latest_price = prices.iloc[-1]
    past_price = prices.iloc[-lookback_days]
    return (latest_price / past_price) - 1



def analyze_risk_return_pe(csv_path):
    positions = load_positions(csv_path)
    tickers = positions["ticker"].tolist()

    metrics = []
    for ticker in tickers:
        price_data = yf.download(
            ticker,
            start='2018-01-01',
            interval='1d',
            progress=False,
            auto_adjust=False
        )

        close_prices = price_data['Close']
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.squeeze("columns")

        hist_price = close_prices.dropna()
        info = yf.Ticker(ticker).info

        vol_series = calculate_vol(hist_price)
        current_vol = vol_series.iloc[-1]
        annual_ret = calculate_annual_ret(hist_price)

        metrics.append({
            'Ticker': ticker,
            'Volatility': current_vol,
            'Annual Return': annual_ret,
            'PE Ratio': info.get('trailingPE', np.nan),
        })

    metrics_df = pd.DataFrame(metrics)

# PLOT 3D SCATTER FOR PE-VOL-RET

    fig = px.scatter_3d(
        metrics_df,
        x='Annual Return',
        y='Volatility',
        z='PE Ratio',
        color='Ticker',
        hover_name='Ticker',
        title='Ticker Volatility vs Annual Return vs PE Ratio'
    )

    return fig, metrics_df

if __name__ == "__main__":

    try:
        fig, df = analyze_risk_return_pe("sample_portfolio.csv")
        print(df)
        fig.show()
    except Exception as e:
        print(f"Local test failed: {e}")