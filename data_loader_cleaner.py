import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf


def _extract_tickers(user_data: pd.DataFrame) -> list[str]:
    """Return the first column as cleaned ticker symbols."""
    if user_data.empty:
        raise ValueError("No tickers provided in user_data.")

    tickers = user_data.iloc[:, 0].dropna().astype(str).str.strip()
    tickers = tickers[tickers != ""]
    if tickers.empty:
        raise ValueError("Ticker column is empty after cleaning.")
    return tickers.tolist()


def get_ticker_data(user_data: pd.DataFrame) -> pd.DataFrame:
    """Fetch daily data for supplied tickers."""
    tickers = _extract_tickers(user_data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 7)

    raw_data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False,
        group_by="column",
    )
    return raw_data.dropna()


def clean_ticker_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Keep Close/Volume columns; flatten MultiIndex headers."""
    desired_columns = ["Close", "Volume"]
    if isinstance(raw_data.columns, pd.MultiIndex):
        idx = pd.IndexSlice
        filtered = raw_data.loc[:, idx[desired_columns, :]].copy()
        filtered.columns = [
            f"{ticker}_{metric.title()}" for metric, ticker in filtered.columns
        ]
        return filtered

    missing = [col for col in desired_columns if col not in raw_data.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")
    return raw_data.loc[:, desired_columns].copy()


if __name__ == "__main__":
    df = pd.read_csv("test_user_input.csv", header=None, names=["ticker", "quantity"])
    cleaned = clean_ticker_data(get_ticker_data(df))
    print(cleaned.head())
