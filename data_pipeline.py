import pandas as pd
from datetime import datetime
from datetime import timedelta
import yfinance as yf



def load_positions(csv_path_or_df):
    """
    Load the user's positions from a CSV path or an in-memory DataFrame.
    Expected columns: ticker, shares
    Returns a cleaned DataFrame with:
        - uppercase tickers
        - numeric share counts
    """
    if isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df.copy()
    else:
        df = pd.read_csv(csv_path_or_df)
    df.columns = [c.strip().lower() for c in df.columns]

    if not {"ticker", "shares"} <= set(df.columns):
        raise ValueError("CSV must contain columns: ticker, shares")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")

    df = df.dropna(subset=["ticker", "shares"])
    df = df[df["shares"] != 0]

    return df.reset_index(drop=True)


def fetch_prices(tickers: list[str],
                 start_date=None,
                 end_date=None,
                 interval="1d"):
    """
    Pull OHLCV data for the tickers.
    Returns a MultiIndex DataFrame: (feature, ticker)
    """

    if start_date is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 7)  # default 7 yrs
    else:
        end_date = end_date or datetime.now()

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        group_by="column"
    )

    return raw.dropna(how="all")

def clean_price_data(raw_data):
    """
    Extract Close & Volume columns and flatten MultiIndex.
    Output columns look like:
        AAPL_Close, AAPL_Volume, MSFT_Close, ...
    """
    if isinstance(raw_data.columns, pd.MultiIndex):
        close = raw_data.loc[:, ("Close", slice(None))].copy()
        volume = raw_data.loc[:, ("Volume", slice(None))].copy()

        # Flatten column names
        close.columns = [f"{t}_Close" for _, t in close.columns]
        volume.columns = [f"{t}_Volume" for _, t in volume.columns]

        cleaned = pd.concat([close, volume], axis=1)
    else:
        # single ticker case
        cleaned = raw_data[["Close", "Volume"]].copy()
        ticker = raw_data.columns[0]
        cleaned.columns = [f"{ticker}_Close", f"{ticker}_Volume"]

    return cleaned.dropna(how="all")

def build_portfolio_series(position_df,
                           clean_prices):
    """
    Compute portfolio value + returns over time.

    Inputs:
        position_df: DataFrame with columns [ticker, shares]
        clean_prices: output of clean_price_data()

    Output:
        DataFrame with:
            - Portfolio_Value
            - Portfolio_Return
    """

    tickers = position_df["ticker"].tolist()
    shares = position_df.set_index("ticker")["shares"].to_dict()

    # Extract Close columns
    price_cols = [f"{t}_Close" for t in tickers]
    missing = [c for c in price_cols if c not in clean_prices.columns]
    if missing:
        raise ValueError(f"Missing price data for tickers: {missing}")

    price_df = clean_prices[price_cols].copy()

    # Validate no total missing data
    if price_df.isna().all().all():
        raise ValueError("No valid price history for provided tickers.")

    # Multiply each by shares → position value
    for t in tickers:
        price_df[f"{t}_Value"] = price_df[f"{t}_Close"] * shares[t]

    # Sum → portfolio value
    portfolio_value = price_df[[f"{t}_Value" for t in tickers]].sum(axis=1)
    portfolio_value.name = "Portfolio_Value"

    # Compute returns
    portfolio_return = portfolio_value.pct_change().fillna(0)
    portfolio_return.name = "Portfolio_Return"

    # Package
    out = pd.DataFrame({
        "Portfolio_Value": portfolio_value,
        "Portfolio_Return": portfolio_return
    })

    return out

def compute_portfolio_returns(portfolio_value):
    """
    Compute daily returns from a portfolio value series.
    """
    returns = portfolio_value.pct_change().fillna(0)
    returns.name = "Portfolio_Return"
    return returns
