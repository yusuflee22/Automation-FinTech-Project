import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def load_positions(csv_path: str) -> pd.DataFrame:
    """
    Expect columns: ticker, shares
    Returns a clean DataFrame with uppercase tickers and float shares.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"ticker", "shares"} <= set(df.columns):
        raise ValueError("CSV must have columns: ticker, shares")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df = df.dropna(subset=["ticker","shares"])
    df = df[df["shares"] != 0]
    return df.reset_index(drop=True)

def fetch_prices(tickers, start="2018-01-01"):
    """
    Returns a DataFrame of adjusted Close prices for the list of tickers.
    """
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    # yfinance returns a MultiIndex for multiple tickers; handle 1 vs many
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"].copy()
    else:
        px = data[["Close"]].rename(columns={"Close": tickers[0]})
    return px.dropna(how="all")
