import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
from data_pipeline import clean_price_data, fetch_prices, load_positions

positions = load_positions("sample_portfolio.csv")
tickers = positions["ticker"].tolist()

#FETCH SECTORS
def fetch_sectors(tickers):
    sectors = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).get_info()
            sectors[t] = info.get("sector", "Unknown")
        except Exception:
            sectors[t] = "Unknown"
    return sectors




def analyze_correlations(csv_path):
    positions = load_positions(csv_path)
    tickers = positions["ticker"].tolist()

    raw_prices = fetch_prices(tickers)
    clean_prices = clean_price_data(raw_prices)

    close_cols = [f"{t}_Close" for t in tickers]
    close_prices = clean_prices[close_cols].copy()
    close_prices.columns = [c.replace("_Close", "") for c in close_prices.columns]

    returns = close_prices.pct_change().dropna(how="all")
    if returns.empty:
        raise ValueError("Not enough price data to compute returns.")


    corr_matrix = returns.corr(method="pearson")

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f", # Displays the correlation numbers
        aspect="auto",
        color_continuous_scale="RdBu_r", # Red=High Corr, Blue=Inverse
        zmin=-1, zmax=1,
        title="Portfolio Correlation Matrix"
    )
    sectors = fetch_sectors(tickers)
    
    return fig, sectors



if __name__ == "__main__":
    
    try:
       
        fig, sector_data = analyze_correlations("sample_portfolio.csv")
        
        print("Sectors found:", sector_data)
        fig.show()
    except Exception as e:
        print(f"Error: {e}")