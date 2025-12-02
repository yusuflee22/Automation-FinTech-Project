import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import plotly.express as px
from data_pipeline import clean_price_data, fetch_prices, load_positions

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




def analyze_correlations(csv_or_df, plot_backend="plotly", return_matrix=False):
    positions = load_positions(csv_or_df)
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

    if plot_backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            cbar=True,
            ax=ax
        )
        ax.set_title("Portfolio Correlation Matrix")
        plt.tight_layout()
    else:
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f", # Displays the correlation numbers
            aspect="auto",
            color_continuous_scale="RdBu_r", # Red=High Corr, Blue=Inverse
            zmin=-1, zmax=1,
            title="Portfolio Correlation Matrix"
        )
    sectors = fetch_sectors(tickers)
    
    if return_matrix:
        return fig, sectors, corr_matrix
    return fig, sectors



if __name__ == "__main__":
    
    try:
       
        fig, sector_data = analyze_correlations("sample_portfolio.csv")
        
        print("Sectors found:", sector_data)
        fig.show()
    except Exception as e:
        print(f"Error: {e}")
