import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
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

sectors = fetch_sectors(tickers)
print("Ticker sectors:")
for ticker, sector in sectors.items():
    print(f"{ticker}: {sector}")

# Pull price history and get the Close columns per ticker
raw_prices = fetch_prices(tickers)
clean_prices = clean_price_data(raw_prices)
close_cols = [f"{t}_Close" for t in tickers]
close_prices = clean_prices[close_cols]

# Compute daily returns and drop the initial NaNs
returns = close_prices.pct_change().dropna(how="all")
if returns.empty:
    raise ValueError("Not enough price data to compute returns.")

# Correlation across tickers
corr = returns.corr(method="pearson")
print("Correlation matrix:")
print(corr)

# Heatmap
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Portfolio Ticker Correlation Matrix")
plt.tight_layout()
plt.show()


