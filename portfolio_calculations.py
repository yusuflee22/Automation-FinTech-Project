from __future__ import annotations
import numpy as np
import pandas as pd
from data_loader_cleaner import get_ticker_data, clean_ticker_data

TRADING_DAYS = 252
def load_portfolio(csv_path: str) -> pd.DataFrame:
   
    return result
def extract_close_prices(market_data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of close prices where columns are ticker symbols.
    """
    close_cols = [col for col in market_data.columns if col.endswith("_Close")]
    if not close_cols:
        raise ValueError("No *_Close columns available.")
    closes = market_data[close_cols].copy()
    closes.columns = [col.replace("_Close", "") for col in close_cols]
    return closes
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily percentage returns for the provided price history.
    """
    return prices.pct_change().dropna(how="all")
def mean_covariance(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Return the mean daily return vector and covariance matrix.
    """
    mu = returns.mean()
    sigma = returns.cov()
    return mu, sigma
def portfolio_return_stats(
    weights: pd.Series, mean_returns: pd.Series, cov_matrix: pd.DataFrame
) -> tuple[float, float]:
    """
    Compute portfolio expected daily return and volatility given weights.
    """
    tickers = [ticker for ticker in weights.index if ticker in mean_returns.index]
    if not tickers:
        raise ValueError("Weights do not align with available return series.")
    w = weights.loc[tickers].values
    mu_vec = mean_returns.loc[tickers].values
    sigma_sub = cov_matrix.loc[tickers, tickers].values
    mu_p = float(np.dot(w, mu_vec))
    var_p = float(np.dot(w, sigma_sub @ w))
    sigma_p = np.sqrt(var_p)
    return mu_p, sigma_p
def annualize_stats(mu_daily: float, sigma_daily: float) -> tuple[float, float]:
    """
    Annualize daily return and volatility using the trading-day constant.
    """
    mu_ann = mu_daily * TRADING_DAYS
    sigma_ann = sigma_daily * np.sqrt(TRADING_DAYS)
    return mu_ann, sigma_ann
def sharpe_ratio(mu_daily: float, sigma_daily: float, risk_free_daily: float = 0.0) -> float:
    """
    Return the annualized Sharpe ratio given daily stats and a daily risk-free rate.
    """
    excess_return = (mu_daily - risk_free_daily) * TRADING_DAYS
    annual_sigma = sigma_daily * np.sqrt(TRADING_DAYS)
    return excess_return / annual_sigma if annual_sigma > 0 else np.nan

if __name__ == "__main__":
    portfolio = load_portfolio("test_user_input.csv")
    stats = calculate_portfolio_statistics(portfolio)
    print(stats)
    history = clean_ticker_data(get_ticker_data(portfolio))
    close_prices = extract_close_prices(history)
    returns = daily_returns(close_prices)
    mu, sigma = mean_covariance(returns)
    weights = stats.set_index("ticker")["portfolio_weight"]
    mu_p, sigma_p = portfolio_return_stats(weights, mu, sigma)
    mu_ann, sigma_ann = annualize_stats(mu_p, sigma_p)
    sharpe = sharpe_ratio(mu_p, sigma_p)
    print("\nDaily portfolio return:", mu_p)
    print("Daily portfolio volatility:", sigma_p)
    print("Annualized return / vol:", mu_ann, sigma_ann)
    print("Sharpe ratio:", sharpe)