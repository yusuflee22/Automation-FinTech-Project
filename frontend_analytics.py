from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

# Force a non-GUI backend so Flask requests don't try to open windows on macOS
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from HMM import data as market_data, get_market_regime_stats
from data_pipeline import build_portfolio_series, clean_price_data, fetch_prices, load_positions
from monte_carlo import (
    build_portfolio_hmm,
    compute_portfolio_features,
    merge_regimes_back,
    run_portfolio_monte_carlo,
    summarize_portfolio_regimes,
)
from portfolio_corr import analyze_correlations
from performance_vs_benchmarks import compute_performance_comparison
from ret_vol_pe import analyze_risk_return_pe

PLOTS_DIR = Path("static/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_matplotlib(fig, filename: str) -> str:
    """Persist a Matplotlib figure to the static plots directory."""
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return f"plots/{filename}"


def save_plotly(fig, filename: str) -> str:
    """Persist a Plotly figure as PNG. Requires `kaleido`."""
    path = PLOTS_DIR / filename
    fig.write_image(path, format="png")
    return f"plots/{filename}"


def parse_manual_positions(tickers: List[str], shares: List[str]) -> pd.DataFrame:
    """
    Take raw ticker/share inputs from the form and return a cleaned positions DataFrame.
    """
    data = {"ticker": tickers, "shares": shares}
    df = pd.DataFrame(data)
    return load_positions(df)


def run_var_analysis(returns: pd.Series, latest_value: float, confidence: float = 0.95) -> Dict:
    """
    Compute simple historical VaR metrics and return a Matplotlib histogram path.
    """
    cleaned = pd.Series(returns).dropna()
    if cleaned.empty:
        raise ValueError("Not enough return data to compute VaR.")

    alpha = 1 - confidence
    var_cutoff = cleaned.quantile(alpha)
    var_1d = -var_cutoff * latest_value
    var_10d = var_1d * math.sqrt(10)

    tail = cleaned[cleaned <= var_cutoff]
    expected_shortfall = -tail.mean() * latest_value if not tail.empty else np.nan

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(cleaned, bins=50, color="#4c7cff", alpha=0.8, edgecolor="white")
    ax.axvline(var_cutoff, color="red", linestyle="--", label=f"VaR cutoff ({confidence:.0%})")
    ax.set_title("Portfolio Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()

    fig_path = save_matplotlib(fig, "var_hist.png")

    return {
        "var_1d": var_1d,
        "var_10d": var_10d,
        "expected_shortfall": expected_shortfall,
        "fig_path": fig_path,
    }


def plot_portfolio_regimes(regime_df: pd.DataFrame) -> plt.Figure:
    """
    Build a Matplotlib figure showing portfolio value with HMM regimes highlighted.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(regime_df.index, regime_df["value"], color="#b0bec5", linewidth=1.5, label="Portfolio Value")

    colors = ["#0077b6", "#ff6b6b", "#ffd166", "#06d6a0"]
    for state in sorted(regime_df["regime"].dropna().unique()):
        mask = regime_df["regime"] == state
        ax.scatter(
            regime_df.index[mask],
            regime_df["value"][mask],
            s=10,
            label=f"Regime {int(state)}",
            color=colors[int(state) % len(colors)],
            alpha=0.8,
        )

    ax.set_title("Portfolio Regimes (HMM)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_market_regimes() -> plt.Figure:
    """
    Matplotlib rendering of the market regimes computed in HMM.py.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(market_data.index, market_data["price"], color="#9fb3c8", linewidth=1.5, label="SPY Price")
    colors = ["#0077b6", "#ff6b6b", "#ffd166", "#06d6a0"]
    for state in sorted(market_data["regime"].unique()):
        mask = market_data["regime"] == state
        ax.scatter(
            market_data.index[mask],
            market_data["price"][mask],
            s=8,
            color=colors[int(state) % len(colors)],
            label=f"Regime {int(state)}",
            alpha=0.7,
        )
    ax.set_title("SPY Market Regimes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def run_hmm_analysis(portfolio_series: pd.DataFrame) -> Dict:
    """
    Compute HMM regimes for the portfolio, merge with market regimes, and return artifacts.
    """
    full_df, features, _ = compute_portfolio_features(portfolio_series)

    market_regimes = market_data[["regime"]].rename(columns={"regime": "regime_market"})
    features = features.merge(market_regimes, left_index=True, right_index=True, how="left").dropna()
    if features.empty:
        raise ValueError("Could not align market regimes with portfolio features.")

    model, scaler, features_w_states = build_portfolio_hmm(features)
    features["regime"] = features_w_states["regime"]

    regime_df = merge_regimes_back(full_df, features_w_states)
    summary_df = summarize_portfolio_regimes(regime_df)
    current_regime = int(features["regime"].iloc[-1])

    fig_portfolio = plot_portfolio_regimes(regime_df)
    fig_portfolio_path = save_matplotlib(fig_portfolio, "hmm_portfolio.png")

    _, market_stats = get_market_regime_stats()
    market_fig_path = save_matplotlib(plot_market_regimes(), "hmm_market.png")

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "regime_df": regime_df,
        "summary": summary_df,
        "current_regime": current_regime,
        "portfolio_plot": fig_portfolio_path,
        "market_plot": market_fig_path,
        "market_stats": market_stats,
    }


def run_monte_carlo_simulation(
    features: pd.DataFrame,
    model,
    scaler,
    start_value: float,
    days: int = 252,
    n_paths: int = 300,
) -> Dict:
    """
    Execute Monte Carlo simulation using the existing HMM-based simulator.
    """
    if features.empty:
        raise ValueError("Features are required for Monte Carlo simulation.")

    mc_paths = run_portfolio_monte_carlo(
        start_value=start_value,
        days=days,
        model=model,
        scaler=scaler,
        features=features,
        n_paths=n_paths,
        plot=False,
    )

    fig_mc, ax = plt.subplots(figsize=(10, 4))
    for path in mc_paths:
        ax.plot(path, color="#3ac0ff", alpha=0.15, linewidth=1)

    mean_path = mc_paths.mean(axis=0)
    ax.plot(mean_path, color="#ff6b6b", linewidth=2.5, label="Mean Projection")
    ax.set_title("Monte Carlo Projection (1 Year)")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    plt.tight_layout()

    mc_fig_path = save_matplotlib(fig_mc, "mc_paths.png")

    mean_path = mc_paths.mean(axis=0)

    return {
        "paths": mc_paths,
        "mean_path": mean_path,
        "fig_path": mc_fig_path,
        "expected_end_value": float(mean_path[-1]),
    }


def run_correlation_analysis(positions_df: pd.DataFrame) -> Dict:
    """
    Build the correlation matrix using the existing helper and save to PNG.
    """
    fig_corr, sectors, _ = analyze_correlations(
        positions_df,
        plot_backend="matplotlib",
        return_matrix=True,
    )
    corr_fig_path = save_matplotlib(fig_corr, "correlation.png")

    return {
        "fig_path": corr_fig_path,
        "sectors": sectors,
    }


def run_factor_3d_analysis(csv_or_df) -> Dict:
    """
    Build a 3D scatter (return, volatility, PE) using the existing helper.
    """
    fig_plotly, metrics_df = analyze_risk_return_pe(csv_or_df)

    # Render via Matplotlib to avoid Plotly image export dependency.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    xs = metrics_df["Annual Return"]
    ys = metrics_df["Volatility"]
    zs = metrics_df["PE Ratio"]
    ax.scatter(xs, ys, zs, c="#4c8dff", s=60, depthshade=True)
    for _, row in metrics_df.iterrows():
        ax.text(row["Annual Return"], row["Volatility"], row["PE Ratio"], row["Ticker"], fontsize=8)
    ax.set_xlabel("Annual Return")
    ax.set_ylabel("Volatility")
    ax.set_zlabel("PE Ratio")
    ax.set_title("Return / Volatility / PE (3D)")
    plt.tight_layout()

    fig_path = save_matplotlib(fig, "return_vol_pe.png")

    return {"fig_path": fig_path, "metrics": metrics_df}


def run_benchmark_analysis(portfolio_series: pd.DataFrame, start_date=None) -> Dict:
    """
    Plot portfolio vs benchmark performance using the existing comparison helper.
    """
    if start_date is None:
        start_date = portfolio_series.index[0]

    perf_df = compute_performance_comparison(
        portfolio_series=portfolio_series,
        start_date=start_date,
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in perf_df.columns:
        ax.plot(perf_df.index, perf_df[col], label=col, linewidth=1.8)
    ax.set_title("Portfolio vs Benchmarks (Start = 100)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Indexed Performance")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()

    fig_path = save_matplotlib(fig, "benchmarks.png")
    return {"fig_path": fig_path}


def run_full_analysis(positions_df: pd.DataFrame, confidence: float = 0.95) -> Dict:
    """
    Orchestrate the full portfolio analysis workflow and return all artifacts.
    """
    positions = load_positions(positions_df)
    if positions.empty:
        raise ValueError("No valid tickers/shares were provided.")

    tickers = positions["ticker"].tolist()
    raw_prices = fetch_prices(tickers)
    clean_prices = clean_price_data(raw_prices)
    portfolio_series = build_portfolio_series(positions, clean_prices)

    results: Dict = {
        "positions": positions,
        "portfolio_series": portfolio_series,
        "warnings": [],
    }

    latest_value = float(portfolio_series["Portfolio_Value"].iloc[-1])
    results["latest_value"] = latest_value

    # VaR
    try:
        var = run_var_analysis(portfolio_series["Portfolio_Return"], latest_value, confidence)
        results["var"] = var
    except Exception as exc:  # pragma: no cover - defensive
        results["warnings"].append(f"VaR analysis failed: {exc}")

    # HMM + Monte Carlo
    try:
        hmm = run_hmm_analysis(portfolio_series)
        results["hmm"] = hmm
        mc = run_monte_carlo_simulation(
            features=hmm["features"],
            model=hmm["model"],
            scaler=hmm["scaler"],
            start_value=latest_value,
        )
        results["monte_carlo"] = mc
    except Exception as exc:  # pragma: no cover - defensive
        results["warnings"].append(f"HMM/Monte Carlo analysis failed: {exc}")

    # Correlation
    try:
        corr = run_correlation_analysis(positions)
        results["correlation"] = corr
    except Exception as exc:  # pragma: no cover - defensive
        results["warnings"].append(f"Correlation analysis failed: {exc}")

    # Return/Vol/PE 3D
    try:
        factors = run_factor_3d_analysis(positions)
        results["factors"] = factors
    except Exception as exc:  # pragma: no cover - defensive
        results["warnings"].append(f"Return/Vol/PE analysis failed: {exc}")

    # Portfolio vs Benchmarks
    try:
        benchmarks = run_benchmark_analysis(portfolio_series)
        results["benchmarks"] = benchmarks
    except Exception as exc:  # pragma: no cover - defensive
        results["warnings"].append(f"Benchmark comparison failed: {exc}")

    return results
