import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

from frontend_analytics import parse_manual_positions, run_full_analysis
from data_pipeline import load_positions

app = Flask(__name__)
app.secret_key = "horizon-analytics-secret"

# Simple in-memory cache for the most recent analysis
analysis_cache = {}


def combine_positions(manual_df: pd.DataFrame, upload_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge manual and uploaded portfolios and re-run the loader to clean them.
    """
    frames = []
    if manual_df is not None and not manual_df.empty:
        frames.append(manual_df)
    if upload_df is not None and not upload_df.empty:
        frames.append(upload_df)
    if not frames:
        return pd.DataFrame(columns=["ticker", "shares"])
    combined = pd.concat(frames, ignore_index=True)
    return load_positions(combined)


def get_results_or_redirect():
    results = analysis_cache.get("results")
    if not results:
        flash("Submit a portfolio on the Home page first.", "error")
        return None
    return results


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("portfolio_csv")
        upload_df = None
        if file and file.filename:
            try:
                upload_df = pd.read_csv(file)
            except Exception as exc:
                flash(f"Could not read CSV: {exc}", "error")
                return redirect(url_for("index"))

        raw_tickers = request.form.getlist("ticker")
        raw_shares = request.form.getlist("shares")
        paired_rows = [
            (t.strip().upper(), s.strip())
            for t, s in zip(raw_tickers, raw_shares)
            if t.strip() or s.strip()
        ]
        tickers = [t for t, _ in paired_rows]
        shares = [s for _, s in paired_rows]
        manual_df = parse_manual_positions(tickers, shares) if paired_rows else pd.DataFrame()

        positions_df = combine_positions(manual_df, upload_df)
        if positions_df.empty:
            flash("Please provide at least one valid ticker and share amount.", "error")
            return redirect(url_for("index"))

        try:
            results = run_full_analysis(positions_df)
            analysis_cache["results"] = results
            flash("Analysis complete. Navigate using the bar above to view results.", "success")
            return redirect(url_for("index"))
        except Exception as exc:  # pragma: no cover - surface errors to user
            flash(f"Analysis failed: {exc}", "error")
            return redirect(url_for("index"))

    results = analysis_cache.get("results")
    return render_template("index.html", results=results, active_page="home")


@app.route("/regimes")
def hmm_regimes():
    results = get_results_or_redirect()
    if not results:
        return redirect(url_for("index"))
    return render_template(
        "hmm.html",
        hmm=results.get("hmm"),
        benchmarks=results.get("benchmarks"),
        active_page="hmm"
    )


@app.route("/var")
def var_view():
    results = get_results_or_redirect()
    if not results:
        return redirect(url_for("index"))
    return render_template("var.html", var=results.get("var"), active_page="var")


@app.route("/monte-carlo")
def monte_carlo_view():
    results = get_results_or_redirect()
    if not results:
        return redirect(url_for("index"))
    return render_template(
        "monte_carlo.html",
        mc=results.get("monte_carlo"),
        latest_value=results.get("latest_value"),
        active_page="monte",
    )


@app.route("/correlation")
def correlation_view():
    results = get_results_or_redirect()
    if not results:
        return redirect(url_for("index"))
    return render_template("correlation.html", correlation=results.get("correlation"), active_page="correlation")


@app.route("/factors")
def factor_view():
    results = get_results_or_redirect()
    if not results:
        return redirect(url_for("index"))
    return render_template("factors.html", factors=results.get("factors"), active_page="factors")


if __name__ == "__main__":
    app.run(debug=True)
