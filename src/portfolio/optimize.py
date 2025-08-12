"""
Task 4: Optimize Portfolio Based on Forecast (Standalone module)

Inputs expected:
- data: dict of pandas.DataFrame for tickers, each dataframe
must include a 'return' column (daily log returns)
        e.g. data['BND']['return'], data['SPY']['return'], data['TSLA']['return']
- arima_forecasts: dict with 'TSLA'
-> {'future_pred': pd.Series(...)} produced in Task2 (optional)
- figures_dir: Path where plots are saved (defaults to 'reports/figures')

Outputs:
- Saves plot 'efficient_frontier_task4.png' in figures_dir
- Returns dict with results for Min Vol and Max Sharpe
    portfolios and the full Monte Carlo results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize


def annualize_return_from_forecast(forecast_series, periods_per_year=252):
    """
    Calculate expected annual return (simple) from a forecasted price series:
      r_t = log(p_t / p_{t-1})
    Returns a scalar annualized expected return (arithmetic).
    """
    if forecast_series is None or len(forecast_series) < 2:
        raise ValueError(
            "forecast_series must be a pd.Series of forecasted prices with length >= 2"
        )
    logrets = np.log(forecast_series.values[1:] / forecast_series.values[:-1])
    mean_daily = np.nanmean(logrets)
    return mean_daily * periods_per_year  # approx annualized log-return


def annualize_historical_returns(returns_series, periods_per_year=252):
    """returns_series is daily simple or log returns
    (we'll treat as simple for expectation)"""
    return returns_series.mean() * periods_per_year


def compute_annual_cov(returns_df, periods_per_year=252):
    """returns_df: daily returns DataFrame aligned by date.
    Returns annualized covariance matrix."""
    daily_cov = returns_df.cov()  # sample covariance (daily)
    return daily_cov * periods_per_year


def portfolio_performance(weights, exp_returns, cov_matrix, risk_free=0.0):
    """
    weights: numpy array
    exp_returns: numpy array (annual)
    cov_matrix: annual covariance matrix
    returns: tuple (expected_return, volatility, sharpe)
    """
    ret = float(np.dot(weights, exp_returns))
    vol = float(np.sqrt(weights @ cov_matrix @ weights))
    sharpe = (ret - risk_free) / vol if vol != 0 else 0.0
    return ret, vol, sharpe


def min_variance_portfolio(exp_returns, cov_matrix, allow_short=False):
    n = len(exp_returns)
    x0 = np.repeat(1 / n, n)
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)

    def fun(x):
        return float(x @ cov_matrix @ x)

    res = minimize(fun, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        raise RuntimeError("Min variance optimization failed: " + str(res.message))
    w = res.x
    return w


def max_sharpe_portfolio(exp_returns, cov_matrix, risk_free=0.0, allow_short=False):
    # maximize (ret - rf)/vol -> minimize negative sharpe
    n = len(exp_returns)
    x0 = np.repeat(1 / n, n)
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)

    def neg_sharpe(x):
        ret = float(np.dot(x, exp_returns))
        vol = float(np.sqrt(x @ cov_matrix @ x))
        return -(ret - risk_free) / vol if vol != 0 else 1e6

    res = minimize(
        neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    if not res.success:
        raise RuntimeError("Max Sharpe optimization failed: " + str(res.message))
    return res.x


def monte_carlo_frontier(
    exp_returns, cov_matrix, n_portfolios=25000, allow_short=False, seed=42
):
    rng = np.random.default_rng(seed)
    n = len(exp_returns)
    results = []
    for _ in range(n_portfolios):
        if allow_short:
            # allow negative weights but still normalized to sum 1
            w = rng.normal(size=n)
            w = w / np.sum(w)
        else:
            # draw random weights from Dirichlet to ensure positivity and sum=1
            alpha = np.ones(n)
            w = rng.dirichlet(alpha)
        ret, vol, sharpe = portfolio_performance(w, exp_returns, cov_matrix)
        results.append((ret, vol, sharpe, w))
    # convert to DataFrame
    rets = np.array([r[0] for r in results])
    vols = np.array([r[1] for r in results])
    sharpes = np.array([r[2] for r in results])
    weights = np.array([r[3] for r in results])
    df = pd.DataFrame({"ret": rets, "vol": vols, "sharpe": sharpes})
    return df, weights


def run_task4(
    data,
    arima_forecasts=None,
    figures_dir=Path("reports/figures"),
    tickers=("TSLA", "BND", "SPY"),
    forecast_asset="TSLA",
    forecast_months=6,
    rf=0.0,
):
    """
    Main runner for Task 4 portfolio optimization.
    - data: dict of DataFrames with 'return' column (daily returns)
    - arima_forecasts: optional dict with 'TSLA' -> {'future_pred': pd.Series(...) }
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build returns_df aligned
    valid_tickers = [t for t in tickers if t in data and "return" in data[t].columns]
    if len(valid_tickers) < 2:
        raise ValueError(
            "Need at least 2 valid tickers with 'return' "
            "column for portfolio optimization."
        )
    returns_df = pd.DataFrame({t: data[t]["return"] for t in valid_tickers}).dropna()

    # 2) Expected returns vector
    exp_returns = {}
    # TSLA from forecast if available
    if (
        forecast_asset in valid_tickers
        and arima_forecasts
        and forecast_asset in arima_forecasts
    ):
        try:
            fc = arima_forecasts[forecast_asset]["future_pred"]
            tsla_ann = annualize_return_from_forecast(fc)
            exp_returns[forecast_asset] = tsla_ann
        except Exception as e:
            # fallback to historical mean if forecast parsing fails
            print(
                f"Warning: could not compute TSLA expected return from forecast:{e}."
                "Falling back to historical mean."
            )
            exp_returns[forecast_asset] = annualize_historical_returns(
                returns_df[forecast_asset]
            )
    else:
        # fallback
        if forecast_asset in valid_tickers:
            exp_returns[forecast_asset] = annualize_historical_returns(
                returns_df[forecast_asset]
            )
        else:
            raise ValueError(
                f"{forecast_asset} not present in data or missing forecast."
            )

    # BND and SPY from historical
    for t in valid_tickers:
        if t == forecast_asset:
            continue
        exp_returns[t] = annualize_historical_returns(returns_df[t])

    # Order them consistently
    ordered_tickers = list(exp_returns.keys())
    mu = np.array([exp_returns[t] for t in ordered_tickers])

    # 3) Covariance matrix (annualized)
    cov_matrix = compute_annual_cov(returns_df[ordered_tickers])

    # 4) Find Min Variance and Max Sharpe (no shorts)
    w_min_vol = min_variance_portfolio(mu, cov_matrix, allow_short=False)
    w_max_sharpe = max_sharpe_portfolio(mu, cov_matrix, risk_free=rf, allow_short=False)

    ret_min, vol_min, sharpe_min = portfolio_performance(w_min_vol, mu, cov_matrix, rf)
    ret_max, vol_max, sharpe_max = portfolio_performance(
        w_max_sharpe, mu, cov_matrix, rf
    )

    print("\n--- Optimization Results ---")
    print("Tickers (ordered):", ordered_tickers)
    print("Min Volatility Portfolio:")
    for t, w in zip(ordered_tickers, w_min_vol):
        print(f"  {t}: {w:.4f}")
    print(f"  Expected annual return: {ret_min:.4%}")
    print(f"  Annual volatility: {vol_min:.4%}")
    print(f"  Sharpe Ratio: {sharpe_min:.4f}")

    print("\nMax Sharpe Portfolio:")
    for t, w in zip(ordered_tickers, w_max_sharpe):
        print(f"  {t}: {w:.4f}")
    print(f"  Expected annual return: {ret_max:.4%}")
    print(f"  Annual volatility: {vol_max:.4%}")
    print(f"  Sharpe Ratio: {sharpe_max:.4f}")

    # 5) Monte Carlo frontier for plotting
    frontier_df, frontier_weights = monte_carlo_frontier(
        mu, cov_matrix, n_portfolios=25000
    )
    # select efficient frontier by minimum vol for a given return (approx)
    # For plotting we just scatter
    plt.figure(figsize=(10, 7))
    plt.scatter(
        frontier_df["vol"],
        frontier_df["ret"],
        c=frontier_df["sharpe"],
        cmap="viridis",
        s=6,
        alpha=0.6,
    )
    # Mark the two portfolios
    plt.scatter(
        [vol_min], [ret_min], c="red", marker="*", s=200, label="Min Vol Portfolio"
    )
    plt.scatter(
        [vol_max], [ret_max], c="green", marker="*", s=200, label="Max Sharpe Portfolio"
    )
    plt.xlabel("Annualized Volatility (Std Dev)")
    plt.ylabel("Expected Annual Return")
    plt.title("Efficient Frontier (Monte Carlo) — Task 4")
    plt.colorbar(label="Sharpe Ratio")
    plt.legend()
    plt.grid(alpha=0.3)
    plot_path = figures_dir / "efficient_frontier_task4.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nEfficient frontier plot saved to {plot_path}")

    # 6) Choose recommendation (policy)
    # Heuristic: if investor target is growth, choose
    # Max Sharpe; if risk-averse, choose Min Vol
    # We'll present both and choose recommended_portfolio
    # according to a neutral risk appetite: maximize Sharpe
    recommended_weights = w_max_sharpe
    recommended_metrics = {
        "expected_return": ret_max,
        "volatility": vol_max,
        "sharpe": sharpe_max,
    }
    recommended = dict(zip(ordered_tickers, recommended_weights))

    results = {
        "ordered_tickers": ordered_tickers,
        "exp_returns": dict(zip(ordered_tickers, mu)),
        "cov_matrix": cov_matrix,
        "min_vol": {
            "weights": dict(zip(ordered_tickers, w_min_vol)),
            "ret": ret_min,
            "vol": vol_min,
            "sharpe": sharpe_min,
        },
        "max_sharpe": {
            "weights": dict(zip(ordered_tickers, w_max_sharpe)),
            "ret": ret_max,
            "vol": vol_max,
            "sharpe": sharpe_max,
        },
        "monte_carlo": frontier_df,
        "recommended": {
            "weights": recommended,
            "metrics": recommended_metrics,
            "policy": "Max Sharpe (risk-adjusted growth)",
        },
        "plot_path": str(plot_path),
    }

    return results


# Example usage:
# results = run_task4(data=data,
# arima_forecasts=arima_forecasts, figures_dir=Path("reports/figures"))
# print(results['recommended'])
