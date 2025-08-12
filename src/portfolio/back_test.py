import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def backtest_strategy(
    data,  # dict of DataFrames with 'return' column keyed by tickers
    optimized_weights,  # dict from Task 4, can contain nested 'weights'
    backtest_start="2024-08-01",
    backtest_end="2025-07-31",
    benchmark_weights={"SPY": 0.6, "BND": 0.4},
    risk_free=0.0,
    tickers=["TSLA", "BND", "SPY"],
):
    """
    Backtest the fixed portfolio vs. benchmark over a specified date range.

    Parameters:
    - data: dict of pd.DataFrame with 'return' columns for each ticker.
    - optimized_weights: dict from Task 4, can contain nested 'weights'.
    - backtest_start, backtest_end: strings for date slicing.
    - benchmark_weights: fixed benchmark weights (default 60% SPY, 40% BND).
    - risk_free: risk-free rate for Sharpe ratio calc (annual).
    - tickers: list of all tickers considered.

    Returns:
    - DataFrame with cumulative returns for strategy and benchmark.
    - dict with performance metrics (total return, annual Sharpe) for both.
    """

    # --- Extract actual weights dict if nested under 'weights' ---
    if isinstance(optimized_weights, dict) and "weights" in optimized_weights:
        opt_weights_dict = optimized_weights["weights"]
    else:
        opt_weights_dict = optimized_weights

    # --- Prepare ticker-level daily returns DataFrame ---
    valid_tickers = [t for t in tickers if t in data and "return" in data[t].columns]
    returns_df = pd.DataFrame({t: data[t]["return"] for t in valid_tickers})
    returns_df = returns_df.loc[backtest_start:backtest_end].dropna()

    # Debug: See what's being used
    print("Backtest period:", returns_df.index.min(), "to", returns_df.index.max())
    print("Valid tickers in backtest:", list(returns_df.columns))
    print(
        "Optimized weights being applied:",
        {t: opt_weights_dict.get(t, 0.0) for t in returns_df.columns},
    )

    # --- Build aligned weight vectors ---
    opt_weights_vector = np.array(
        [opt_weights_dict.get(t, 0.0) for t in returns_df.columns]
    )
    bench_weights_vector = np.array(
        [benchmark_weights.get(t, 0.0) for t in returns_df.columns]
    )

    # Prevent all-zero weights issue
    if np.isclose(opt_weights_vector.sum(), 0):
        raise ValueError("Optimized weights sum to 0. Check ticker names and mapping.")

    # --- Calculate daily portfolio returns ---
    strategy_returns = returns_df @ opt_weights_vector
    benchmark_returns = returns_df @ bench_weights_vector

    # --- Calculate cumulative returns (growth curves) ---
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()

    # --- Performance metrics ---
    trading_days = 252
    total_return_strategy = strategy_cum.iloc[-1] - 1
    total_return_benchmark = benchmark_cum.iloc[-1] - 1

    sharpe_strategy = (strategy_returns.mean() * trading_days - risk_free) / (
        strategy_returns.std() * np.sqrt(trading_days)
    )
    sharpe_benchmark = (benchmark_returns.mean() * trading_days - risk_free) / (
        benchmark_returns.std() * np.sqrt(trading_days)
    )

    # --- Plot cumulative returns ---
    plt.figure(figsize=(10, 6))
    plt.plot(
        strategy_cum.index, strategy_cum - 1, label="Strategy Portfolio", color="blue"
    )
    plt.plot(
        benchmark_cum.index,
        benchmark_cum - 1,
        label="Benchmark (60% SPY / 40% BND)",
        color="orange",
    )
    plt.title("Backtest: Cumulative Returns Comparison")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Summary metrics ---
    performance = {
        "strategy": {
            "total_return": total_return_strategy,
            "annual_sharpe": sharpe_strategy,
        },
        "benchmark": {
            "total_return": total_return_benchmark,
            "annual_sharpe": sharpe_benchmark,
        },
    }

    cum_returns_df = pd.DataFrame(
        {"strategy": strategy_cum - 1, "benchmark": benchmark_cum - 1}
    )

    return cum_returns_df, performance
