import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
import warnings
from portfolio.optimize import run_task4
from portfolio.back_test import backtest_strategy

warnings.filterwarnings("ignore")

# --------------------- CONFIG ---------------------
TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-07-01"
END_DATE = "2025-07-31"
FORECAST_MONTHS = 6
SEQ_LENGTH = 20
BACKTEST_START = "2024-08-01"
BACKTEST_END = "2025-07-31"
BENCHMARK_WEIGHTS = {"SPY": 0.6, "BND": 0.4}
# Define the folder path
figures_dir = Path("reports/figures")
figures_dir.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist
DATA_RAW_DIR = Path("data/raw")
# Create models directory if not exists
models_dir = Path("models")
models_dir.mkdir(parents=True, exist_ok=True)


# ----------------- DATA ACQUISITION -----------------
def load_local_parquet(ticker):
    parquet_path = DATA_RAW_DIR / f"{ticker}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Local parquet file not found for ticker {ticker} at {parquet_path}"
        )

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {ticker} parquet with columns: {df.columns.tolist()}")

    # Your parquet has MultiIndex columns:
    # level 0 = Price (Close, High...), level 1 = Ticker
    # Select only the columns for the given ticker from second level
    if isinstance(df.columns, pd.MultiIndex):
        # This extracts all columns under level 'Ticker' == ticker
        df = df.xs(ticker, axis=1, level="Ticker").copy()
    else:
        # fallback: flatten column names
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    # Rename columns to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]

    # If 'adj_close' missing but 'close' present, create adj_close as copy of close
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    return df


def download_data():
    print("Downloading data...")
    try:
        df = yf.download(
            TICKERS,
            start=START_DATE,
            end=END_DATE,
            group_by="ticker",
            auto_adjust=False,
            timeout=30,  # increase timeout
        )
        print("Download succeeded.")
    except Exception as e:
        print(f"Warning: Exception during yfinance download: {e}")
        df = None

    data_dict = {}
    required_cols = ["open", "high", "low", "close", "adj_close", "volume"]

    # If download completely failed, load all tickers locally
    if df is None or df.empty:
        print(
            "Download failed or empty, loading all tickers from local parquet files..."
        )
        for ticker in TICKERS:
            try:
                ticker_df = load_local_parquet(ticker)
                missing_cols = [
                    col for col in required_cols if col not in ticker_df.columns
                ]
                if missing_cols:
                    print(
                        f"Local parquet for {ticker} "
                        "missing columns {missing_cols}, skipping."
                    )
                    continue
                data_dict[ticker] = ticker_df[required_cols].copy()
                print(f"Loaded {ticker} from local parquet.")
            except Exception as e:
                print(f"Failed to load local parquet for {ticker}: {e}")
        if not data_dict:
            raise RuntimeError("No data loaded from download or local files. Exiting.")
        return data_dict

    # Otherwise, try to load each ticker from download with fallback to local
    for ticker in TICKERS:
        ticker_df = None
        if ticker in df.columns.levels[0]:
            temp_df = df[ticker].copy()
            temp_df.columns = temp_df.columns.str.lower().str.replace(" ", "_")
            missing_cols = [col for col in required_cols if col not in temp_df.columns]
            if not missing_cols:
                ticker_df = temp_df
            else:
                print(
                    f"Warning: Missing columns {missing_cols} for {ticker} "
                    "in downloaded data, falling back to local parquet."
                )
        else:
            print(f"{ticker} not found in downloaded data, trying local parquet.")

        if ticker_df is None:
            try:
                ticker_df = load_local_parquet(ticker)
                print(f"Loaded local parquet for {ticker}")
            except Exception as e:
                print(f"Error loading local parquet for {ticker}: {e}")
                continue

        missing_cols = [col for col in required_cols if col not in ticker_df.columns]
        if missing_cols:
            print(
                f"Error: Missing columns {missing_cols}"
                " in final data for {ticker}, skipping."
            )
            continue

        data_dict[ticker] = ticker_df[required_cols].copy()

    if not data_dict:
        raise RuntimeError(
            "Failed to load data for all tickers from download and local files."
        )

    return data_dict


# -------------- PREPROCESSING & FEATURES --------------
def preprocess_data(df):
    all_days = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(all_days)
    df["adj_close"].fillna(method="ffill", inplace=True)
    df["volume"].fillna(0, inplace=True)
    df["return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    df["volatility"] = df["return"].rolling(window=20).std() * np.sqrt(252)
    return df


# -------------- STATIONARITY TEST --------------
def adf_test(series, asset_name=""):
    result = adfuller(series.dropna())
    print(f"\nADF Test for {asset_name}:")
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print("  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.4f}")
    return result[1]


# -------------- ARIMA MODELING --------------
def train_arima(train_series):
    model_auto = pm.auto_arima(
        train_series.dropna(), seasonal=False, stepwise=True, suppress_warnings=True
    )
    order = model_auto.order
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    return model_fit, order


def arima_forecast(model_fit, steps, alpha=0.05):
    fc = model_fit.get_forecast(steps=steps)
    mean_forecast = fc.predicted_mean
    conf_int = fc.conf_int(alpha=alpha)
    return mean_forecast, conf_int


# -------------- LSTM MODELING --------------
def create_sequences(data, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_lstm(train_series, epochs=30, batch_size=16):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
    X_train, y_train = create_sequences(scaled)
    model = Sequential(
        [LSTM(50, activation="relu", input_shape=(SEQ_LENGTH, 1)), Dense(1)]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model, scaler


def lstm_predict(model, scaler, series):
    scaled = scaler.transform(series.values.reshape(-1, 1))
    X, y = create_sequences(scaled)
    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)
    y_true = scaler.inverse_transform(y)
    return pred.flatten(), y_true.flatten()


# -------------- METRICS --------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


# -------------- PORTFOLIO OPTIMIZATION --------------
def portfolio_optimization(expected_returns_vec, returns_df):
    cov_matrix = risk_models.sample_cov(returns_df)

    ef_max_sharpe = EfficientFrontier(expected_returns_vec, cov_matrix)
    weights_max_sharpe = ef_max_sharpe.max_sharpe()
    ef_max_sharpe.portfolio_performance(verbose=True)

    ef_min_vol = EfficientFrontier(expected_returns_vec, cov_matrix)
    weights_min_vol = ef_min_vol.min_volatility()
    ef_min_vol.portfolio_performance(verbose=True)

    return weights_max_sharpe, weights_min_vol


# -------------- BACKTESTING --------------
def backtest_portfolio(weights, price_data, start, end):
    adj_closes = pd.DataFrame(
        {t: df.loc[start:end, "adj_close"] for t, df in price_data.items()}
    )
    returns = adj_closes.pct_change().dropna()
    weights_series = pd.Series(weights)
    port_returns = returns.dot(weights_series)
    cum_returns = (1 + port_returns).cumprod()
    return cum_returns


# -------------- MAIN PIPELINE --------------
def main():
    print("Downloading data...")
    data = download_data()

    for ticker in TICKERS:
        if ticker not in data or data[ticker].empty:
            print(f"No data available for {ticker}, skipping analysis.")
            continue
        data[ticker] = preprocess_data(data[ticker])

    for ticker in TICKERS:
        if ticker not in data or data[ticker].empty:
            print(f"No data available for {ticker}, skipping analysis.")
            continue
        print(f"\n--- {ticker} ---")
        adf_test(data[ticker]["adj_close"], ticker + " Adj Close")
        adf_test(data[ticker]["return"], ticker + " Returns")

    train_end_date = "2023-12-31"
    test_start_date = "2024-01-01"
    forecast_steps = FORECAST_MONTHS * 21

    arima_forecasts = {}
    lstm_forecasts = {}
    metrics = {}

    for ticker in TICKERS:
        if ticker not in data or data[ticker].empty:
            print(f"No data available for {ticker}, skipping analysis.")
            continue
        print(f"\nTraining models for {ticker}...")

        series = data[ticker]["adj_close"]

        train = series[:train_end_date]
        test = series[test_start_date:]

        # ARIMA
        arima_model, order = train_arima(train)
        print(f"ARIMA order for {ticker}: {order}")

        steps_test = len(test)
        arima_pred_test, conf_test = arima_forecast(arima_model, steps_test)
        arima_pred_future, conf_future = arima_forecast(arima_model, forecast_steps)

        arima_mae, arima_rmse, arima_mape = evaluate(
            test.values, arima_pred_test.values
        )
        print(
            f"ARIMA {ticker} Test MAE: {arima_mae:.4f}, RMSE: "
            "{arima_rmse:.4f}, MAPE: {arima_mape:.2f}%"
        )

        arima_forecasts[ticker] = {
            "test_pred": arima_pred_test,
            "test_conf": conf_test,
            "future_pred": arima_pred_future,
            "future_conf": conf_future,
        }
        # Save ARIMA model
        arima_model_path = models_dir / f"{ticker}_arima.pkl"
        arima_model.save(arima_model_path)
        print(f"Saved ARIMA model for {ticker} at {arima_model_path}")

        # LSTM
        lstm_model, scaler = train_lstm(train)
        lstm_pred, y_true = lstm_predict(
            lstm_model, scaler, pd.concat([train[-SEQ_LENGTH:], test])
        )
        lstm_mae, lstm_rmse, lstm_mape = evaluate(y_true, lstm_pred)
        print(
            f"LSTM {ticker} Test MAE: {lstm_mae:.4f}, RMSE: "
            "{lstm_rmse:.4f}, MAPE: {lstm_mape:.2f}%"
        )

        # Save LSTM model to models folder
        lstm_model_path = models_dir / f"{ticker}_lstm_model.h5"
        lstm_model.save(lstm_model_path)
        print(f"LSTM model saved to {lstm_model_path}")

        # Save scaler with pickle
        scaler_path = models_dir / f"{ticker}_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")

        lstm_forecasts[ticker] = {
            "test_pred": lstm_pred,
            "test_true": y_true,
            "scaler": scaler,
            "model": lstm_model,
        }

        # Save LSTM model and scaler
        lstm_model_path = models_dir / f"{ticker}_lstm.h5"
        scaler_path = models_dir / f"{ticker}_scaler.pkl"

        metrics[ticker] = {
            "ARIMA": (arima_mae, arima_rmse, arima_mape),
            "LSTM": (lstm_mae, lstm_rmse, lstm_mape),
        }

        # Plot ARIMA vs LSTM on test set
        plt.figure(figsize=(12, 6))
        plt.plot(test.index, test.values, label="Actual")
        plt.plot(test.index, arima_pred_test, label="ARIMA Pred")
        plt.plot(test.index, lstm_pred, label="LSTM Pred")
        plt.title(f"{ticker} - ARIMA vs LSTM Forecast on Test Set")
        plt.xlabel("Date")
        plt.ylabel("Adjusted Close Price")
        plt.legend()
        plt.savefig(figures_dir / f"{ticker}_arima_vs_lstm_test.png")
        plt.close()

        # Plot 6-month ARIMA forecast with confidence interval
        future_dates = pd.bdate_range(
            test.index[-1], periods=forecast_steps + 1, freq="B"
        )[1:]
        plt.figure(figsize=(12, 6))
        plt.plot(series.index, series, label="Historical Adj Close")
        plt.plot(future_dates, arima_pred_future, label="ARIMA 6-month Forecast")
        plt.fill_between(
            future_dates,
            arima_forecasts[ticker]["future_conf"].iloc[:, 0],
            arima_forecasts[ticker]["future_conf"].iloc[:, 1],
            color="pink",
            alpha=0.3,
            label="Confidence Interval",
        )
        plt.title(f"{ticker} - ARIMA 6-Month Forecast with Confidence Interval")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig(figures_dir / f"{ticker}_arima_6m_forecast.png")
        plt.close()

    # Prepare returns dataframe only for tickers with 'return' column
    valid_tickers = [t for t in TICKERS if t in data and "return" in data[t].columns]
    if not valid_tickers:
        print(
            "No valid tickers with 'return' column "
            "found for portfolio optimization. Exiting."
        )
        return

    returns_df = pd.DataFrame(
        {ticker: data[ticker]["return"] for ticker in valid_tickers}
    ).dropna()

    # Calculate expected returns vector
    expected_returns_vec = pd.Series(dtype="float64", index=valid_tickers)

    if "TSLA" in valid_tickers and "TSLA" in arima_forecasts:
        tsla_forecast_returns = np.log(
            arima_forecasts["TSLA"]["future_pred"].values[1:]
            / arima_forecasts["TSLA"]["future_pred"].values[:-1]
        )
        expected_returns_vec["TSLA"] = tsla_forecast_returns.mean() * 252
    if "BND" in valid_tickers:
        expected_returns_vec["BND"] = returns_df["BND"].mean() * 252
    if "SPY" in valid_tickers:
        expected_returns_vec["SPY"] = returns_df["SPY"].mean() * 252

    print("\nExpected Annualized Returns:")
    print(expected_returns_vec)

    weights_max_sharpe, weights_min_vol = portfolio_optimization(
        expected_returns_vec, returns_df
    )

    print("\nMax Sharpe Portfolio Weights:")
    print(weights_max_sharpe)

    print("\nMin Volatility Portfolio Weights:")
    print(weights_min_vol)

    # Plot Efficient Frontier manually
    weights_max_sharpe, weights_min_vol = portfolio_optimization(
        expected_returns_vec, returns_df
    )

    # Calculate max possible return for safe upper bound
    # ef_max_ret = EfficientFrontier(
    #     expected_returns_vec, risk_models.sample_cov(returns_df)
    # )
    # Calculate the maximum possible return as the max expected return among assets
    max_return = expected_returns_vec.max()

    # Generate points along the efficient frontier safely
    risks = []
    returns = []
    for ret in np.linspace(
        expected_returns_vec.min(), max_return * 0.99, 50
    ):  # 0.99 to avoid boundary issues
        ef_temp = EfficientFrontier(
            expected_returns_vec, risk_models.sample_cov(returns_df)
        )
        ef_temp.efficient_return(target_return=ret)
        ret_, risk_, _ = ef_temp.portfolio_performance()
        returns.append(ret_)
        risks.append(risk_)

    plt.figure(figsize=(10, 7))
    plt.plot(risks, returns, "b-", label="Efficient Frontier")

    # Plot Max Sharpe point
    weights_arr = np.array(list(weights_max_sharpe.values()))
    cov = risk_models.sample_cov(returns_df).values
    sharpe_risk = np.sqrt(weights_arr.T.dot(cov).dot(weights_arr))
    sharpe_return = expected_returns_vec.dot(pd.Series(weights_max_sharpe))
    plt.scatter(
        sharpe_risk,
        sharpe_return,
        marker="*",
        color="r",
        s=200,
        label="Max Sharpe Portfolio",
    )

    plt.xlabel("Risk (Std Dev)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier with Max Sharpe Portfolio")
    plt.legend()
    plt.savefig(figures_dir / "efficient_frontier.png")
    plt.close()

    # Backtesting
    print("\nBacktesting portfolio vs benchmark...")

    portfolio_cum_returns = backtest_portfolio(
        weights_max_sharpe, data, BACKTEST_START, BACKTEST_END
    )
    benchmark_prices = {k: data[k] for k in BENCHMARK_WEIGHTS.keys() if k in data}
    benchmark_cum_returns = backtest_portfolio(
        BENCHMARK_WEIGHTS, benchmark_prices, BACKTEST_START, BACKTEST_END
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        portfolio_cum_returns.index, portfolio_cum_returns, label="Strategy Portfolio"
    )
    plt.plot(
        benchmark_cum_returns.index,
        benchmark_cum_returns,
        label="Benchmark (60% SPY / 40% BND)",
    )
    plt.title("Backtest Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.savefig(figures_dir / "backtest_cumulative_returns.png")
    plt.close()

    # Sharpe Ratio
    def sharpe_ratio(returns, risk_free_rate=0.0):
        return (returns.mean() * 252) / (returns.std() * np.sqrt(252))

    strat_returns = portfolio_cum_returns.pct_change().dropna()
    bench_returns = benchmark_cum_returns.pct_change().dropna()

    print(f"Strategy Portfolio Sharpe Ratio: {sharpe_ratio(strat_returns):.4f}")
    print(f"Benchmark Portfolio Sharpe Ratio: {sharpe_ratio(bench_returns):.4f}")

    # optimize
    results = run_task4(
        data=data, arima_forecasts=arima_forecasts, figures_dir=figures_dir
    )
    print(results["recommended"])

    optimized_weights = results["recommended"]["weights"]
    #  `data` and `optimized_weights` from Task 4):
    cum_returns, perf = backtest_strategy(data, optimized_weights)
    print("Performance Summary:", perf)
    print("Returns DF head:\n", cum_returns.head())
    print("Returns DF tail:\n", cum_returns.tail())

    opt_weights_vector = np.array(
        [optimized_weights.get(t, 0.0) for t in returns_df.columns]
    )
    bench_weights_vector = np.array(
        [BENCHMARK_WEIGHTS.get(t, 0.0) for t in returns_df.columns]
    )

    print("Optimized weights vector:", opt_weights_vector)
    print(
        "optimized_weights keys:", optimized_weights.keys()
    )  # This will show ticker keys correctly

    print("Benchmark weights vector:", bench_weights_vector)
    print("returns_df columns:", cum_returns.columns)
    print("optimized_weights keys:", optimized_weights.keys())


if __name__ == "__main__":
    main()
