import yfinance as yf
import os

# Create directory if it doesn't exist
os.makedirs("../../data/raw", exist_ok=True)

tickers = ["TSLA", "BND", "SPY"]
start_date = "2015-07-01"
end_date = "2025-07-31"

for ticker in tickers:
    print(f"Downloading {ticker} data...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df.index.name = "Date"  # ensure index has a name for parquet
    filepath = f"data/raw/{ticker}.parquet"
    df.to_parquet(filepath)
    print(f"Saved {ticker} data to {filepath}")

print("All data downloaded and saved as parquet files.")
