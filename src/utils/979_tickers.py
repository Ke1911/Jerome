import pandas as pd
snap = pd.read_parquet("artifacts/permno_ticker_snapshot_2024-12.parquet")
tickers = snap["ticker"].dropna().unique()
pd.DataFrame({"Ticker": tickers}).to_csv("artifacts/tickers_979.csv", index=False)
