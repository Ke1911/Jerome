# src/utils/export_tickers_for_ciq.py
from pathlib import Path
import pandas as pd

SNAP = Path("artifacts/permno_ticker_snapshot_2024-12.parquet")
OUT  = Path("artifacts/ciq_ticker_list.csv")

snap = pd.read_parquet(SNAP)
tickers = snap["ticker"].dropna().drop_duplicates().sort_values()
tickers.to_frame(name="ticker").to_csv(OUT, index=False)
print(f"Saved {OUT} with {len(tickers):,} tickers")
