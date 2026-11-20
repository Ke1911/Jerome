# src/utils/snapshot_universe_2024_dec.py
from pathlib import Path
import pandas as pd
import wrds



import wrds
db = wrds.Connection(wrds_username="your_wrds_username")

PANEL = Path("data/panel.parquet")
OUT   = Path("artifacts/permno_ticker_snapshot_2024-12.parquet")

df = pd.read_parquet(PANEL)
df["month_end"] = pd.to_datetime(df["month_end"])
last_m = df["month_end"].max()           # should be 2024-12-31
uni = df[df["month_end"]==last_m][["permno"]].drop_duplicates()

db = wrds.Connection()
names = db.raw_sql("""
    select permno, namedt, nameendt, ticker, comnam
    from crsp.msenames
""", date_cols=["namedt","nameendt"])

snap = (uni.merge(names, on="permno", how="left")
            .assign(last_m=last_m)
            .loc[lambda x: x["namedt"].notna() & (x["namedt"] <= x["last_m"])]
            .sort_values(["permno","namedt"])
            .groupby("permno", as_index=False).tail(1)
            [["permno","ticker","comnam"]]
            .dropna(subset=["ticker"]))

snap.to_parquet(OUT, index=False)
print(f"Saved {OUT} with {len(snap):,} mappings")
