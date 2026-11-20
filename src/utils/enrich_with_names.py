# src/utils/enrich_with_names.py
import pandas as pd
import wrds
from pathlib import Path

IN  = Path("artifacts/picks_next_90lag_2025-01-31.parquet")         # <= change if needed
OUT = Path("artifacts/picks_next_90lag_2025-01-31_named.parquet")   # <= change if needed

def main():
    print(f"Loading {IN} ...")
    preds = pd.read_parquet(IN)
    preds["permno"]    = pd.to_numeric(preds["permno"], errors="coerce").astype("Int64")
    preds["month_end"] = pd.to_datetime(preds["month_end"])

    print("Connecting to WRDS and loading CRSP.msenames ...")
    db = wrds.Connection()
    names = db.raw_sql("""
        select permno, namedt, nameendt, comnam, ticker
        from crsp.msenames
    """, date_cols=["namedt","nameendt"])

    # dtype hygiene
    names["permno"]  = pd.to_numeric(names["permno"], errors="coerce").astype("Int64")
    names["namedt"]  = pd.to_datetime(names["namedt"])
    names["nameendt"]= pd.to_datetime(names["nameendt"]).fillna(pd.Timestamp("2100-01-01"))

    # 1) Explode via left merge on permno (no asof)
    m = preds.merge(names, on="permno", how="left")

    # 2) Keep only name rows active at month_end: namedt <= month_end <= nameendt
    active = (m["namedt"] <= m["month_end"]) & (m["month_end"] <= m["nameendt"])
    m = m.loc[active].copy()

    if m.empty:
        print("No active name records matched; writing original file unchanged (no names).")
        preds.to_parquet(OUT, index=False)
        return

    # 3) Within each (permno, month_end) pick the most recent namedt
    m = m.sort_values(["permno","month_end","namedt"])
    best = m.groupby(["permno","month_end"], as_index=False).tail(1)

    # 4) Keep base cols + ticker/comnam
    out = preds.merge(
        best[["permno","month_end","ticker","comnam"]],
        on=["permno","month_end"], how="left"
    )

    # Optional cast back
    out["permno"] = pd.to_numeric(out["permno"], errors="ignore")

    out.to_parquet(OUT, index=False)
    print(f"Saved enriched file -> {OUT}  (rows={len(out):,})")

    # Pretty preview: top/bottom 20 by y_pred with names
    print("\nTop 20 LONG candidates (with names):")
    print(out.sort_values("y_pred", ascending=False)[["permno","ticker","comnam","y_pred"]].head(20).to_string(index=False))
    print("\nTop 20 SHORT candidates (with names):")
    print(out.sort_values("y_pred")[["permno","ticker","comnam","y_pred"]].head(20).to_string(index=False))

if __name__ == "__main__":
    main()
