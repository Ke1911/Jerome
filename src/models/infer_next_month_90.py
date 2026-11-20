import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PANEL  = "data/panel_ext.parquet"           # 90-day lag panel
CKPT   = "artifacts/gru_best.pt"        # adjust if different
PARAMS = "artifacts/gru_best_params.json"  # try to load; fallback defaults below
OUTDIR = Path("artifacts")

FEATURES = ['log_mktcap','ret_12_2','ret_6_2','ret_3_2','ret_1','BM','OP','LEV','vol_3m','turnover']
SEQ_LEN = 12

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers, batch_first=True,
                          dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out,_ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

def load_params():
    try:
        with open(PARAMS, "r") as f:
            p = json.load(f)
        return int(p.get("hidden", 32)), int(p.get("layers", 2)), float(p.get("dropout", 0.1))
    except Exception:
        # fallback to your best trial
        return 32, 2, 0.1

def build_latest_sequences(df: pd.DataFrame, L=12):
    df = df.sort_values(["permno","month_end"]).copy()
    for c in FEATURES:
        df[c] = df[c].astype("float32").fillna(0.0)
    last_m = pd.to_datetime(df["month_end"]).max()
    window_start = last_m - pd.offsets.MonthEnd(L-1)
    dfw = df[(df["month_end"]>=window_start) & (df["month_end"]<=last_m)].copy()
    counts = dfw.groupby("permno")["month_end"].nunique()
    keep = counts[counts==L].index
    dfw = dfw[dfw["permno"].isin(keep)]
    X_list, permnos = [], []
    for permno, g in dfw.groupby("permno"):
        g = g.sort_values("month_end")
        X_list.append(g[FEATURES].to_numpy(dtype=np.float32))
        permnos.append(permno)
    if not X_list:
        return np.zeros((0,L,len(FEATURES)), dtype=np.float32), np.array([], dtype=np.int64), last_m
    X = np.stack(X_list, axis=0).astype(np.float32)
    return X, np.array(permnos, dtype=np.int64), last_m

def enrich_with_names(df: pd.DataFrame, db=None) -> pd.DataFrame:
    """
    Add ticker/comnam from CRSP msenames in a point-in-time way:
    - left-merge by permno
    - keep only name rows active on month_end (namedt <= month_end <= nameendt*)
    - within each (permno, month_end) pick the latest namedt
    """
    try:
        if db is None:
            import wrds
            db = wrds.Connection()

        names = db.raw_sql(
            """
            select permno, namedt, nameendt, comnam, ticker
            from crsp.msenames
            """,
            date_cols=["namedt","nameendt"]
        )

        # dtype hygiene
        out = df.copy()
        out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
        names["permno"] = pd.to_numeric(names["permno"], errors="coerce").astype("Int64")
        out["month_end"] = pd.to_datetime(out["month_end"])
        names["namedt"] = pd.to_datetime(names["namedt"])
        names["nameendt"] = pd.to_datetime(names["nameendt"]).fillna(pd.Timestamp("2100-01-01"))

        # explode via left-merge, then filter to active names on month_end
        m = out.merge(names, on="permno", how="left")
        active = (m["namedt"] <= m["month_end"]) & (m["month_end"] <= m["nameendt"])
        m = m[active].copy()

        if m.empty:
            # No matches; return original df untouched
            out["permno"] = out["permno"].astype("int64", errors="ignore")
            return out

        # within each (permno, month_end) take the record with the latest namedt
        m = m.sort_values(["permno","month_end","namedt"])
        best = m.groupby(["permno","month_end"], as_index=False).tail(1)

        # keep base columns + name fields
        enriched = df.merge(
            best[["permno","month_end","ticker","comnam"]],
            on=["permno","month_end"], how="left"
        )
        # cast back if desired
        enriched["permno"] = pd.to_numeric(enriched["permno"], errors="ignore")
        return enriched

    except Exception as e:
        print(f"[WARN] Could not enrich with names: {e}")
        return df




def main():
    hidden, layers, dropout = load_params()
    df = pd.read_parquet(PANEL); df["month_end"] = pd.to_datetime(df["month_end"])
    X, permno, last_m = build_latest_sequences(df, L=SEQ_LEN)
    print(f"Latest month in panel: {last_m.date()} | Eligible names: {len(permno)}")
    if len(permno)==0:
        print("No eligible sequences with full 12-month window."); return

    device = "cpu"
    model = GRUModel(len(FEATURES), hidden, layers, dropout).to(device)

    state = torch.load(CKPT, map_location=device)
    # remap old names -> new names if needed
    if "head.weight" in state:
        state["fc.weight"]  = state.pop("head.weight")
        state["fc.bias"]    = state.pop("head.bias")
    # load (allow missing/extra just in case)
    model.load_state_dict(state, strict=False)
    model.eval()

    dl = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=256)
    preds = []
    with torch.no_grad():
        for (xb,) in dl:
            preds.append(model(xb.to(device)).cpu().numpy())
    preds = np.concatenate(preds)

    next_m = (pd.to_datetime(last_m) + pd.offsets.MonthEnd(1))
    out = pd.DataFrame({"permno": permno, "month_end": next_m, "y_pred": preds}).sort_values("y_pred", ascending=False)

    # Optional: enrich with company names & tickers
    try:
        import wrds
        db = wrds.Connection()
        out = enrich_with_names(out, db)
    except Exception as e:
        print(f"[WARN] Name enrichment skipped: {e}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"picks_next_90lagyf_{next_m.strftime('%Y-%m-%d')}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Saved next-month ranking to {out_path} (rows={len(out):,})")

    print("\nTop 20 LONG candidates:")
    print(out.head(20).to_string(index=False))
    print("\nTop 20 SHORT candidates:")
    print(out.tail(20).sort_values('y_pred').to_string(index=False))

if __name__ == "__main__":
    main()
