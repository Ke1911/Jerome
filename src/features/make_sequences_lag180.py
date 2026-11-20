from pathlib import Path
import numpy as np
import pandas as pd



PANEL = Path("data/panel_lag180.parquet")
OUT = Path("artifacts/lag180")
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ['log_mktcap','ret_12_2','ret_6_2','ret_3_2','ret_1','BM','OP','LEV','vol_3m','turnover']
SEQ_LEN = 12
HORIZON = 1  # predict next month

# calendar splits (by TARGET month)
TRAIN_END = pd.Timestamp("2016-12-31")
VAL_END   = pd.Timestamp("2020-12-31")
# TEST is everything after VAL_END

def build_samples(df: pd.DataFrame, features, L=12, horizon=1):
    """
    Returns:
      X_seq: (N, L, F) float32
      X_flat: (N, L*F) float32
      y: (N,) float32
      t_idx: pd.Series target month_end (datetime64[ns])
      permnos: pd.Series permno per sample (int64)
    """
    X_seq_list, X_flat_list, y_list, t_idx, permnos = [], [], [], [], []

    df = df.sort_values(["permno", "month_end"]).copy()
    # fill remaining NaNs (features were z-scored per month, so 0 ~ mean)
    for col in features:
        df[col] = df[col].astype("float32").fillna(0.0)
    df["ret"] = df["ret"].astype("float32")

    for permno, g in df.groupby("permno", sort=True):
        g = g.sort_values("month_end")
        vals   = g[features].to_numpy(dtype=np.float32)
        ret    = g["ret"].to_numpy(dtype=np.float32)
        months = g["month_end"].to_numpy()

        # window end at t -> uses [t-L+1 .. t] to predict ret at t+1
        for t in range(L-1, len(g)-horizon):
            Xw = vals[t-L+1:t+1]               # (L,F)
            yw = ret[t + horizon]              # scalar next-month return
            tgt_month = months[t + horizon]    # target timestamp

            if np.isnan(yw):
                continue

            X_seq_list.append(Xw)
            X_flat_list.append(Xw.reshape(-1))  # (L*F,)
            y_list.append(yw)
            t_idx.append(tgt_month)
            permnos.append(permno)

    if X_seq_list:
        X_seq  = np.stack(X_seq_list, axis=0).astype(np.float32)
        X_flat = np.stack(X_flat_list, axis=0).astype(np.float32)
    else:
        X_seq  = np.zeros((0, L, len(features)), dtype=np.float32)
        X_flat = np.zeros((0, L*len(features)), dtype=np.float32)

    y      = np.array(y_list, dtype=np.float32)
    t_idx  = pd.to_datetime(pd.Series(t_idx))
    permnos= pd.Series(permnos, dtype="int64")
    return X_seq, X_flat, y, t_idx, permnos

def time_split_and_save(X_seq, X_flat, y, t_idx, permnos):
    # masks by TARGET month
    train_mask = (t_idx <= TRAIN_END)
    val_mask   = (t_idx > TRAIN_END) & (t_idx <= VAL_END)
    test_mask  = (t_idx > VAL_END)

    def save_split(name, mask):
        np.savez_compressed(OUT / f"gru_{name}.npz",      X=X_seq[mask],  y=y[mask])
        np.savez_compressed(OUT / f"baseline_{name}.npz", X=X_flat[mask], y=y[mask])
        # index file for backtests:
        idx = pd.DataFrame({
            "permno":    permnos[mask].to_numpy(),
            "month_end": t_idx[mask].to_numpy()
        })
        idx.to_parquet(OUT / f"{name}_index.parquet", index=False)
        print(f"Saved {name}: GRU X {X_seq[mask].shape}, y {y[mask].shape} | Baseline X {X_flat[mask].shape} | idx {len(idx)}")

    save_split("train", train_mask)
    save_split("val",   val_mask)
    save_split("test",  test_mask)

def main():
    print("Loading panel…")
    df = pd.read_parquet(PANEL)

    # keep only needed columns
    need = set(["permno","month_end","ret"] + FEATURES)
    df = df[[c for c in df.columns if c in need]].copy()
    df["month_end"] = pd.to_datetime(df["month_end"])

    print("Building sequences…")
    X_seq, X_flat, y, t_idx, permnos = build_samples(df, FEATURES, L=SEQ_LEN, horizon=HORIZON)
    print(f"Built: X_seq {X_seq.shape}, X_flat {X_flat.shape}, y {y.shape}")

    print("Splitting by calendar and saving…")
    time_split_and_save(X_seq, X_flat, y, t_idx, permnos)

if __name__ == "__main__":
    main()
