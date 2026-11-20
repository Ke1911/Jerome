import numpy as np, pandas as pd
IDX = "artifacts/test_index.parquet"
PRED_NPZ = "artifacts/gru_best_test_preds.npz"  # from eval_gru_best.py
OUT = "artifacts/test_predictions.parquet"

def main():
    idx = pd.read_parquet(IDX)
    d = np.load(PRED_NPZ)
    preds, y = d["preds"], d["y"]
    assert len(idx)==len(preds)==len(y), "Length mismatch; rebuild index via make_sequences."
    df = idx.copy()
    df["y_true"] = y
    df["y_pred"] = preds
    df.to_parquet(OUT, index=False)
    print(f"Saved {OUT} with {len(df):,} rows")

if __name__ == "__main__":
    main()
