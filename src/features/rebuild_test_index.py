from pathlib import Path
import numpy as np, pandas as pd

PANEL = Path("data/panel.parquet")
OUT = Path("artifacts"); OUT.mkdir(exist_ok=True, parents=True)

FEATURES = ['log_mktcap','ret_12_2','ret_6_2','ret_3_2','ret_1','BM','OP','LEV','vol_3m','turnover']
L, H = 12, 1  # seq len, horizon
VAL_END = pd.Timestamp("2020-12-31")

def main():
    df = pd.read_parquet(PANEL)
    df = df[['permno','month_end','ret'] + FEATURES].copy()
    df['month_end'] = pd.to_datetime(df['month_end'])
    # fill remaining NaNs in features (0 ≈ monthly z-score mean)
    for c in FEATURES: df[c] = df[c].fillna(0.0).astype('float32')
    df = df.sort_values(['permno','month_end'])

    t_months, permnos, y_list = [], [], []
    for permno, g in df.groupby('permno'):
        g = g.sort_values('month_end')
        ret = g['ret'].to_numpy()
        months = g['month_end'].to_numpy()
        T = len(g)
        for t in range(L-1, T-H):
            tgt_month = months[t+H]
            y = ret[t+H]
            if np.isnan(y): continue
            t_months.append(tgt_month)
            permnos.append(permno)
            y_list.append(y)

    t_idx = pd.to_datetime(pd.Series(t_months))
    perm = pd.Series(permnos, dtype='int64')
    y = np.array(y_list, dtype='float32')

    test_mask = (t_idx > VAL_END)
    idx = pd.DataFrame({'permno': perm[test_mask].to_numpy(),
                        'month_end': t_idx[test_mask].to_numpy()})
    idx.to_parquet(OUT/'test_index.parquet', index=False)
    np.savez_compressed(OUT/'_tmp_test_y.npz', y=y[test_mask])
    print(f"Saved {OUT/'test_index.parquet'} with {len(idx):,} rows")

if __name__ == "__main__":
    main()
