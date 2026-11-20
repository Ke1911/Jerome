# src/models/backtest_deciles.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


USE_NEUTRALIZE = False
PRED_PATH   = "artifacts/test_predictions.parquet"  # must contain: permno, month_end-like, y_true-like, y_pred-like
PANEL_PATH  = "data/panel.parquet"                 # to fetch log_mktcap for neutralization
ARTIFACTS   = Path("artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

COST_PER_SIDE = 0.001  # 10 bps per side
MIN_CROSSSEC  = 50     # skip months with too few names

# ---------- helpers ----------

def load_predictions(path: str) -> pd.DataFrame:
    """
    Load predictions and normalize column names:
    - month_end: accepts ['month_end','date','month','monthend','t_month','t_idx']
    - y_pred:    accepts ['y_pred','pred','prediction','yhat','preds']
    - y_true:    accepts ['y_true','ret','target','y']
    - permno:    accepts ['permno','permno_x','permno_y']
    """
    df = pd.read_parquet(path)
    # Build lowercase index
    lower = {c.lower(): c for c in df.columns}

    def rename_first(candidates, new):
        for cand in candidates:
            if cand in lower:
                df.rename(columns={lower[cand]: new}, inplace=True)
                return True
        return False

    ok_date = rename_first(["month_end","date","month","monthend","t_month","t_idx"], "month_end")
    ok_pred = rename_first(["y_pred","pred","prediction","yhat","preds"], "y_pred")
    ok_true = rename_first(["y_true","ret","target","y"], "y_true")
    ok_perm = rename_first(["permno","permno_x","permno_y"], "permno")

    missing = [n for ok, n in [(ok_date,"month_end"),(ok_pred,"y_pred"),(ok_true,"y_true"),(ok_perm,"permno")] if not ok]
    if missing:
        raise KeyError(f"Missing required columns (after normalization): {missing}. Found columns: {list(df.columns)}")

    df["month_end"] = pd.to_datetime(df["month_end"])
    return df[["permno","month_end","y_true","y_pred"]]

def neutralize_by_size(df_pred: pd.DataFrame, panel_path: str) -> pd.DataFrame:
    """
    Merge log_mktcap from panel and do cross-sectional OLS: y_pred ~ 1 + log_mktcap.
    Use residuals (y_pred_n) as neutralized scores.
    """
    panel = pd.read_parquet(panel_path)[["permno","month_end","log_mktcap"]]
    panel["month_end"] = pd.to_datetime(panel["month_end"])
    df = df_pred.merge(panel, on=["permno","month_end"], how="left")

    def _fit_month(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) < MIN_CROSSSEC or g["log_mktcap"].isna().all():
            g["y_pred_n"] = g["y_pred"]
            return g
        x = g["log_mktcap"].fillna(g["log_mktcap"].median())
        X = np.c_[np.ones(len(x)), x.to_numpy()]     # intercept + size
        y = g["y_pred"].to_numpy()
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        g["y_pred_n"] = y - X @ beta
        return g

    out = df.groupby("month_end", group_keys=False).apply(_fit_month)
    out["y_pred_used"] = out["y_pred_n"].fillna(out["y_pred"])
    return out

def label_deciles(df_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Rank by y_pred_used each month and assign deciles 1..10.
    """
    def _by_month(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) < MIN_CROSSSEC:
            g["decile"] = np.nan
            return g
        g = g.sort_values("y_pred_used")
        g["decile"] = pd.qcut(
            g["y_pred_used"].rank(method="first"),
            10, labels=False, duplicates="drop"
        ) + 1
        return g

    return df_scores.groupby("month_end", group_keys=False).apply(_by_month)

def estimate_turnover(df_with_deciles: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate monthly turnover for D1 and D10 as the fraction of names that changed
    relative to the previous month; return a DataFrame with month_end and ls_turn (D10 + D1).
    """
    def _turnover_for_decile(dec: int) -> pd.DataFrame:
        subset = df_with_deciles[df_with_deciles["decile"] == dec][["month_end", "permno"]]
        subset = subset.sort_values("month_end")
        turns = []
        prev = set()
        for m, g in subset.groupby("month_end"):
            cur = set(g["permno"].tolist())
            if prev:
                changed = 1.0 - len(cur & prev) / max(len(cur | prev), 1)
                turns.append((m, changed))
            prev = cur
        return pd.DataFrame(turns, columns=["month_end", f"turn_D{dec}"])

    to_d10 = _turnover_for_decile(10)
    to_d1  = _turnover_for_decile(1)
    to = pd.merge(to_d10, to_d1, on="month_end", how="outer").fillna(0.0)
    to["ls_turn"] = to["turn_D10"] + to["turn_D1"]
    return to.sort_values("month_end")

# ---------- main backtest ----------

def decile_backtest(pred_path: str, panel_path: str):
    # 1) Load predictions (robust to column naming)
    pred = load_predictions(pred_path)

    # 2) Neutralize by size
    if USE_NEUTRALIZE:
        pred = neutralize_by_size(pred, panel_path)
        pred['y_pred_used'] = pred['y_pred_n'].fillna(pred['y_pred'])
    else:
        pred['y_pred_used'] = pred['y_pred']


    # 3) Assign deciles each month using neutralized scores
    pred = label_deciles(pred)
    pred = pred.dropna(subset=["decile"]).copy()
    pred["decile"] = pred["decile"].astype(int)

    # 4) Equal-weight realized decile returns each month (gross)
    decile_ret = (
        pred.groupby(["month_end","decile"])["y_true"]
        .mean()
        .unstack("decile")
        .sort_index()
    )
    decile_ret.columns = [f"D{int(c)}" for c in decile_ret.columns]

    # 5) Long–Short series (gross)
    ls = decile_ret["D10"].fillna(0.0) - decile_ret["D1"].fillna(0.0)

    # 6) Turnover & costs (10 bps per side on changes in each leg)
    to = estimate_turnover(pred)
    ls_net = ls.copy()
    common = ls_net.index.intersection(to["month_end"])
    cost = pd.Series(0.0, index=ls_net.index)
    cost.loc[common] = COST_PER_SIDE * to.set_index("month_end").loc[common, "ls_turn"]
    ls_net = ls_net - cost

    # 7) Stats
    def _stats(series: pd.Series, label: str):
        s = series.dropna()
        mean_ls = s.mean()
        std_ls  = s.std(ddof=1)
        sharpe_m = mean_ls / (std_ls + 1e-12)
        sharpe_a = sharpe_m * np.sqrt(12)
        hit_rate = (s > 0).mean()
        t_stat   = mean_ls / (std_ls / np.sqrt(len(s))) if len(s) > 1 else np.nan
        print(f"\n=== Long–Short {label} monthly stats ===")
        print(f"Mean: {mean_ls:.4f} | Std: {std_ls:.4f} | Sharpe(m): {sharpe_m:.2f} | Sharpe(ann): {sharpe_a:.2f}")
        print(f"Hit rate: {hit_rate:.2%} | Months: {len(s)} | t-stat: {t_stat:.2f}")

    _stats(ls, "GROSS")
    _stats(ls_net, "NET (costs)")

    # 8) Average decile returns (gross)
    avg_deciles = decile_ret.mean().to_frame("avg_ret").T
    print("\nAverage monthly return by decile (gross):")
    print(avg_deciles.to_string(index=False))

    # 9) Save tables
    decile_ret.to_parquet(ARTIFACTS / "decile_returns.parquet")
    ls.to_frame("ls_gross").to_parquet(ARTIFACTS / "ls_series_gross.parquet")
    ls_net.to_frame("ls_net").to_parquet(ARTIFACTS / "ls_series_net.parquet")

    # 10) Plot cumulative long–short (net)
    cum = (1.0 + ls_net.fillna(0.0)).cumprod()
    ax = cum.plot(title="Cumulative Long–Short (D10 - D1), Net of Costs", legend=False)
    ax.set_xlabel("Month"); ax.set_ylabel("Cumulative Growth (×)")
    plt.tight_layout()
    out_png = ARTIFACTS / "ls_cum.png"
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"\nSaved decile returns, LS series, and plot to '{ARTIFACTS.resolve()}'")
    print(f"Plot: {out_png.resolve()}")

def main():
    decile_backtest(PRED_PATH, PANEL_PATH)

if __name__ == "__main__":
    main()
