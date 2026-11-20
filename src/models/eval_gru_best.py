# src/models/eval_gru_best.py
import json
import numpy as np
import torch
import torch.nn as nn

TEST_NPZ = "artifacts/gru_test.npz"
BEST_PARAMS_JSON = "artifacts/gru_best_params.json"
BEST_CKPT = "artifacts/gru_best.pt"
DEVICE = "cpu"  # use "cuda" only if you trained/plan to load on GPU

class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)         # (B, L, H)
        last = out[:, -1, :]         # (B, H)
        return self.head(last).squeeze(-1)

def main():
    # --- load best hyperparameters picked by Optuna ---
    with open(BEST_PARAMS_JSON, "r") as f:
        p = json.load(f)
    hidden   = int(p.get("hidden", 64))
    layers   = int(p.get("layers", 1))
    dropout  = float(p.get("dropout", 0.0))
    input_sz = 10  # number of features per timestep in your data

    # --- load test data ---
    d = np.load(TEST_NPZ)
    X = torch.from_numpy(d["X"]).to(DEVICE)  # shape: (N, 12, 10)
    y = torch.from_numpy(d["y"]).to(DEVICE)  # shape: (N,)

    # --- build model & load weights ---
    model = GRURegressor(input_sz, hidden, num_layers=layers, dropout=dropout).to(DEVICE)
    state = torch.load(BEST_CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # --- predict ---
    with torch.no_grad():
        preds = model(X).cpu().numpy()
    y_true = y.cpu().numpy()

    # --- metrics ---
    mse = float(np.mean((preds - y_true) ** 2))
    corr = float(np.corrcoef(preds, y_true)[0, 1])  # Pearson IC

    print(f"Test MSE: {mse:.6f} | Pearson corr (IC): {corr:.4f}")

    # Save for any later backtests/plots
    np.savez_compressed("artifacts/gru_best_test_preds.npz", preds=preds, y=y_true)
    print("Saved preds to artifacts/gru_best_test_preds.npz")

if __name__ == "__main__":
    main()
