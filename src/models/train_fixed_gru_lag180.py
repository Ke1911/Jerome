import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import pandas as pd

# ---------- config ----------
TRAIN = "artifacts/lag180/gru_train.npz"
VAL   = "artifacts/lag180/gru_val.npz"
TEST  = "artifacts/lag180/gru_test.npz"
IDX   = "artifacts/lag180/test_index.parquet"
OUT   = Path("artifacts/lag180/gru_lag180_test_preds.parquet")

params = {
  "hidden": 32,
  "layers": 2,
  "dropout": 0.1,
  "batch_size": 128,
  "lr": 0.0002189906487353606,
  "weight_decay": 1.1697520298381522e-06,
  "clip_norm": 1.5,
  "epochs": 14
}

# ---------- GRU model ----------
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

# ---------- helpers ----------
def load_npz(path):
    d = np.load(path)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.float32)
    return TensorDataset(X, y)

# ---------- train ----------
def train_model():
    train_ds = load_npz(TRAIN)
    val_ds   = load_npz(VAL)
    test_ds  = load_npz(TEST)

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=params["batch_size"])
    test_loader  = DataLoader(test_ds, batch_size=params["batch_size"])

    model = GRUModel(10, params["hidden"], params["layers"], params["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.MSELoss()

    for epoch in range(params["epochs"]):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["clip_norm"])
            opt.step()
            total_loss += loss.item() * len(xb)
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(xb)
        print(f"Epoch {epoch+1:02d}: train {total_loss/len(train_ds):.6f}, val {val_loss/len(val_ds):.6f}")

    # ---------- test ----------
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb).numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    # save parquet with permno & month_end
    idx = pd.read_parquet(IDX)
    out_df = pd.DataFrame({
        "permno": idx["permno"].to_numpy(),
        "month_end": pd.to_datetime(idx["month_end"]),
        "y_true": trues,
        "y_pred": preds
    })
    out_df.to_parquet(OUT, index=False)
    print(f"Saved {OUT} with shape {out_df.shape}")

if __name__ == "__main__":
    train_model()
