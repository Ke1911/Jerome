import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---------- Config ----------
TRAIN = "artifacts/gru_train.npz"
VAL   = "artifacts/gru_val.npz"
INPUT_SIZE = 10
SEQ_LEN    = 12
HIDDEN_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
DEVICE = "cpu"

# ---------- Load Data ----------
def load_npz(path):
    d = np.load(path)
    X = torch.tensor(d["X"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.float32)
    return X, y

Xtr, ytr = load_npz(TRAIN)
Xva, yva = load_npz(VAL)

train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=BATCH_SIZE, shuffle=False)

# ---------- Model ----------
class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

model = GRURegressor(INPUT_SIZE, HIDDEN_SIZE).to(DEVICE)
criterion = torch.nn.HuberLoss(delta=0.02)          # instead of MSELoss
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # right after loss.backward()


# ---------- Training Loop ----------
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(yb)
    train_loss /= len(train_loader.dataset)

    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            val_loss += criterion(preds, yb).item() * len(yb)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch:02d}: train {train_loss:.6f}, val {val_loss:.6f}")

torch.save(model.state_dict(), "artifacts/gru_baseline.pt")
print("✅ Training complete and model saved to artifacts/gru_baseline.pt")
