import os, time, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import optuna

# --- keep CPU calm ---
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)

TRAIN="artifacts/gru_train.npz"; VAL="artifacts/gru_val.npz"; TEST="artifacts/gru_test.npz"
INPUT_SIZE=10; DEVICE="cpu"
MAX_EPOCHS=15; PATIENCE=2
N_TRIALS=8  # small
SEED=1337
torch.manual_seed(SEED); np.random.seed(SEED)

def load_npz(p):
    d=np.load(p); return torch.from_numpy(d["X"]), torch.from_numpy(d["y"])
Xtr,ytr=load_npz(TRAIN); Xva,yva=load_npz(VAL); Xte,yte=load_npz(TEST)

class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Linear(hidden_size,1)
    def forward(self,x):
        out,_=self.gru(x); return self.head(out[:, -1, :]).squeeze(-1)

def make_loaders(bs):
    tr=TensorDataset(Xtr,ytr); va=TensorDataset(Xva,yva)
    return (DataLoader(tr,batch_size=bs,shuffle=True,num_workers=0),
            DataLoader(va,batch_size=bs,shuffle=False,num_workers=0))

def train_one_trial(trial: optuna.Trial):
    hidden   = trial.suggest_categorical("hidden",[32,64,96])
    layers   = trial.suggest_categorical("layers",[1,2])
    dropout  = trial.suggest_float("dropout",0.0,0.3,step=0.1)
    batch_sz = trial.suggest_categorical("batch_size",[64,128])
    lr       = trial.suggest_float("lr",1e-4,1e-3,log=True)
    wd       = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    clip     = trial.suggest_float("clip_norm",0.5,1.5,step=0.5)
    epochs   = trial.suggest_int("epochs",8,MAX_EPOCHS)

    tr_loader, va_loader = make_loaders(batch_sz)
    model = GRURegressor(INPUT_SIZE, hidden, num_layers=layers, dropout=dropout).to(DEVICE)
    criterion = nn.HuberLoss(delta=0.02)  # robust to outliers
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best=float("inf"); best_state=None; bad=0
    for ep in range(1, epochs+1):
        model.train()
        for xb,yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        # val
        model.eval(); val_sum=0.0; n=0
        with torch.no_grad():
            for xb,yb in va_loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                l = criterion(model(xb), yb).item()
                val_sum += l * yb.size(0); n += yb.size(0)
        val = val_sum/max(n,1)

        trial.report(val, ep)
        if trial.should_prune(): raise optuna.TrialPruned()

        if val + 1e-9 < best:
            best = val; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad += 1
            if bad >= PATIENCE: break

    os.makedirs("artifacts", exist_ok=True)
    torch.save(best_state, f"artifacts/gru_trial_{trial.number:03d}.pt")
    return best

def main():
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    study = optuna.create_study(direction="minimize", pruner=pruner, study_name="gru_opt_lite")
    start=time.time()
    study.optimize(train_one_trial, n_trials=N_TRIALS, gc_after_trial=True)
    mins=(time.time()-start)/60

    bt=study.best_trial
    print("\nBest trial:", bt.number, "val MSE:", bt.value, "\nparams:", bt.params)

    with open("artifacts/gru_best_params.json","w") as f: json.dump(bt.params,f,indent=2)
    # copy best checkpoint as canonical
    src=f"artifacts/gru_trial_{bt.number:03d}.pt"; dst="artifacts/gru_best.pt"
    with open(src,"rb") as fin, open(dst,"wb") as fout: fout.write(fin.read())
    print(f"Saved best params + checkpoint. Total tuning time: {mins:.2f} min")

if __name__=="__main__":
    main()
