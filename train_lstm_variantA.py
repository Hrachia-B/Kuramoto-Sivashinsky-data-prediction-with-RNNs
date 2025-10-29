# ====================== train_lstm_variantA.py ======================
# LSTM baseline for KS: full field (N) next-step prediction with residual & K-step loss

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# ---------- utils ----------
def make_windows(U: np.ndarray, Tin: int = 20, Tout: int = 1, stride: int = 1):
    T, N = U.shape
    Xs, Ys = [], []
    for t0 in range(0, T - Tin - Tout + 1, stride):
        Xs.append(U[t0:t0 + Tin])
        Ys.append(U[t0 + Tin:t0 + Tin + Tout])
    X = np.stack(Xs); Y = np.stack(Ys)
    return X.astype(np.float32), Y.astype(np.float32)

# ---------- model ----------
class LSTMModel(nn.Module):
    def __init__(self, N: int, hidden: int = 256, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=N, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, N)

    def forward(self, x):  # x: (B, Tin, N)
        h, _ = self.lstm(x)
        y = self.head(self.drop(h[:, -1]))
        return y  # (B, N)

# ---------- metrics ----------
def rmse(y_true, y_pred):   return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def nrmse(y_true, y_pred):  return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.max(y_true)-np.min(y_true)+1e-12))
def mae(y_true, y_pred):    return float(np.mean(np.abs(y_true - y_pred)))
def pearson_r(y_true, y_pred):
    yt = y_true.reshape(-1); yp = y_pred.reshape(-1)
    yt = yt - yt.mean(); yp = yp - yp.mean()
    num = float((yt @ yp)); den = float(np.linalg.norm(yt) * np.linalg.norm(yp) + 1e-12)
    return num / den if den > 0 else 0.0
def spectral_mse(y_true, y_pred):
    T, _ = y_true.shape; mse_list = []
    for t in range(T):
        Yt = np.abs(rfft(y_true[t])); Yp = np.abs(rfft(y_pred[t]))
        mse_list.append(np.mean((Yt - Yp) ** 2))
    return float(np.mean(mse_list))

# ---------- rollout ----------
def rollout(model: nn.Module, x0: torch.Tensor, H: int, residual: bool):
    model.eval(); preds = []; cur = x0.clone()
    with torch.no_grad():
        for _ in range(H):
            step = model(cur)                 # (1, N) either Δu or u_{t+1}
            if residual: step = cur[:, -1, :] + step
            preds.append(step.squeeze(0).cpu().numpy())
            cur = torch.cat([cur[:, 1:, :], step.unsqueeze(1)], dim=1)
    return np.stack(preds)

# ---------- main ----------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ks_data_long.npz')
    parser.add_argument('--Tin', type=int, default=40)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--train_k', type=int, default=5)
    parser.add_argument('--residual', action='store_true', default=True)
    parser.add_argument('--rollout_H', type=int, default=400)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--seed_idx', type=int, default=20)
    args = parser.parse_args()

    Z = np.load(args.data)
    uu, x, t = Z['uu'], Z['x'], Z['t']
    dt_eff = float(Z['dt_eff'])
    T, N = uu.shape
    print(f"Loaded {args.data}: uu.shape={uu.shape}, dt_eff={dt_eff}")

    Tin = args.Tin
    X, Y = make_windows(uu, Tin=Tin, Tout=1, stride=1)
    S = X.shape[0]; split = int((1.0 - args.val_frac) * S)
    Xtr_raw, Ytr_raw = X[:split], Y[:split]; Xva_raw, Yva_raw = X[split:], Y[split:]

    # per-space normalization (fit on train only)
    train_concat = np.concatenate([Xtr_raw.reshape(-1, N), Ytr_raw.reshape(-1, N)], axis=0)
    mean = train_concat.mean(axis=0, keepdims=True)
    std  = train_concat.std(axis=0, keepdims=True) + 1e-8
    norm = lambda a: (a - mean) / std

    Xtr = norm(Xtr_raw); Ytr = norm(Ytr_raw)[:, 0]
    Xva = norm(Xva_raw); Yva = norm(Yva_raw)[:, 0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                          batch_size=args.batch, shuffle=True)
    val_dl   = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
                          batch_size=args.batch, shuffle=False)

    model = LSTMModel(N=N, hidden=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    K = max(1, args.train_k)

    for epoch in range(args.epochs):
        model.train(); tr_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            if K == 1:
                pred = model(xb)
                if args.residual: pred = xb[:, -1, :] + pred
                loss = loss_fn(pred, yb)
            else:
                cur = xb.clone(); loss = 0.0
                for _ in range(K):
                    step = model(cur)
                    if args.residual: step = cur[:, -1, :] + step
                    loss = loss + loss_fn(step, yb)
                    cur = torch.cat([cur[:, 1:, :], step.unsqueeze(1)], dim=1)
                loss = loss / K
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(xb)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                if args.residual: pred = xb[:, -1, :] + pred
                va_loss += loss_fn(pred, yb).item() * len(xb)

        tr_loss /= len(train_dl.dataset); va_loss /= len(val_dl.dataset)
        print(f"epoch {epoch:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")

    # rollout
    seed_idx = int(np.clip(args.seed_idx, 0, max(0, Xva.shape[0]-1)))
    x0 = torch.from_numpy(Xva[seed_idx:seed_idx+1]).to(device)
    t0_abs = split + seed_idx + Tin
    H_avail = int(min(args.rollout_H, uu.shape[0] - t0_abs))
    if H_avail <= 0: raise RuntimeError("Not enough future steps for rollout.")

    pred_norm = rollout(model, x0, H_avail, residual=args.residual)
    true_seg  = uu[t0_abs:t0_abs + H_avail]
    pred_den  = pred_norm * std + mean

    # metrics
    print("\nRollout metrics (physical space):")
    print(f"H_avail={H_avail} (requested {args.rollout_H}) | seed_idx={seed_idx}")
    print(f"RMSE={rmse(true_seg,pred_den):.6f} | NRMSE={nrmse(true_seg,pred_den):.6f} | "
          f"MAE={mae(true_seg,pred_den):.6f} | Pearson r={pearson_r(true_seg,pred_den):.4f} | "
          f"Spectral MSE={spectral_mse(true_seg,pred_den):.6f}")

    # plots
    import matplotlib.pyplot as plt
    fig1, axs = plt.subplots(1, 2, figsize=(12, 4))
    im0 = axs[0].imshow(true_seg.T, aspect='auto', origin='lower',
                        extent=[0, H_avail*dt_eff, x[0], x[-1]])
    axs[0].set_title('True field (rollout)'); axs[0].set_xlabel('time'); axs[0].set_ylabel('x')
    fig1.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(pred_den.T, aspect='auto', origin='lower',
                        extent=[0, H_avail*dt_eff, x[0], x[-1]])
    axs[1].set_title('Predicted field (rollout)'); axs[1].set_xlabel('time'); axs[1].set_ylabel('x')
    fig1.colorbar(im1, ax=axs[1]); plt.tight_layout()

    tidx = min(H_avail-1, H_avail//2)
    plt.figure(figsize=(8,4))
    plt.plot(x, true_seg[tidx], label='true')
    plt.plot(x, pred_den[tidx], label='pred', linestyle='--')
    plt.xlabel('x'); plt.ylabel('u(x)'); plt.title(f'Snapshot @ t={tidx*dt_eff:.3f}'); plt.legend(); plt.tight_layout()

    Y_true = np.abs(rfft(true_seg, axis=1)); Y_pred = np.abs(rfft(pred_den, axis=1))
    f = rfftfreq(N, d=(x[1]-x[0]))
    plt.figure(figsize=(8,4))
    plt.plot(f, Y_true.mean(axis=0), label='true avg |F|')
    plt.plot(f, Y_pred.mean(axis=0), label='pred avg |F|', linestyle='--')
    plt.xlabel('spatial frequency'); plt.ylabel('|F|'); plt.title('Average spatial spectra'); plt.legend(); plt.tight_layout()

    errs = np.sqrt(np.mean((true_seg - pred_den) ** 2, axis=1))
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(H_avail)*dt_eff, errs)
    plt.xlabel('time'); plt.ylabel('RMSE'); plt.title('Rollout RMSE vs time'); plt.tight_layout()
    plt.show()
