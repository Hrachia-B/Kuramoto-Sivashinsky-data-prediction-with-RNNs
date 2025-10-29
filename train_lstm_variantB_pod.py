# ====================== train_lstm_variantB_pod.py ======================
# POD (SVD) + LSTM in latent space for KS forecasting

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

def make_windows(U: np.ndarray, Tin: int = 20, Tout: int = 1, stride: int = 1):
    T, N = U.shape
    Xs, Ys = [], []
    for t0 in range(0, T - Tin - Tout + 1, stride):
        Xs.append(U[t0:t0 + Tin]); Ys.append(U[t0 + Tin:t0 + Tin + Tout])
    X = np.stack(Xs); Y = np.stack(Ys)
    return X.astype(np.float32), Y.astype(np.float32)

def explained_energy(S):
    s2 = S**2; return np.cumsum(s2) / (np.sum(s2) + 1e-12)

class LSTMModel(nn.Module):
    def __init__(self, D: int, hidden: int = 256, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=D, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, D)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(self.drop(h[:, -1]))

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

def rollout(model: nn.Module, x0: torch.Tensor, H: int, residual: bool):
    model.eval(); preds = []; cur = x0.clone()
    with torch.no_grad():
        for _ in range(H):
            step = model(cur)
            if residual: step = cur[:, -1, :] + step
            preds.append(step.squeeze(0).cpu().numpy())
            cur = torch.cat([cur[:, 1:, :], step.unsqueeze(1)], dim=1)
    return np.stack(preds)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ks_data_long.npz')
    parser.add_argument('--r', type=int, default=30)
    parser.add_argument('--Tin', type=int, default=60)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--train_k', type=int, default=8)
    parser.add_argument('--residual', action='store_true', default=True)
    parser.add_argument('--rollout_H', type=int, default=400)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--seed_idx', type=int, default=20)
    args = parser.parse_args()

    Z = np.load(args.data); uu, x, t = Z['uu'], Z['x'], Z['t']
    dt_eff = float(Z['dt_eff']); T, N = uu.shape
    print(f"Loaded {args.data}: uu.shape={uu.shape}, dt_eff={dt_eff}")

    # train/val split by time
    Ttrain = int((1.0 - args.val_frac) * T)
    uu_tr, uu_va = uu[:Ttrain], uu[Ttrain:]

    # POD on train
    mu = uu_tr.mean(axis=0, keepdims=True)
    Uc = uu_tr - mu
    W, S, Vt = np.linalg.svd(Uc, full_matrices=False)
    V = Vt.T
    energy = explained_energy(S)
    r = int(min(args.r, N)); Vr = V[:, :r]
    print(f"POD rank r={r}; explained energy ~ {energy[r-1]*100:.2f}%")

    project    = lambda U: (U - mu) @ Vr
    reconstruct = lambda A: (A @ Vr.T) + mu

    a_tr, a_va = project(uu_tr), project(uu_va)

    # z-score in latent (fit on train)
    a_mean = a_tr.mean(axis=0, keepdims=True)
    a_std  = a_tr.std(axis=0, keepdims=True) + 1e-8
    norm_a    = lambda A: (A - a_mean) / a_std
    denorm_a  = lambda Ah: Ah * a_std + a_mean

    a_tr_n, a_va_n = norm_a(a_tr), norm_a(a_va)

    # windows
    Tin = args.Tin
    Xtr, Ytr = make_windows(a_tr_n, Tin=Tin, Tout=1, stride=1); Ytr = Ytr[:, 0]
    Xva, Yva = make_windows(a_va_n, Tin=Tin, Tout=1, stride=1); Yva = Yva[:, 0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                          batch_size=args.batch, shuffle=True)
    val_dl   = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
                          batch_size=args.batch, shuffle=False)

    model = LSTMModel(D=r, hidden=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    K = max(1, args.train_k)

    # train
    for epoch in range(args.epochs):
        model.train(); tr_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            if K == 1:
                pred = model(xb); 
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

    # rollout in latent
    seed_idx = int(np.clip(args.seed_idx, 0, max(0, Xva.shape[0]-1)))
    x0 = torch.from_numpy(Xva[seed_idx:seed_idx+1]).to(device)
    t0_abs_val = seed_idx + Tin
    H_avail = int(min(args.rollout_H, a_va_n.shape[0] - t0_abs_val))
    if H_avail <= 0: raise RuntimeError("Not enough future steps for rollout; reduce --seed_idx/--rollout_H.")

    a_pred_n = rollout(model, x0, H_avail, residual=args.residual)
    a_true_n = a_va_n[t0_abs_val:t0_abs_val + H_avail]

    # back to physical
    a_pred = denorm_a(a_pred_n)
    u_pred = reconstruct(a_pred)
    u_true = reconstruct(a_true_n)

    # metrics
    print("\nRollout metrics (physical space):")
    print(f"H_avail={H_avail} (requested {args.rollout_H}) | seed_idx={seed_idx} | r={r}")
    print(f"RMSE={rmse(u_true,u_pred):.6f} | NRMSE={nrmse(u_true,u_pred):.6f} | "
          f"MAE={mae(u_true,u_pred):.6f} | Pearson r={pearson_r(u_true,u_pred):.4f} | "
          f"Spectral MSE={spectral_mse(u_true,u_pred):.6f}")

    # plots
    fig0 = plt.figure(figsize=(6,3))
    plt.plot(np.arange(1,len(energy)+1), energy); plt.axvline(r, color='k', ls='--', lw=1)
    plt.xlabel('rank'); plt.ylabel('cumulative energy'); plt.title('POD energy (train)'); plt.tight_layout()

    fig1, axs = plt.subplots(1, 2, figsize=(12, 4))
    im0 = axs[0].imshow(u_true.T, aspect='auto', origin='lower',
                        extent=[0, H_avail*dt_eff, x[0], x[-1]])
    axs[0].set_title('True field (rollout, POD)'); axs[0].set_xlabel('time'); axs[0].set_ylabel('x')
    fig1.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(u_pred.T, aspect='auto', origin='lower',
                        extent=[0, H_avail*dt_eff, x[0], x[-1]])
    axs[1].set_title('Predicted field (rollout, POD)'); axs[1].set_xlabel('time'); axs[1].set_ylabel('x')
    fig1.colorbar(im1, ax=axs[1]); plt.tight_layout()

    tidx = min(H_avail-1, H_avail//2)
    plt.figure(figsize=(8,4))
    plt.plot(x, u_true[tidx], label='true')
    plt.plot(x, u_pred[tidx], label='pred', linestyle='--')
    plt.xlabel('x'); plt.ylabel('u(x)'); plt.title(f'Snapshot @ t={tidx*dt_eff:.3f}'); plt.legend(); plt.tight_layout()

    Y_true = np.abs(rfft(u_true, axis=1)); Y_pred = np.abs(rfft(u_pred, axis=1))
    f = rfftfreq(N, d=(x[1]-x[0]))
    plt.figure(figsize=(8,4))
    plt.plot(f, Y_true.mean(axis=0), label='true avg |F|')
    plt.plot(f, Y_pred.mean(axis=0), label='pred avg |F|', linestyle='--')
    plt.xlabel('spatial frequency'); plt.ylabel('|F|'); plt.title('Average spatial spectra'); plt.legend(); plt.tight_layout()

    errs = np.sqrt(np.mean((u_true - u_pred)**2, axis=1))
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(H_avail)*dt_eff, errs)
    plt.xlabel('time'); plt.ylabel('RMSE'); plt.title('Rollout RMSE vs time (POD)'); plt.tight_layout()
    plt.show()
