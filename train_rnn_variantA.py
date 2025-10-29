# ====================== train_rnn_variantA.py ======================
# Variant A: predict next full field (N) for 1D KS with simple RNN (tanh)
# Improvements (still plain RNN):
# - residual Δu prediction (default ON)
# - multi-step (K) loss during training (default K=5)
# - stronger defaults: Tin=40, hidden=256, epochs=200, lr=3e-4
# - weight decay + grad clipping
# - rollout aware of residual mode; safe H_avail

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# ---------- utils: dataset windows ----------
def make_windows(U: np.ndarray, Tin: int = 20, Tout: int = 1, stride: int = 1):
    T, N = U.shape
    Xs, Ys = [], []
    for t0 in range(0, T - Tin - Tout + 1, stride):
        Xs.append(U[t0:t0 + Tin])
        Ys.append(U[t0 + Tin:t0 + Tin + Tout])
    X = np.stack(Xs)  # (S, Tin, N)
    Y = np.stack(Ys)  # (S, Tout, N)
    return X.astype(np.float32), Y.astype(np.float32)

# ---------- model ----------
class SimpleRNN(nn.Module):
    def __init__(self, N: int, hidden: int = 256, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.RNN(input_size=N, hidden_size=hidden, num_layers=num_layers,
                          nonlinearity='tanh', batch_first=True)
        self.head = nn.Linear(hidden, N)

    def forward(self, x):  # x: (B, Tin, N)
        h, _ = self.rnn(x)
        y = self.head(h[:, -1])  # (B, N)
        return y

# ---------- metrics ----------
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def nrmse(y_true, y_pred):
    denom = np.max(y_true) - np.min(y_true) + 1e-12
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / denom)

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def pearson_r(y_true, y_pred):
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    yt = yt - yt.mean(); yp = yp - yp.mean()
    num = float((yt @ yp))
    den = float(np.linalg.norm(yt) * np.linalg.norm(yp) + 1e-12)
    return num / den if den > 0 else 0.0

def spectral_mse(y_true, y_pred):
    # rFFT along space for each time; average power spectra difference
    T, _ = y_true.shape
    mse_list = []
    for t in range(T):
        Yt = np.abs(rfft(y_true[t]))
        Yp = np.abs(rfft(y_pred[t]))
        mse_list.append(np.mean((Yt - Yp) ** 2))
    return float(np.mean(mse_list))

# ---------- rollout ----------
def rollout(model: nn.Module, x0: torch.Tensor, H: int, residual: bool):
    """
    x0: (1, Tin, N) normalized window
    returns: (H, N) numpy array of predictions (normalized space)
    """
    model.eval()
    preds = []
    cur = x0.clone()
    with torch.no_grad():
        for _ in range(H):
            step = model(cur)                  # predicts u_{t+1} or Δu
            if residual:
                step = cur[:, -1, :] + step    # u_{t+1} = u_t + Δu
            preds.append(step.squeeze(0).cpu().numpy())
            cur = torch.cat([cur[:, 1:, :], step.unsqueeze(1)], dim=1)
    return np.stack(preds)

# ---------- main ----------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ks_data.npz')
    parser.add_argument('--Tin', type=int, default=40)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--train_k', type=int, default=5, help='K-step free rollout loss during training (>=1)')
    parser.add_argument('--residual', action='store_true', default=True,
                        help='Predict Δu and add to last input frame (default ON)')
    parser.add_argument('--rollout_H', type=int, default=200)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--seed_idx', type=int, default=None,
                        help='Validation window index to seed rollout (0 = earliest). Default: pick with future available.')
    args = parser.parse_args()

    # load data
    Z = np.load(args.data)
    uu = Z['uu']  # (T, N)
    x = Z['x']
    t = Z['t']
    dt_eff = float(Z['dt_eff'])
    T, N = uu.shape
    print(f"Loaded {args.data}: uu.shape={uu.shape}, dt_eff={dt_eff}")

    # make windows on raw, then split, then fit mean/std on train only
    Tin, Tout = args.Tin, 1
    X, Y = make_windows(uu, Tin=Tin, Tout=Tout, stride=1)
    S = X.shape[0]
    split = int((1.0 - args.val_frac) * S)
    Xtr_raw, Ytr_raw = X[:split], Y[:split]
    Xva_raw, Yva_raw = X[split:], Y[split:]

    # normalization (per space index), fit on train only
    train_concat = np.concatenate([Xtr_raw.reshape(-1, N), Ytr_raw.reshape(-1, N)], axis=0)
    mean = train_concat.mean(axis=0, keepdims=True)
    std = train_concat.std(axis=0, keepdims=True) + 1e-8

    def norm(arr):
        return (arr - mean) / std

    Xtr = norm(Xtr_raw)
    Ytr = norm(Ytr_raw)[:, 0]  # (S_tr, N)
    Xva = norm(Xva_raw)
    Yva = norm(Yva_raw)[:, 0]

    # torch loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                          batch_size=args.batch, shuffle=False)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
                        batch_size=args.batch, shuffle=False)

    # model & optim
    model = SimpleRNN(N=N, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # ---------- training with multi-step loss ----------
    best_val = float('inf')
    K = max(1, args.train_k)

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device)               # (B, Tin, N), normalized
            yb = yb.to(device)               # (B, N), target u_{t+1} normalized
            opt.zero_grad()

            if K == 1:
                pred_next = model(xb)        # u_{t+1} or Δu
                if args.residual:
                    pred_next = xb[:, -1, :] + pred_next
                loss = loss_fn(pred_next, yb)
            else:
                cur = xb.clone()
                loss = 0.0
                # Простая схема: оптимизируем 1-шаговый таргет многократно,
                # подавая собственные предсказания назад (свободный роллаут)
                for _ in range(K):
                    step = model(cur)
                    if args.residual:
                        step = cur[:, -1, :] + step
                    loss = loss + loss_fn(step, yb)
                    cur = torch.cat([cur[:, 1:, :], step.unsqueeze(1)], dim=1)
                loss = loss / K

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(xb)

        # validation (one-step)
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                if args.residual:
                    pred = xb[:, -1, :] + pred
                va_loss += loss_fn(pred, yb).item() * len(xb)

        tr_loss /= len(train_dl.dataset)
        va_loss /= len(val_dl.dataset)
        print(f"epoch {epoch:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")
        best_val = min(best_val, va_loss)

    # ---------- rollout eval ----------
    # pick validation window so that future exists
    if args.seed_idx is None:
        seed_idx = max(0, Xva.shape[0] - 1 - 50)  # keep future margin
    else:
        seed_idx = int(np.clip(args.seed_idx, 0, Xva.shape[0] - 1))

    x0 = torch.from_numpy(Xva[seed_idx:seed_idx+1]).to(device)  # (1, Tin, N)

    # absolute time index right after x0
    t0_abs = split + seed_idx + Tin

    # available horizon
    H_avail = int(min(args.rollout_H, uu.shape[0] - t0_abs))
    if H_avail <= 0:
        raise RuntimeError(
            f"Not enough future steps for rollout: t0_abs={t0_abs}, T={uu.shape[0]}, Tin={Tin}. "
            f"Try smaller --rollout_H or smaller --seed_idx."
        )

    pred_norm = rollout(model, x0, H=H_avail, residual=args.residual)   # (H_avail, N)
    true_seg  = uu[t0_abs:t0_abs + H_avail]                              # (H_avail, N)

    # denorm predictions for metrics/plots
    pred_denorm = pred_norm * std + mean

    # metrics (physical space)
    m_rmse = rmse(true_seg, pred_denorm)
    m_nrmse = nrmse(true_seg, pred_denorm)
    m_mae = mae(true_seg, pred_denorm)
    m_pr = pearson_r(true_seg, pred_denorm)
    m_spec = spectral_mse(true_seg, pred_denorm)

    print("\nRollout metrics (physical space):")
    print(f"H_avail={H_avail} (requested {args.rollout_H}) | seed_idx={seed_idx}")
    print(f"RMSE={m_rmse:.6f} | NRMSE={m_nrmse:.6f} | MAE={m_mae:.6f} | Pearson r={m_pr:.4f} | Spectral MSE={m_spec:.6f}")

    # ---------- visualizations ----------
    fig1, axs = plt.subplots(1, 2, figsize=(12, 4))
    im0 = axs[0].imshow(true_seg.T, aspect='auto', origin='lower',
                        extent=[0, H_avail * dt_eff, x[0], x[-1]])
    axs[0].set_title('True field (rollout)')
    axs[0].set_xlabel('time'); axs[0].set_ylabel('x')
    fig1.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred_denorm.T, aspect='auto', origin='lower',
                        extent=[0, H_avail * dt_eff, x[0], x[-1]])
    axs[1].set_title('Predicted field (rollout)')
    axs[1].set_xlabel('time'); axs[1].set_ylabel('x')
    fig1.colorbar(im1, ax=axs[1])
    fig1.suptitle('KS RNN Variant A (plain RNN): heatmaps (physical space)')
    plt.tight_layout()

    tidx = min(H_avail - 1, H_avail // 2)
    plt.figure(figsize=(8, 4))
    plt.plot(x, true_seg[tidx], label='true')
    plt.plot(x, pred_denorm[tidx], label='pred', linestyle='--')
    plt.xlabel('x'); plt.ylabel('u(x)'); plt.title(f'Snapshot @ t={tidx * dt_eff:.3f}')
    plt.legend(); plt.tight_layout()

    Y_true = np.abs(rfft(true_seg, axis=1))
    Y_pred = np.abs(rfft(pred_denorm, axis=1))
    f = rfftfreq(N, d=(x[1] - x[0]))
    plt.figure(figsize=(8, 4))
    plt.plot(f, Y_true.mean(axis=0), label='true avg |F|')
    plt.plot(f, Y_pred.mean(axis=0), label='pred avg |F|', linestyle='--')
    plt.xlabel('spatial frequency'); plt.ylabel('|F|')
    plt.title('Average spatial spectra over rollout')
    plt.legend(); plt.tight_layout()

    errs = np.sqrt(np.mean((true_seg - pred_denorm) ** 2, axis=1))
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(H_avail) * dt_eff, errs)
    plt.xlabel('time'); plt.ylabel('RMSE'); plt.title('Rollout RMSE vs time')
    plt.tight_layout()
    plt.show()
