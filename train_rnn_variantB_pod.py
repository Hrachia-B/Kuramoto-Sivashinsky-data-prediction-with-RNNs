# # ====================== train_rnn_variantB_pod.py ======================
# # POD (SVD) + simple RNN (tanh) for KS forecasting (Variant B)
# # - Compute spatial POD modes on the TRAIN split (uu: T x N)
# # - Project to r-dimensional latent coefficients a(t)
# # - Train RNN on a(t) with residual Δa and multi-step loss (K)
# # - Rollout in latent, reconstruct to physical space, compute metrics & plots
# # - Keeps simple RNN (no GRU/LSTM) as requested
# #
# # Usage (PowerShell one-liner):
# #   python train_rnn_variantB_pod.py --data ks_data_long.npz --r 20 --Tin 40 \
# #       --hidden 256 --epochs 200 --batch 64 --lr 3e-4 --train_k 5 --residual \
# #       --rollout_H 400 --seed_idx 20

# from __future__ import annotations
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# import matplotlib.pyplot as plt
# from scipy.fft import rfft, rfftfreq


# # ---------------- utils ----------------

# def make_windows(U: np.ndarray, Tin: int = 20, Tout: int = 1, stride: int = 1):
#     T, N = U.shape
#     Xs, Ys = [], []
#     for t0 in range(0, T - Tin - Tout + 1, stride):
#         Xs.append(U[t0:t0 + Tin])
#         Ys.append(U[t0 + Tin:t0 + Tin + Tout])
#     X = np.stack(Xs)
#     Y = np.stack(Ys)
#     return X.astype(np.float32), Y.astype(np.float32)


# def explained_energy(singular_vals: np.ndarray):
#     s2 = singular_vals**2
#     cum = np.cumsum(s2)
#     tot = np.sum(s2)
#     return cum / (tot + 1e-12)


# # ---------------- model ----------------

# class SimpleRNN(nn.Module):
#     def __init__(self, D: int, hidden: int = 256, num_layers: int = 1):
#         super().__init__()
#         self.rnn = nn.RNN(input_size=D, hidden_size=hidden, num_layers=num_layers,
#                           nonlinearity='tanh', batch_first=True)
#         self.head = nn.Linear(hidden, D)

#     def forward(self, x):  # x: (B, Tin, D)
#         h, _ = self.rnn(x)
#         y = self.head(h[:, -1])  # (B, D)
#         return y


# # ---------------- metrics ----------------

# def rmse(y_true, y_pred):
#     return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# def nrmse(y_true, y_pred):
#     denom = np.max(y_true) - np.min(y_true) + 1e-12
#     return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / denom)

# def mae(y_true, y_pred):
#     return float(np.mean(np.abs(y_true - y_pred)))

# def pearson_r(y_true, y_pred):
#     yt = y_true.reshape(-1)
#     yp = y_pred.reshape(-1)
#     yt = yt - yt.mean(); yp = yp - yp.mean()
#     num = float((yt @ yp))
#     den = float(np.linalg.norm(yt) * np.linalg.norm(yp) + 1e-12)
#     return num / den if den > 0 else 0.0

# def spectral_mse(y_true, y_pred):
#     T, _ = y_true.shape
#     mse_list = []
#     for t in range(T):
#         Yt = np.abs(rfft(y_true[t]))
#         Yp = np.abs(rfft(y_pred[t]))
#         mse_list.append(np.mean((Yt - Yp) ** 2))
#     return float(np.mean(mse_list))


# # ---------------- rollout ----------------

# def rollout(model: nn.Module, x0: torch.Tensor, H: int, residual: bool):
#     model.eval()
#     preds = []
#     cur = x0.clone()
#     with torch.no_grad():
#         for _ in range(H):
#             step = model(cur)
#             if residual:
#                 step = cur[:, -1, :] + step
#             preds.append(step.squeeze(0).cpu().numpy())
#             cur = torch.cat([cur[:, 1:, :], step.unsqueeze(1)], dim=1)
#     return np.stack(preds)


# # ---------------- main ----------------
# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', type=str, default='ks_data_long.npz')
#     parser.add_argument('--r', type=int, default=20, help='POD rank (latent dim)')
#     parser.add_argument('--Tin', type=int, default=40)
#     parser.add_argument('--hidden', type=int, default=256)
#     parser.add_argument('--epochs', type=int, default=200)
#     parser.add_argument('--batch', type=int, default=64)
#     parser.add_argument('--lr', type=float, default=3e-4)
#     parser.add_argument('--train_k', type=int, default=5, help='K-step free rollout loss during training (>=1)')
#     parser.add_argument('--residual', action='store_true', default=True,
#                         help='Predict Δa and add to last latent frame (default ON)')
#     parser.add_argument('--rollout_H', type=int, default=400)
#     parser.add_argument('--val_frac', type=float, default=0.2)
#     parser.add_argument('--seed_idx', type=int, default=20,
#                         help='Validation window index to seed rollout (0 = earliest)')
#     args = parser.parse_args()

#     # ---- load data ----
#     Z = np.load(args.data)
#     uu = Z['uu']  # (T, N)
#     x = Z['x']
#     t = Z['t']
#     dt_eff = float(Z['dt_eff'])
#     T, N = uu.shape
#     print(f"Loaded {args.data}: uu.shape={uu.shape}, dt_eff={dt_eff}")

#     # ---- train/val split in TIME (before POD) ----
#     Ttrain = int((1.0 - args.val_frac) * T)
#     uu_tr = uu[:Ttrain]
#     uu_va = uu[Ttrain:]

#     # ---- POD on TRAIN only ----
#     mu = uu_tr.mean(axis=0, keepdims=True)              # (1, N)
#     Uc = uu_tr - mu                                     # center over time
#     # SVD: Uc = W diag(S) V^T, where columns of V are spatial POD modes
#     W, S, Vt = np.linalg.svd(Uc, full_matrices=False)
#     V = Vt.T                                           # (N, N)
#     energy = explained_energy(S)

#     r = int(min(args.r, N))
#     Vr = V[:, :r]                                       # (N, r)
#     print(f"POD rank r={r}; explained energy ~ {energy[r-1]*100:.2f}%")

#     # ---- project to latent coeffs a(t) ----
#     def project(U):
#         return (U - mu) @ Vr            # (T, r)
#     def reconstruct(A):
#         return (A @ Vr.T) + mu          # (T, N)

#     a_tr = project(uu_tr)               # (Ttr, r)
#     a_va = project(uu_va)               # (Tva, r)

#     # ---- normalize latent coefficients per-dim on train only ----
#     a_mean = a_tr.mean(axis=0, keepdims=True)
#     a_std  = a_tr.std(axis=0, keepdims=True) + 1e-8

#     def norm_a(A): return (A - a_mean) / a_std
#     def denorm_a(Ah): return Ah * a_std + a_mean

#     a_tr_n = norm_a(a_tr)
#     a_va_n = norm_a(a_va)

#     # ---- build windows in latent space ----
#     Tin, Tout = args.Tin, 1
#     Xtr, Ytr = make_windows(a_tr_n, Tin=Tin, Tout=Tout, stride=1)   # (S_tr, Tin, r), (S_tr, 1, r)
#     Xva, Yva = make_windows(a_va_n, Tin=Tin, Tout=Tout, stride=1)
#     Ytr = Ytr[:, 0]
#     Yva = Yva[:, 0]

#     # ---- dataloaders ----
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     train_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
#                           batch_size=args.batch, shuffle=False)
#     val_dl   = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
#                           batch_size=args.batch, shuffle=False)

#     # ---- model ----
#     model = SimpleRNN(D=r, hidden=args.hidden).to(device)
#     opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
#     loss_fn = nn.MSELoss()

#     # ---- train with K-step free rollout loss in latent ----
#     K = max(1, args.train_k)
#     for epoch in range(args.epochs):
#         model.train()
#         tr_loss = 0.0
#         for xb, yb in train_dl:  # xb: (B, Tin, r); yb: (B, r)
#             xb = xb.to(device); yb = yb.to(device)
#             opt.zero_grad()
#             if K == 1:
#                 pred = model(xb)
#                 if args.residual:
#                     pred = xb[:, -1, :] + pred
#                 loss = loss_fn(pred, yb)
#             else:
#                 cur = xb.clone()
#                 loss = 0.0
#                 for _ in range(K):
#                     step = model(cur)
#                     if args.residual:
#                         step = cur[:, -1, :] + step
#                     loss = loss + loss_fn(step, yb)
#                     cur = torch.cat([cur[:, 1:, :], step.unsqueeze(1)], dim=1)
#                 loss = loss / K
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             opt.step()
#             tr_loss += loss.item() * len(xb)

#         # validation one-step in latent
#         model.eval()
#         va_loss = 0.0
#         with torch.no_grad():
#             for xb, yb in val_dl:
#                 xb = xb.to(device); yb = yb.to(device)
#                 pred = model(xb)
#                 if args.residual:
#                     pred = xb[:, -1, :] + pred
#                 va_loss += loss_fn(pred, yb).item() * len(xb)

#         tr_loss /= len(train_dl.dataset)
#         va_loss /= len(val_dl.dataset)
#         print(f"epoch {epoch:03d} | train {tr_loss:.6f} | val {va_loss:.6f}")

#     # ---- rollout in latent on validation segment ----
#     # choose seed window inside validation with enough future
#     # seed_idx is relative to validation windows
#     seed_idx = int(np.clip(args.seed_idx, 0, max(0, Xva.shape[0]-1)))
#     x0 = torch.from_numpy(Xva[seed_idx:seed_idx+1]).to(device)   # (1, Tin, r)

#     # absolute index in validation portion
#     t0_abs_val = seed_idx + Tin
#     H_avail = int(min(args.rollout_H, a_va_n.shape[0] - t0_abs_val))
#     if H_avail <= 0:
#         raise RuntimeError("Not enough future steps for rollout in validation. Reduce --seed_idx or --rollout_H.")

#     a_pred_n = rollout(model, x0, H=H_avail, residual=args.residual)  # (H, r) in normalized latent
#     a_true_n = a_va_n[t0_abs_val:t0_abs_val + H_avail]                 # (H, r)

#     # reconstruct to physical
#     a_pred = denorm_a(a_pred_n)
#     u_pred = reconstruct(a_pred)
#     u_true = reconstruct(a_true_n)

#     # ---- metrics in physical space ----
#     m_rmse = rmse(u_true, u_pred)
#     m_nrmse = nrmse(u_true, u_pred)
#     m_mae = mae(u_true, u_pred)
#     m_pr = pearson_r(u_true, u_pred)
#     m_spec = spectral_mse(u_true, u_pred)

#     print("\nRollout metrics (physical space):")
#     print(f"H_avail={H_avail} (requested {args.rollout_H}) | seed_idx={seed_idx} | r={r}")
#     print(f"RMSE={m_rmse:.6f} | NRMSE={m_nrmse:.6f} | MAE={m_mae:.6f} | Pearson r={m_pr:.4f} | Spectral MSE={m_spec:.6f}")

#     # ---- plots ----
#     # 0) POD energy
#     plt.figure(figsize=(6,3))
#     plt.plot(np.arange(1, len(energy)+1), energy)
#     plt.axvline(r, color='k', linestyle='--', linewidth=1)
#     plt.xlabel('rank'); plt.ylabel('cumulative energy'); plt.title('POD energy (train)')
#     plt.tight_layout()

#     # 1) Heatmaps
#     # Map validation indices to physical time for x-axis extent
#     # here we only know dt_eff; y-axis is x
#     fig1, axs = plt.subplots(1, 2, figsize=(12, 4))
#     im0 = axs[0].imshow(u_true.T, aspect='auto', origin='lower',
#                         extent=[0, H_avail * dt_eff, x[0], x[-1]])
#     axs[0].set_title('True field (rollout, POD)')
#     axs[0].set_xlabel('time'); axs[0].set_ylabel('x')
#     fig1.colorbar(im0, ax=axs[0])

#     im1 = axs[1].imshow(u_pred.T, aspect='auto', origin='lower',
#                         extent=[0, H_avail * dt_eff, x[0], x[-1]])
#     axs[1].set_title('Predicted field (rollout, POD)')
#     axs[1].set_xlabel('time'); axs[1].set_ylabel('x')
#     fig1.colorbar(im1, ax=axs[1])
#     fig1.suptitle('KS RNN Variant B: POD + RNN (tanh)')
#     plt.tight_layout()

#     # 2) Snapshot comparison at mid-horizon
#     tidx = min(H_avail - 1, H_avail // 2)
#     plt.figure(figsize=(8, 4))
#     plt.plot(x, u_true[tidx], label='true')
#     plt.plot(x, u_pred[tidx], label='pred', linestyle='--')
#     plt.xlabel('x'); plt.ylabel('u(x)'); plt.title(f'Snapshot @ t={tidx * dt_eff:.3f}')
#     plt.legend(); plt.tight_layout()

#     # 3) Average spatial spectra
#     Y_true = np.abs(rfft(u_true, axis=1))
#     Y_pred = np.abs(rfft(u_pred, axis=1))
#     f = rfftfreq(N, d=(x[1] - x[0]))
#     plt.figure(figsize=(8, 4))
#     plt.plot(f, Y_true.mean(axis=0), label='true avg |F|')
#     plt.plot(f, Y_pred.mean(axis=0), label='pred avg |F|', linestyle='--')
#     plt.xlabel('spatial frequency'); plt.ylabel('|F|')
#     plt.title('Average spatial spectra over rollout')
#     plt.legend(); plt.tight_layout()

#     # 4) Error vs horizon
#     errs = np.sqrt(np.mean((u_true - u_pred) ** 2, axis=1))
#     plt.figure(figsize=(8, 4))
#     plt.plot(np.arange(H_avail) * dt_eff, errs)
#     plt.xlabel('time'); plt.ylabel('RMSE'); plt.title('Rollout RMSE vs time (POD)')
#     plt.tight_layout()

#     plt.show()




# ====================== train_rnn_variantB_pod.py ======================
# POD (SVD) + simple RNN (tanh) for KS forecasting (Variant B, stronger cfg)
# Без смены модели: усилили Tin/hidden/K/r и добавили Dropout

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# ---------------- utils ----------------
def make_windows(U: np.ndarray, Tin: int = 20, Tout: int = 1, stride: int = 1):
    T, N = U.shape
    Xs, Ys = [], []
    for t0 in range(0, T - Tin - Tout + 1, stride):
        Xs.append(U[t0:t0 + Tin])
        Ys.append(U[t0 + Tin:t0 + Tin + Tout])
    X = np.stack(Xs)
    Y = np.stack(Ys)
    return X.astype(np.float32), Y.astype(np.float32)

def explained_energy(singular_vals: np.ndarray):
    s2 = singular_vals**2
    cum = np.cumsum(s2); tot = np.sum(s2)
    return cum / (tot + 1e-12)

# ---------------- model ----------------
class SimpleRNN(nn.Module):
    def __init__(self, D: int, hidden: int = 512, num_layers: int = 1, p_drop: float = 0.2):
        super().__init__()
        self.rnn = nn.RNN(input_size=D, hidden_size=hidden, num_layers=num_layers,
                          nonlinearity='tanh', batch_first=True)
        self.drop = nn.Dropout(p_drop)
        self.head = nn.Linear(hidden, D)

    def forward(self, x):  # x: (B, Tin, D)
        h, _ = self.rnn(x)
        z = self.drop(h[:, -1])          # regularize last hidden
        y = self.head(z)                  # (B, D)
        return y

# ---------------- metrics ----------------
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

# ---------------- rollout ----------------
def rollout(model: nn.Module, x0: torch.Tensor, H: int, residual: bool):
    model.eval(); preds = []; cur = x0.clone()
    with torch.no_grad():
        for _ in range(H):
            step = model(cur)
            if residual: step = cur[:, -1, :] + step
            preds.append(step.squeeze(0).cpu().numpy())
            cur = torch.cat([cur[:, 1:, :], step.unsqueeze(1)], dim=1)
    return np.stack(preds)

# ---------------- main ----------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ks_data_long.npz')
    parser.add_argument('--r', type=int, default=30, help='POD rank (latent dim)')
    parser.add_argument('--Tin', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--train_k', type=int, default=10, help='K-step free-rollout loss (>=1)')
    parser.add_argument('--residual', action='store_true', default=True,
                        help='Predict Δa and add to last latent frame (default ON)')
    parser.add_argument('--rollout_H', type=int, default=400)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--seed_idx', type=int, default=20,
                        help='Validation window index for rollout (0 = earliest)')
    args = parser.parse_args()

    # ---- load data ----
    Z = np.load(args.data)
    uu = Z['uu']; x = Z['x']; t = Z['t']; dt_eff = float(Z['dt_eff'])
    T, N = uu.shape
    print(f"Loaded {args.data}: uu.shape={uu.shape}, dt_eff={dt_eff}")

    # ---- split in time ----
    Ttrain = int((1.0 - args.val_frac) * T)
    uu_tr, uu_va = uu[:Ttrain], uu[Ttrain:]

    # ---- POD on TRAIN ----
    mu = uu_tr.mean(axis=0, keepdims=True)
    Uc = uu_tr - mu
    W, S, Vt = np.linalg.svd(Uc, full_matrices=False)
    V = Vt.T
    energy = explained_energy(S)
    r = int(min(args.r, N))
    Vr = V[:, :r]
    print(f"POD rank r={r}; explained energy ~ {energy[r-1]*100:.2f}%")

    # ---- project <-> reconstruct ----
    def project(U):     return (U - mu) @ Vr
    def reconstruct(A): return (A @ Vr.T) + mu

    a_tr, a_va = project(uu_tr), project(uu_va)

    # ---- normalize latent per-dim on train ----
    a_mean = a_tr.mean(axis=0, keepdims=True); a_std = a_tr.std(axis=0, keepdims=True) + 1e-8
    def norm_a(A):    return (A - a_mean)/a_std
    def denorm_a(Ah): return Ah * a_std + a_mean

    a_tr_n, a_va_n = norm_a(a_tr), norm_a(a_va)

    # ---- windows in latent ----
    Tin, Tout = args.Tin, 1
    Xtr, Ytr = make_windows(a_tr_n, Tin=Tin, Tout=Tout, stride=1); Ytr = Ytr[:, 0]
    Xva, Yva = make_windows(a_va_n, Tin=Tin, Tout=Tout, stride=1); Yva = Yva[:, 0]

    # ---- dataloaders ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                          batch_size=args.batch, shuffle=True)
    val_dl   = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
                          batch_size=args.batch, shuffle=False)

    # ---- model & optim ----
    model = SimpleRNN(D=r, hidden=args.hidden, p_drop=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    K = max(1, args.train_k)

    # ---- train ----
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

    # ---- rollout on validation ----
    seed_idx = int(np.clip(args.seed_idx, 0, max(0, Xva.shape[0]-1)))
    x0 = torch.from_numpy(Xva[seed_idx:seed_idx+1]).to(device)
    t0_abs_val = seed_idx + Tin
    H_avail = int(min(args.rollout_H, a_va_n.shape[0] - t0_abs_val))
    if H_avail <= 0:
        raise RuntimeError("Not enough future steps for rollout; reduce --seed_idx/--rollout_H.")

    a_pred_n = rollout(model, x0, H=H_avail, residual=args.residual)
    a_true_n = a_va_n[t0_abs_val:t0_abs_val + H_avail]
    a_pred = denorm_a(a_pred_n)
    u_pred = reconstruct(a_pred)
    u_true = reconstruct(a_true_n)

    # ---- metrics ----
    m_rmse = rmse(u_true, u_pred); m_nrmse = nrmse(u_true, u_pred)
    m_mae  = mae(u_true, u_pred);  m_pr    = pearson_r(u_true, u_pred)
    m_spec = spectral_mse(u_true, u_pred)
    print("\nRollout metrics (physical space):")
    print(f"H_avail={H_avail} (requested {args.rollout_H}) | seed_idx={seed_idx} | r={r}")
    print(f"RMSE={m_rmse:.6f} | NRMSE={m_nrmse:.6f} | MAE={m_mae:.6f} | Pearson r={m_pr:.4f} | Spectral MSE={m_spec:.6f}")

    # ---- plots ----
    energy = explained_energy(S)
    plt.figure(figsize=(6,3))
    plt.plot(np.arange(1, len(energy)+1), energy)
    plt.axvline(r, color='k', linestyle='--', linewidth=1)
    plt.xlabel('rank'); plt.ylabel('cumulative energy'); plt.title('POD energy (train)')
    plt.tight_layout()

    fig1, axs = plt.subplots(1, 2, figsize=(12, 4))
    im0 = axs[0].imshow(u_true.T, aspect='auto', origin='lower',
                        extent=[0, H_avail * dt_eff, x[0], x[-1]])
    axs[0].set_title('True field (rollout, POD)')
    axs[0].set_xlabel('time'); axs[0].set_ylabel('x')
    fig1.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(u_pred.T, aspect='auto', origin='lower',
                        extent=[0, H_avail * dt_eff, x[0], x[-1]])
    axs[1].set_title('Predicted field (rollout, POD)')
    axs[1].set_xlabel('time'); axs[1].set_ylabel('x')
    fig1.colorbar(im1, ax=axs[1])
    fig1.suptitle('KS RNN Variant B: POD + RNN (tanh, strong cfg)')
    plt.tight_layout()

    tidx = min(H_avail - 1, H_avail // 2)
    plt.figure(figsize=(8, 4))
    plt.plot(x, u_true[tidx], label='true')
    plt.plot(x, u_pred[tidx], label='pred', linestyle='--')
    plt.xlabel('x'); plt.ylabel('u(x)'); plt.title(f'Snapshot @ t={tidx * dt_eff:.3f}')
    plt.legend(); plt.tight_layout()

    Y_true = np.abs(rfft(u_true, axis=1)); Y_pred = np.abs(rfft(u_pred, axis=1))
    f = rfftfreq(N, d=(x[1] - x[0]))
    plt.figure(figsize=(8, 4))
    plt.plot(f, Y_true.mean(axis=0), label='true avg |F|')
    plt.plot(f, Y_pred.mean(axis=0), label='pred avg |F|', linestyle='--')
    plt.xlabel('spatial frequency'); plt.ylabel('|F|')
    plt.title('Average spatial spectra over rollout')
    plt.legend(); plt.tight_layout()

    errs = np.sqrt(np.mean((u_true - u_pred) ** 2, axis=1))
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(H_avail) * dt_eff, errs)
    plt.xlabel('time'); plt.ylabel('RMSE'); plt.title('Rollout RMSE vs time (POD)')
    plt.tight_layout()
    plt.show()
