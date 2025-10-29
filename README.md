# 🌪️ Forecasting Chaotic Dynamics of the Kuramoto–Sivashinsky System using RNN and LSTM

This project studies how **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory networks (LSTMs)** can learn and predict the temporal evolution of a chaotic dynamical system — the **Kuramoto–Sivashinsky (KS) equation**.

It reproduces and extends methods used in physics-informed machine learning for modeling nonlinear spatiotemporal systems.

---

## 📘 Overview

The **Kuramoto–Sivashinsky equation** is a canonical model for **chaotic spatiotemporal behavior** in nonlinear systems such as flame fronts, thin liquid films, or reaction–diffusion media.

The goal is to train neural networks to **forecast the system’s evolution** in time, learning the underlying dynamics directly from simulated data.

Two model variants are implemented:
- **Variant A:** Plain RNN / LSTM trained directly on full spatial fields.  
- **Variant B:** POD (Proper Orthogonal Decomposition) + RNN / LSTM trained in reduced latent space.

---

## 🧩 Project Structure

.
├── ks_dataset.py # Kuramoto–Sivashinsky data generator
├── ks_data.npz # Default dataset (short version)
├── ks_data_long.npz # Long simulation for training

├── train_rnn_variantA.py # RNN model: full field
├── train_rnn_variantB_pod.py # RNN model: POD-reduced

├── train_lstm_variantA.py # LSTM model: full field
├── train_lstm_variantB_pod.py # LSTM model: POD-reduced

├── RNN_results_variantA/ # Output figures for RNN (Variant A)
├── RNN_results_variantB_pod/ # Output figures for RNN (Variant B)
├── LSTM_results_variantA/ # Output figures for LSTM (Variant A)
├── LSTM_results_variantB_pod/ # Output figures for LSTM (Variant B)

├── venv/ # Virtual environment (ignored by git)
├── .gitignore
└── README.md

---

## ⚙️ How it Works

### 1️⃣ Dataset Generation (`ks_dataset.py`)

The script numerically integrates the **Kuramoto–Sivashinsky PDE**:

\[
u_t + u u_x + u_{xx} + u_{xxxx} = 0,
\]

using spectral methods and the **ETDRK4** time-stepping scheme.

The output file (`ks_data.npz` or `ks_data_long.npz`) contains:
- `uu` — spatiotemporal field \( u(x, t) \)
- `x` — spatial grid
- `t` — time steps
- `dt_eff` — effective time interval

Each row in `uu` corresponds to one time step; each column corresponds to a spatial point.

---

### 2️⃣ Training the Networks

#### Variant A — Full Field
Trains directly on the full physical space \( u(x, t) \):
- Input: sequence of `Tin` past frames (e.g., 20–60 time steps)
- Output: next frame
- Rollout: multi-step prediction without teacher forcing

#### Variant B — POD-Reduced Space
Performs dimensionality reduction via **POD (SVD)**:
\[
u(x,t) \approx \bar{u}(x) + \sum_{i=1}^r a_i(t) \phi_i(x),
\]
where \( a_i(t) \) are temporal coefficients.  
The RNN/LSTM models the evolution of these coefficients \( a_i \), and then the full field is reconstructed.

---

## 🧮 Evaluation Metrics

After training, models are evaluated by **rollout** — multi-step autoregressive prediction.

| Metric | Description |
|--------|--------------|
| **RMSE** | Root Mean Square Error in physical space |
| **NRMSE** | Normalized RMSE |
| **MAE** | Mean Absolute Error |
| **Pearson r** | Correlation coefficient between true and predicted fields |
| **Spectral MSE** | Error in frequency spectrum (Fourier domain) |

---

## 📊 Visualization Outputs

Each training script automatically generates:
- `Figure_1.png` — POD energy curve (Variant B only)  
- `Figure_2.png` — Heatmaps: true vs predicted spatiotemporal fields  
- `Figure_3.png` — Snapshot comparison at a fixed time  
- `Figure_4.png` — Average spatial spectrum or RMSE vs time

---

## 🚀 How to Run

### 1. Create environment
```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
pip install torch numpy matplotlib scipy


### 2. Generate dataset
```bash
python ks_dataset.py --L 16 --N 128 --dt 0.25 --tend 600 --iout 1 --out ks_data_long.npz


🧠 Model Training

Training is performed via PowerShell commands.
Each script receives configuration parameters through command-line arguments (using backticks ``` for line continuation).
Below are example configurations for all model variants.

🔹 Variant A — Plain RNN (Full Field)
```powershell
python train_rnn_variantA.py `
  --data ks_data_long.npz `
  --Tin 40 `
  --hidden 256 `
  --epochs 200 `
  --batch 64 `
  --lr 1e-3 `
  --rollout_H 400 `
  --seed_idx 20

🔹 Variant B — RNN with POD Compression
```powershell
python train_rnn_variantB_pod.py `
  --data ks_data_long.npz `
  --Tin 40 `
  --hidden 256 `
  --epochs 200 `
  --batch 64 `
  --lr 1e-3 `
  --rollout_H 400 `
  --seed_idx 20 `
  --r 30

🔹 Variant A — LSTM (Full Field)
```powershell
python train_lstm_variantA.py `
  --data ks_data_long.npz `
  --Tin 40 `
  --hidden 256 `
  --layers 1 `
  --dropout 0.1 `
  --epochs 200 `
  --batch 64 `
  --lr 3e-4 `
  --train_k 5 `
  --residual `
  --rollout_H 400 `
  --seed_idx 20

🔹 Variant B — LSTM with POD Compression
```powershell
python train_lstm_variantB_pod.py `
  --data ks_data_long.npz `
  --Tin 40 `
  --hidden 256 `
  --layers 1 `
  --dropout 0.1 `
  --epochs 200 `
  --batch 64 `
  --lr 3e-4 `
  --train_k 5 `
  --residual `
  --rollout_H 400 `
  --seed_idx 20 `
  --r 30

📊 Training Output

Each training script automatically:

    Prints metrics after every epoch (train and validation loss).

    Performs autoregressive rollout for evaluation.

    Saves visualization figures:

        Heatmaps: true vs predicted fields

        POD energy curve (for Variant B)

        Snapshot comparison

        Average spatial spectra

        RMSE vs time

Example output directories:
    RNN_results_variantA/
    RNN_results_variantB_pod/
    LSTM_results_variantA/
    LSTM_results_variantB_pod/

📌 Developed by Hrachya Baghdasaryan