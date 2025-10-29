# ========================= ks_dataset.py =========================
# Kuramoto–Sivashinsky (1D) simulator + dataset generator (Variant A)
# - Spatial: Fourier spectral
# - Temporal: ETDRK4 (Kassam & Trefethen, 2005)
# - Fixes: scipy.fft, integer wavenumbers, complex128, reproducibility
# - Output: npz with uu (T x N), x, dt_eff, meta

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from numpy import pi
from scipy.fft import fft, ifft
from typing import Optional, Dict

np.seterr(over='raise', invalid='raise')

@dataclass
class KSConfig:
    L: float = 16.0        # domain parameter: x in [0, 2*pi*L]
    N: int = 128           # spatial grid points
    dt: float = 0.25       # integrator time step
    tend: float = 150.0    # total simulated time
    iout: int = 1          # save each iout-th step (effective Δt = dt*iout)
    seed: Optional[int] = 42

class KS:
    """
    1D Kuramoto–Sivashinsky: u_t + u*u_x + u_xx + u_xxxx = 0 (periodic)
    Spectral (Fourier) in space, ETDRK4 in time.
    """
    def __init__(self, L: float = 16.0, N: int = 128, dt: float = 0.25,
                 nsteps: Optional[int] = None, tend: float = 150.0, iout: int = 1,
                 seed: Optional[int] = None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._rng = rng
            np.random.seed(seed)
        else:
            self._rng = np.random.default_rng()

        L = float(L); dt = float(dt); tend = float(tend)
        if nsteps is None:
            nsteps = int(tend / dt)
        else:
            nsteps = int(nsteps)
            tend = dt * nsteps

        self.L = L
        self.N = int(N)
        self.dx = 2 * pi * L / self.N
        self.dt = dt
        self.nsteps = nsteps
        self.iout = int(iout)
        self.nout = int(nsteps // self.iout)

        # spatial grid
        self.x = 2 * pi * self.L * np.arange(self.N) / self.N

        # initial condition in physical space (small noise)
        u0 = (self._rng.random(self.N) - 0.5) * 0.01
        self.u0 = u0
        self.v0 = fft(u0)
        self.v = self.v0.copy()
        self.t = 0.0
        self.stepnum = 0
        self.ioutnum = 0

        # storage (complex128 for stability)
        self.vv = np.zeros((self.nout + 1, self.N), dtype=np.complex128)
        self.tt = np.zeros(self.nout + 1, dtype=np.float64)
        self.vv[0] = self.v0
        self.tt[0] = 0.0

        # wavenumbers k (integer indices scaled by 1/L)
        Nh = self.N // 2
        self.k = np.concatenate([np.arange(0, Nh), np.array([0]), np.arange(-Nh + 1, 0)]) / self.L

        # Linear operator (normal-form): l = k^2 - k^4
        self.l = self.k ** 2 - self.k ** 4
        self.g = -0.5j * self.k   # factor for nonlinear term

        # ETDRK4 precomputations
        self._setup_etdrk4()

    def _setup_etdrk4(self):
        dt = self.dt
        l = self.l
        self.E = np.exp(dt * l)
        self.E2 = np.exp(dt * l / 2.0)
        M = 16
        r = np.exp(1j * pi * (np.arange(1, M + 1) - 0.5) / M)
        LR = dt * l[:, None] + r[None, :]
        self.Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
        self.f1 = dt * np.real(np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / (LR ** 3), axis=1))
        self.f2 = dt * np.real(np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / (LR ** 3), axis=1))
        self.f3 = dt * np.real(np.mean((-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / (LR ** 3), axis=1))

    def _nonlinear(self, v):
        u = np.real(ifft(v))
        return self.g * fft(u ** 2)

    def step(self):
        v = self.v
        Nv = self._nonlinear(v)
        a = self.E2 * v + self.Q * Nv
        Na = self._nonlinear(a)
        b = self.E2 * v + self.Q * Na
        Nb = self._nonlinear(b)
        c = self.E2 * a + self.Q * (2.0 * Nb - Nv)
        Nc = self._nonlinear(c)
        self.v = self.E * v + Nv * self.f1 + 2.0 * (Na + Nb) * self.f2 + Nc * self.f3
        self.stepnum += 1
        self.t += self.dt

    def simulate(self, nsteps: Optional[int] = None, iout: Optional[int] = None):
        if nsteps is not None:
            self.nsteps = int(nsteps)
        if iout is not None:
            self.iout = int(iout)
        self.nout = int(self.nsteps // self.iout)

        self.vv = np.zeros((self.nout + 1, self.N), dtype=np.complex128)
        self.tt = np.zeros(self.nout + 1, dtype=np.float64)
        self.v = self.v0.copy()
        self.t = 0.0
        self.stepnum = 0
        self.ioutnum = 0
        self.vv[0] = self.v0
        self.tt[0] = 0.0

        for n in range(1, self.nsteps + 1):
            try:
                self.step()
            except FloatingPointError:
                # truncate if exploded
                self.nout = self.ioutnum
                self.vv = self.vv[: self.nout + 1]
                self.tt = self.tt[: self.nout + 1]
                break
            if (self.iout > 0) and (n % self.iout == 0):
                self.ioutnum += 1
                self.vv[self.ioutnum] = self.v
                self.tt[self.ioutnum] = self.t

    def fourier_to_real(self):
        self.uu = np.real(ifft(self.vv, axis=1))
        return self.uu


def generate_ks_timeseries(cfg: KSConfig) -> Dict:
    ks = KS(L=cfg.L, N=cfg.N, dt=cfg.dt, tend=cfg.tend, iout=cfg.iout, seed=cfg.seed)
    ks.simulate()
    uu = ks.fourier_to_real()  # shape (T, N)
    meta = {
        'L': ks.L,
        'N': ks.N,
        'dx': ks.dx,
        'dt': ks.dt,
        'iout': ks.iout,
        'dt_eff': ks.dt * ks.iout,
        't': ks.tt,
        'seed': cfg.seed,
    }
    return {'uu': uu.astype(np.float64), 'x': ks.x.astype(np.float64), 'meta': meta}


if __name__ == '__main__':
    # Example usage: python ks_dataset.py → saves ks_data.npz
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=float, default=16.0)
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--dt', type=float, default=0.25)
    parser.add_argument('--tend', type=float, default=150.0)
    parser.add_argument('--iout', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='ks_data.npz')
    args = parser.parse_args()

    cfg = KSConfig(L=args.L, N=args.N, dt=args.dt, tend=args.tend, iout=args.iout, seed=args.seed)
    data = generate_ks_timeseries(cfg)

    # pack meta as a lightweight dict of arrays/scalars
    meta = data['meta']
    np.savez_compressed(args.out, uu=data['uu'], x=data['x'], t=meta['t'],
                        L=meta['L'], N=meta['N'], dx=meta['dx'], dt=meta['dt'], iout=meta['iout'],
                        dt_eff=meta['dt_eff'], seed=meta['seed'])
    print(f"Saved {args.out}: uu.shape={data['uu'].shape}, N={meta['N']}, dt_eff={meta['dt_eff']}")


