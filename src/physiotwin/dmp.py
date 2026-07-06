"""Discrete Dynamic Movement Primitives (Ijspeert et al., 2013).

One DMP per output dimension; used here on the 3 components of the
heading-normalized rotation vector of the wrist orientation. The
canonical system is shared, so the dimensions stay synchronized.

  tau^2 y'' = alpha (beta (g - y) - tau y') + f(x) (g - y0)
  tau x'    = -alpha_x x
  f(x)      = sum_i w_i psi_i(x) x / sum_i psi_i(x)

Temporal scaling: change tau. Amplitude scaling: move the goal g.
"""
from __future__ import annotations

import numpy as np


class DMP:
    def __init__(self, n_basis: int = 50, alpha: float = 25.0,
                 alpha_x: float = 4.0):
        self.n_basis = n_basis
        self.alpha = alpha
        self.beta = alpha / 4.0
        self.alpha_x = alpha_x
        # RBF centers equally spaced in phase (exponential in time)
        c_t = np.linspace(0, 1, n_basis)
        self.c = np.exp(-alpha_x * c_t)
        self.h = 1.0 / np.gradient(self.c) ** 2
        self.w = None  # (dims, n_basis)

    def _psi(self, x: float) -> np.ndarray:
        return np.exp(-self.h * (x - self.c) ** 2)

    def fit(self, y: np.ndarray, dt: float):
        """Learn weights from one demonstration y (T, dims)."""
        y = np.atleast_2d(y.T).T
        T, dims = y.shape
        self.tau = (T - 1) * dt
        self.y0, self.g = y[0].copy(), y[-1].copy()
        yd = np.gradient(y, dt, axis=0)
        ydd = np.gradient(yd, dt, axis=0)
        x = np.exp(-self.alpha_x * np.linspace(0, 1, T))
        f_target = (self.tau ** 2 * ydd
                    - self.alpha * (self.beta * (self.g - y) - self.tau * yd))
        scale = self.g - self.y0
        scale[np.abs(scale) < 1e-6] = 1e-6
        self.w = np.zeros((dims, self.n_basis))
        psi = np.stack([self._psi(xi) for xi in x])          # (T, n_basis)
        for d in range(dims):
            target = f_target[:, d] / scale[d]
            for i in range(self.n_basis):
                num = np.sum(x * psi[:, i] * target)
                den = np.sum(x ** 2 * psi[:, i])
                self.w[d, i] = num / den if den > 1e-10 else 0.0
        return self

    def rollout(self, dt: float, tau: float | None = None,
                goal: np.ndarray | None = None,
                y0: np.ndarray | None = None) -> np.ndarray:
        """Integrate the DMP. Returns (T, dims)."""
        tau = tau or self.tau
        g = self.g if goal is None else np.asarray(goal, float)
        y = (self.y0 if y0 is None else np.asarray(y0, float)).copy()
        z = np.zeros_like(y)
        x = 1.0
        scale = g - (self.y0 if y0 is None else y)
        scale[np.abs(scale) < 1e-6] = 1e-6
        out = [y.copy()]
        steps = int(round(tau / dt))
        for _ in range(steps):
            psi = self._psi(x)
            f = (self.w @ psi) / (psi.sum() + 1e-10) * x
            zd = (self.alpha * (self.beta * (g - y) - z) + f * scale) / tau
            z += zd * dt
            y += z / tau * dt
            x += (-self.alpha_x * x / tau) * dt
            out.append(y.copy())
        return np.asarray(out)
