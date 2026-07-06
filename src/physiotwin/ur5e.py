"""UR5e kinematics: DH forward kinematics, numeric Jacobian, and
damped-least-squares inverse kinematics for pose tracking.

DH parameters from Universal Robots' official specification.
"""
from __future__ import annotations

import numpy as np

# modified? no -- standard DH: theta, d, a, alpha
D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]

Q_LIM = np.array([[-2 * np.pi, 2 * np.pi]] * 6)          # rad
QD_LIM = np.radians([180, 180, 180, 360, 360, 360])      # rad/s (UR5e spec)

HOME = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


def _dh(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fk(q: np.ndarray) -> np.ndarray:
    """Tool pose as 4x4 homogeneous transform."""
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh(q[i], D[i], A[i], ALPHA[i])
    return T


def _pose_err(T: np.ndarray, p_ref: np.ndarray, R_ref: np.ndarray):
    """6-vector [position error; orientation error (rotation vector)]."""
    ep = p_ref - T[:3, 3]
    R_err = R_ref @ T[:3, :3].T
    # rotation vector of R_err
    cos = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
    ang = np.arccos(cos)
    if ang < 1e-9:
        eo = np.zeros(3)
    else:
        eo = ang / (2 * np.sin(ang)) * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]])
    return np.concatenate([ep, eo])


def _jacobian(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numeric geometric Jacobian (6x6) via central differences on fk."""
    T0 = fk(q)
    J = np.zeros((6, 6))
    for i in range(6):
        dq = np.zeros(6); dq[i] = eps
        Tp, Tm = fk(q + dq), fk(q - dq)
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
        dR = (Tp[:3, :3] - Tm[:3, :3]) / (2 * eps) @ T0[:3, :3].T
        J[3:, i] = [dR[2, 1], dR[0, 2], dR[1, 0]]
    return J


def ik_step(q: np.ndarray, p_ref: np.ndarray, R_ref: np.ndarray,
            iters: int = 15, tol: float = 1e-5,
            max_step: float = 0.05) -> tuple[np.ndarray, float, float]:
    """Track one waypoint from seed q. Returns (q, pos_err_m, ori_err_rad).

    Damping adapts to the manipulability (smallest singular value of J)
    so steps stay stable near singularities; each iteration's step norm
    is capped to keep the solution on the seed's configuration branch.
    """
    for _ in range(iters):
        e = _pose_err(fk(q), p_ref, R_ref)
        if np.linalg.norm(e) < tol:
            break
        J = _jacobian(q)
        sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
        damping = max(1e-2, 0.1 * (0.1 - sigma_min) / 0.1) \
            if sigma_min < 0.1 else 1e-2
        JJT = J @ J.T + damping ** 2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, e)
        n = np.linalg.norm(dq)
        if n > max_step:
            dq *= max_step / n
        q = q + dq
    e = _pose_err(fk(q), p_ref, R_ref)
    return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
