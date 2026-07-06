"""Frame-invariant motion profiles and left/right comparison metrics.

The two watches sit in different body frames (opposite wrists, opposite
crown orientations) and each Device Motion stream has an arbitrary yaw
reference. Comparisons therefore use quantities that do not depend on
the sensor frame:

  * theta(t): rotation angle between current orientation and the pose at
    the start of the overlap window (heading normalization built in).
  * speed(t): angular speed |omega(t)| from the gyroscope.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _to_rotation(quat_wxyz: np.ndarray) -> Rotation:
    q = quat_wxyz[:, [1, 2, 3, 0]]  # scipy wants x,y,z,w
    return Rotation.from_quat(q)


def resample(t_ms: np.ndarray, quat_wxyz: np.ndarray, gyro: np.ndarray,
             grid_ms: np.ndarray):
    """Slerp orientations and linearly interpolate gyro onto grid_ms."""
    # Slerp requires strictly increasing key times; sort and deduplicate.
    _, keep = np.unique(t_ms, return_index=True)
    rot = Slerp(t_ms[keep], _to_rotation(quat_wxyz[keep]))(grid_ms)
    g = np.column_stack([
        np.interp(grid_ms, t_ms[keep], gyro[keep, i]) for i in range(3)
    ])
    return rot, g


def theta_profile(rot: Rotation, ref_window: int = 50) -> np.ndarray:
    """Rotation angle (rad) relative to the mean pose of the first
    `ref_window` samples. Frame-invariant."""
    ref = rot[:ref_window].mean()
    return (ref.inv() * rot).magnitude()


def angular_speed(gyro: np.ndarray) -> np.ndarray:
    return np.linalg.norm(gyro, axis=1)


def best_lag_corr(a: np.ndarray, b: np.ndarray, fs: float = 100.0,
                  max_lag_s: float = 1.0):
    """Pearson r between a and b maximized over a small time lag
    (compensates residual clock offset between the two watches).
    Returns (r, lag_seconds)."""
    max_lag = int(max_lag_s * fs)
    best = (-2.0, 0)
    for lag in range(-max_lag, max_lag + 1, 5):
        if lag >= 0:
            x, y = a[lag:], b[:len(b) - lag]
        else:
            x, y = a[:len(a) + lag], b[-lag:]
        n = min(len(x), len(y))
        if n < 100:
            continue
        r = np.corrcoef(x[:n], y[:n])[0, 1]
        if r > best[0]:
            best = (r, lag)
    return best[0], best[1] / fs


def compare_pair(lw, rw):
    """All Experiment-1 metrics for one synchronized LW/RW session pair."""
    t0 = max(lw.t_ms[0], rw.t_ms[0])
    t1 = min(lw.t_ms[-1], rw.t_ms[-1])
    grid = np.arange(t0, t1, 10.0)  # 100 Hz

    rot_l, gyro_l = resample(lw.t_ms, lw.quat_wxyz, lw.gyro, grid)
    rot_r, gyro_r = resample(rw.t_ms, rw.quat_wxyz, rw.gyro, grid)

    th_l, th_r = theta_profile(rot_l), theta_profile(rot_r)
    sp_l, sp_r = angular_speed(gyro_l), angular_speed(gyro_r)

    r_theta, lag = best_lag_corr(th_l, th_r)
    r_speed, _ = best_lag_corr(sp_l, sp_r)

    # RMSE of the theta profiles after applying the found lag, in degrees
    k = int(round(lag * 100))
    if k >= 0:
        x, y = th_l[k:], th_r[:len(th_r) - k]
    else:
        x, y = th_l[:len(th_l) + k], th_r[-k:]
    n = min(len(x), len(y))
    rmse_deg = float(np.degrees(np.sqrt(np.mean((x[:n] - y[:n]) ** 2))))

    return {
        "subject": lw.subject, "exercise": lw.exercise,
        "condition": lw.condition, "overlap_s": (t1 - t0) / 1000.0,
        "r_theta": float(r_theta), "r_speed": float(r_speed),
        "rmse_deg": rmse_deg, "lag_s": float(lag),
        "profiles": (grid, th_l, th_r),
    }
