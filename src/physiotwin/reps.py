"""Repetition segmentation from the rotation-angle profile."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.transform import Rotation

from .motion import resample, theta_profile


def session_rotvec(session, fs: float = 100.0):
    """Heading-normalized rotation-vector trajectory (T, 3), theta (T,),
    and the absolute reference attitude (rotation vector) at session
    start — needed to reconstruct absolute watch attitude later."""
    t = session.t_ms
    grid = np.arange(t[0], t[-1], 1000.0 / fs)
    rot, _ = resample(t, session.quat_wxyz, session.gyro, grid)
    ref = rot[:50].mean()
    rel = ref.inv() * rot
    return rel.as_rotvec(), rel.magnitude(), ref.as_rotvec()


def segment_reps(theta: np.ndarray, fs: float = 100.0):
    """Return list of (start, end) sample indices, one per repetition.

    Peaks of the smoothed rotation angle mark mid-repetition; rep
    boundaries are the minima between consecutive peaks.
    """
    win = int(0.5 * fs) | 1
    sm = savgol_filter(theta, win, 2)
    rng = sm.max() - sm.min()
    peaks, _ = find_peaks(sm, prominence=0.4 * rng,
                          distance=int(1.0 * fs))
    if len(peaks) < 2:
        return []
    bounds = [int(np.argmin(sm[:peaks[0]]))]
    for a, b in zip(peaks[:-1], peaks[1:]):
        bounds.append(a + int(np.argmin(sm[a:b])))
    bounds.append(peaks[-1] + int(np.argmin(sm[peaks[-1]:])))
    reps = [(s, e) for s, e in zip(bounds[:-1], bounds[1:])
            if 1.0 * fs <= e - s <= 20.0 * fs]
    return reps
