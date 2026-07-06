"""Windowing and feature extraction for exercise recognition.

Channels used (from Device Motion): user acceleration x/y/z, rotation
rate x/y/z, gravity x/y/z, pitch, roll, plus the frame-invariant
magnitudes |acc| and |gyro|. Absolute yaw is excluded: its reference
heading is arbitrary per recording and would leak session identity.

Per channel: 9 time-domain + 3 frequency-domain features.
"""
from __future__ import annotations

import numpy as np

from .data import COLS, Session

WIN = 256          # 2.56 s @ 100 Hz
STEP = 128         # 50 % overlap

_CHANNELS = ["ax", "ay", "az", "wx", "wy", "wz",
             "gx", "gy", "gz", "pitch", "roll"]

FEATURE_NAMES: list[str] = []
for ch in _CHANNELS + ["acc_mag", "gyro_mag"]:
    for stat in ["mean", "std", "min", "max", "rms", "iqr", "mad",
                 "zcr", "corr_lag1", "domfreq", "specent", "specpow"]:
        FEATURE_NAMES.append(f"{ch}_{stat}")


def _channel_feats(x: np.ndarray, fs: float = 100.0) -> list[float]:
    mean = float(np.mean(x))
    std = float(np.std(x))
    q75, q25 = np.percentile(x, [75, 25])
    xc = x - mean
    zcr = float(np.mean(np.abs(np.diff(np.signbit(xc).astype(int)))))
    c1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if std > 1e-9 else 0.0
    spec = np.abs(np.fft.rfft(xc)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    tot = spec[1:].sum()
    if tot > 1e-12:
        domfreq = float(freqs[1:][np.argmax(spec[1:])])
        p = spec[1:] / tot
        specent = float(-np.sum(p * np.log(p + 1e-12)))
    else:
        domfreq, specent = 0.0, 0.0
    return [mean, std, float(x.min()), float(x.max()),
            float(np.sqrt(np.mean(x ** 2))), float(q75 - q25),
            float(np.mean(np.abs(xc))), zcr, c1,
            domfreq, specent, float(tot / len(x))]


def window_features(session: Session):
    """Yield (feature_vector, subject, exercise, wrist) per window."""
    a = session.load()
    chans = [a[:, COLS[c]] for c in _CHANNELS]
    chans.append(np.linalg.norm(a[:, [COLS["ax"], COLS["ay"], COLS["az"]]], axis=1))
    chans.append(np.linalg.norm(a[:, [COLS["wx"], COLS["wy"], COLS["wz"]]], axis=1))
    n = len(a)
    for start in range(0, n - WIN + 1, STEP):
        vec = []
        for ch in chans:
            vec.extend(_channel_feats(ch[start:start + WIN]))
        yield np.asarray(vec), session.subject, session.exercise, session.wrist


def build_matrix(sessions):
    """Feature matrix + label arrays for a list of sessions."""
    X, subj, ex, wrist = [], [], [], []
    for s in sessions:
        for vec, sb, e, w in window_features(s):
            X.append(vec); subj.append(sb); ex.append(e); wrist.append(w)
    return (np.vstack(X), np.asarray(subj), np.asarray(ex), np.asarray(wrist))
