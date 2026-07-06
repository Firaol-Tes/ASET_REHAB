"""Dataset access layer for the HCT dual-wrist Apple Watch rehab dataset.

Subjects are anonymized to S01..Snn (stable, sorted by folder name).
Real names never leave this module.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field

import numpy as np

DATA_ROOT = os.environ.get(
    "PHYSIOTWIN_DATA", "/home/f/Downloads/HCT_Students/0-UnCompressed"
)

# Pilot / non-cohort folders excluded from all experiments.
EXCLUDED_FOLDERS = {"Yassine Benachour"}

_FNAME = re.compile(
    r"(\d{8}-\d{6})_DM_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([A-Z]{2})_([A-Z]{2})_(\d+)\.csv$"
)

# DM CSV column indices
COLS = {
    "t_ms": 2, "pitch": 3, "roll": 4, "yaw": 5,
    "qx": 6, "qy": 7, "qw": 8, "qz": 9,          # note: file order is X,Y,W,Z
    "wx": 10, "wy": 11, "wz": 12,
    "ax": 13, "ay": 14, "az": 15,
    "gx": 16, "gy": 17, "gz": 18,
}


@dataclass
class Session:
    subject: str          # anonymized, e.g. "S07"
    condition: str
    exercise: str
    wrist: str            # "LW" | "RW"
    crown: str
    timestamp: str
    path: str
    _arr: np.ndarray | None = field(default=None, repr=False)

    def load(self) -> np.ndarray:
        """Full DM record as float array (n, 19)."""
        if self._arr is None:
            rows = []
            with open(self.path) as f:
                r = csv.reader(f)
                next(r)
                for row in r:
                    if row and row[0].strip():
                        rows.append([float(v) for v in row[2:]])
            a = np.asarray(rows)
            # re-insert two dummy cols so COLS indices apply unchanged
            self._arr = np.hstack([np.zeros((len(a), 2)), a])
        return self._arr

    def col(self, name: str) -> np.ndarray:
        return self.load()[:, COLS[name]]

    @property
    def t_ms(self) -> np.ndarray:
        return self.col("t_ms")

    @property
    def quat_wxyz(self) -> np.ndarray:
        """Quaternions as (n, 4) in w,x,y,z order (file stores x,y,w,z)."""
        a = self.load()
        return a[:, [COLS["qw"], COLS["qx"], COLS["qy"], COLS["qz"]]]

    @property
    def gyro(self) -> np.ndarray:
        a = self.load()
        return a[:, [COLS["wx"], COLS["wy"], COLS["wz"]]]


def index_sessions() -> list[Session]:
    """Scan the dataset, anonymize subjects, return all cohort DM sessions."""
    folders = sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d not in EXCLUDED_FOLDERS
        and not d.startswith("_")
    )
    anon = {name: f"S{i + 1:02d}" for i, name in enumerate(folders)}
    sessions = []
    for folder in folders:
        for fn in sorted(os.listdir(os.path.join(DATA_ROOT, folder))):
            m = _FNAME.search(fn)
            if not m:
                continue
            ts, _user, cond, ex, _status, wrist, crown, _hz = m.groups()
            sessions.append(Session(
                subject=anon[folder], condition=cond, exercise=ex,
                wrist=wrist, crown=crown, timestamp=ts,
                path=os.path.join(DATA_ROOT, folder, fn),
            ))
    return sessions


def simultaneous_pairs(min_overlap_s: float = 10.0):
    """Yield (lw, rw, overlap_s) for sessions recorded at the same time."""
    by_key: dict[tuple, list[Session]] = {}
    for s in index_sessions():
        by_key.setdefault((s.subject, s.condition, s.exercise), []).append(s)
    for group in by_key.values():
        for lw in (s for s in group if s.wrist == "LW"):
            for rw in (s for s in group if s.wrist == "RW"):
                t0 = max(lw.t_ms[0], rw.t_ms[0])
                t1 = min(lw.t_ms[-1], rw.t_ms[-1])
                ov = (t1 - t0) / 1000.0
                if ov >= min_overlap_s:
                    yield lw, rw, ov
