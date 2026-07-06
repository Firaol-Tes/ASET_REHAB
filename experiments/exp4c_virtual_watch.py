"""Experiment 4c: virtual-watch verification of the robot's execution.

A virtual Apple Watch is strapped to the UR5e's wrist: from the joint
states recorded during Gazebo execution (Exp 4b) we reconstruct the
tool motion with forward kinematics, re-express it in the demonstrating
watch's frame, compensate the therapy-pace slowdown, and synthesize the
Device-Motion channels (attitude, gravity, rotation rate, user
acceleration). The Experiment-2 classifier — trained exclusively on
human recordings — then predicts which exercise the robot performed.

A correct prediction closes the loop: record human -> learn primitive
-> execute on robot -> the same sensing pipeline recognizes the result.

Outputs:
  results/exp4c_verification.csv
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scipy.spatial.transform import Rotation
from sklearn.ensemble import RandomForestClassifier

from physiotwin import ur5e
from physiotwin.data import COLS, index_sessions
from physiotwin.features import build_matrix, window_features

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FS = 100.0


class SyntheticSession:
    """Duck-typed Session carrying a synthesized DM record array."""

    def __init__(self, arr, exercise):
        self._arr = arr
        self.subject, self.exercise, self.wrist = "ROBOT", exercise, "RW"

    def load(self):
        return self._arr


def synthesize(ex: str) -> SyntheticSession | None:
    gz = os.path.join(RESULTS, "gz_exec", f"{ex}.npz")
    prim = os.path.join(RESULTS, "primitives", f"{ex}.npz")
    if not (os.path.exists(gz) and os.path.exists(prim)):
        return None
    d = np.load(gz)
    rec = d["recorded"]
    t = rec[:, 0] - rec[0, 0]

    # speed compensation: measured joint motion back to human tempo
    slow = {}
    with open(os.path.join(RESULTS, "exp4a_feasibility.csv")) as f:
        for row in csv.DictReader(f):
            slow[row["exercise"]] = max(2.0, float(row["required_slowdown"]))
    factor = slow[ex]
    t_h = t / factor                       # human-tempo timeline
    grid = np.arange(t_h[0], t_h[-1], 1 / FS)
    q = np.column_stack([np.interp(grid, t_h, rec[:, 1 + j])
                         for j in range(6)])

    # tool motion via FK, re-expressed as watch attitude
    T0 = ur5e.fk(ur5e.HOME)
    R0_tool = Rotation.from_matrix(T0[:3, :3])
    poses = [ur5e.fk(qi) for qi in q]
    R_tool = Rotation.from_matrix(np.stack([P[:3, :3] for P in poses]))
    p = np.stack([P[:3, 3] for P in poses])
    R_rel = R_tool * R0_tool.inv()
    R0_watch = Rotation.from_rotvec(np.load(prim)["demo"][0])
    R_watch = R0_watch * R_rel             # attitude of the virtual watch

    # Device-Motion channels in the device frame
    Rm = R_watch.as_matrix()
    grav = np.einsum("nji,j->ni", Rm, np.array([0.0, 0.0, -1.0]))
    dR = np.gradient(Rm, 1 / FS, axis=0)
    Wm = np.einsum("nij,nkj->nik", dR, Rm)          # omega_world skew
    om_w = np.stack([Wm[:, 2, 1], Wm[:, 0, 2], Wm[:, 1, 0]], axis=1)
    gyro = np.einsum("nji,nj->ni", Rm, om_w)
    acc_w = np.gradient(np.gradient(p, 1 / FS, axis=0), 1 / FS, axis=0)
    user_acc = np.einsum("nji,nj->ni", Rm, acc_w) / 9.81
    eul = R_watch.as_euler("ZXY")           # yaw, pitch, roll (Apple-like)

    arr = np.zeros((len(grid), 19))
    arr[:, COLS["pitch"]], arr[:, COLS["roll"]] = eul[:, 1], eul[:, 2]
    for i, ch in enumerate(["wx", "wy", "wz"]):
        arr[:, COLS[ch]] = gyro[:, i]
    for i, ch in enumerate(["ax", "ay", "az"]):
        arr[:, COLS[ch]] = user_acc[:, i]
    for i, ch in enumerate(["gx", "gy", "gz"]):
        arr[:, COLS[ch]] = grav[:, i]
    return SyntheticSession(arr, ex)


def main():
    print("training verifier on all human windows...")
    X, _, ex_lab, _ = build_matrix(index_sessions())
    clf = RandomForestClassifier(n_estimators=300, random_state=42,
                                 n_jobs=-1, min_samples_leaf=2)
    clf.fit(X, ex_lab)

    rows = []
    for ex in sorted({e for e in ex_lab}):
        syn = synthesize(ex)
        if syn is None:
            continue
        Xr = np.vstack([v for v, *_ in window_features(syn)])
        pred = clf.predict(Xr)
        vals, cnt = np.unique(pred, return_counts=True)
        top = vals[np.argmax(cnt)]
        vote = cnt.max() / cnt.sum()
        ok = top == ex
        rows.append({"executed": ex, "predicted": top,
                     "vote_frac": round(float(vote), 3),
                     "n_windows": len(Xr), "correct": ok})
        print(f"{ex:32s} -> {top:32s} vote {vote:5.1%}  "
              f"{'OK' if ok else 'X'}")

    acc = np.mean([r["correct"] for r in rows])
    print(f"\nverification accuracy: {acc:.1%} ({int(acc * len(rows))}/{len(rows)})")
    with open(os.path.join(RESULTS, "exp4c_verification.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
