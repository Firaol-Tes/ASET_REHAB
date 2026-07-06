"""Experiment 3: learning exercises as Dynamic Movement Primitives.

Pipeline per session: heading-normalized rotation-vector trajectory ->
repetition segmentation -> median-duration rep as demonstration ->
50-basis DMP fit -> reconstruction error (deg) against the original.

Also demonstrates the clinical scaling that motivates DMPs: the same
learned primitive replayed at half speed (tau x2) and reduced range of
motion (goal scaled to 60%).

Outputs:
  results/exp3_dmp.csv                per-session fit quality
  results/primitives/<exercise>.npz   representative primitive per
                                      exercise (for the ROS 2 stage)
  results/figures/exp3_dmp.png
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physiotwin.data import index_sessions
from physiotwin.dmp import DMP
from physiotwin.reps import segment_reps, session_rotvec

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FS = 100.0
DT = 1.0 / FS


def fit_session(session):
    rotvec, theta, ref = session_rotvec(session)
    reps = segment_reps(theta)
    if not reps:
        return None
    # median-duration repetition = the representative demonstration
    reps.sort(key=lambda r: r[1] - r[0])
    s, e = reps[len(reps) // 2]
    demo = rotvec[s:e]
    dmp = DMP(n_basis=50).fit(demo, DT)
    recon = dmp.rollout(DT)
    n = min(len(demo), len(recon))
    rmse = np.degrees(np.sqrt(np.mean((demo[:n] - recon[:n]) ** 2)))
    return {
        "session": session, "demo": demo, "dmp": dmp, "recon": recon,
        "ref": ref,
        "n_reps": len(reps), "rep_dur_s": (e - s) / FS,
        "rmse_deg": float(rmse),
    }


def main():
    rows, fits = [], {}
    for s in index_sessions():
        r = fit_session(s)
        if r is None:
            print(f"skip (no reps found): {s.subject} {s.exercise} {s.wrist}")
            continue
        rows.append({
            "subject": s.subject, "exercise": s.exercise, "wrist": s.wrist,
            "n_reps": r["n_reps"], "rep_dur_s": round(r["rep_dur_s"], 2),
            "rmse_deg": round(r["rmse_deg"], 2),
        })
        key = s.exercise
        if key not in fits or r["rmse_deg"] < fits[key]["rmse_deg"]:
            fits[key] = r

    with open(os.path.join(RESULTS, "exp3_dmp.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\nfitted {len(rows)} sessions "
          f"({len(rows) / len(index_sessions()):.0%} of dataset)")
    print("=== per exercise: median DMP reconstruction RMSE (deg) ===")
    exercises = sorted({r["exercise"] for r in rows})
    for ex in exercises:
        e_rmse = [r["rmse_deg"] for r in rows if r["exercise"] == ex]
        e_reps = [r["n_reps"] for r in rows if r["exercise"] == ex]
        print(f"{ex:32s} rmse {np.median(e_rmse):5.2f} deg  "
              f"(n_sessions {len(e_rmse)}, total reps {sum(e_reps)})")

    # export representative primitives for the robot stage
    prim_dir = os.path.join(RESULTS, "primitives")
    os.makedirs(prim_dir, exist_ok=True)
    for ex, r in fits.items():
        np.savez(os.path.join(prim_dir, f"{ex}.npz"),
                 demo=r["demo"], ref=r["ref"],
                 weights=r["dmp"].w, y0=r["dmp"].y0,
                 goal=r["dmp"].g, tau=r["dmp"].tau,
                 subject=r["session"].subject, wrist=r["session"].wrist)
    print(f"\nexported {len(fits)} primitives to results/primitives/")

    make_figure(fits)


def make_figure(fits):
    ex = "Bicep-Curl"
    r = fits[ex]
    demo, dmp = r["demo"], r["dmp"]
    t = np.arange(len(demo)) * DT

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    for d, lab in enumerate(["x", "y", "z"]):
        ax.plot(t, np.degrees(demo[:, d]), lw=2.2, alpha=0.35,
                color=f"C{d}")
        rec = dmp.rollout(DT)
        ax.plot(np.arange(len(rec)) * DT, np.degrees(rec[:, d]),
                lw=1.2, color=f"C{d}", label=f"rot-vec {lab}")
    ax.set_title(f"{ex}: demonstration (thick) vs DMP (thin)\n"
                 f"RMSE {r['rmse_deg']:.1f} deg", fontsize=10)
    ax.set_xlabel("time (s)"); ax.set_ylabel("rotation (deg)")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    base = dmp.rollout(DT)
    slow = dmp.rollout(DT, tau=dmp.tau * 2)
    fast = dmp.rollout(DT, tau=dmp.tau * 0.5)
    for traj, lab in [(base, "learned speed"), (slow, "half speed (tau x2)"),
                      (fast, "double speed (tau x0.5)")]:
        mag = np.degrees(np.linalg.norm(traj, axis=1))
        ax.plot(np.arange(len(traj)) * DT, mag, label=lab)
    ax.set_title("temporal scaling: same primitive,\npatient-adapted tempo",
                 fontsize=10)
    ax.set_xlabel("time (s)"); ax.set_ylabel("rotation angle (deg)")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    for k, lab in [(1.0, "full range"), (0.6, "60% range"),
                   (0.3, "30% range")]:
        traj = dmp.rollout(DT, goal=dmp.y0 + k * (dmp.g - dmp.y0))
        # amplitude illustration: scale the whole forcing by moving goal
        mag = np.degrees(np.linalg.norm(traj, axis=1))
        ax.plot(np.arange(len(traj)) * DT, mag, label=lab)
    ax.set_title("amplitude scaling: reduced range of\nmotion for early rehab",
                 fontsize=10)
    ax.set_xlabel("time (s)"); ax.set_ylabel("rotation angle (deg)")
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "exp3_dmp.png"), dpi=160)


if __name__ == "__main__":
    main()
