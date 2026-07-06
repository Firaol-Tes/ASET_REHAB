"""Experiment 1: Bilateral mirror-symmetry validation.

For every synchronized left/right wrist pair, compare frame-invariant
motion profiles. High agreement means the unaffected arm's motion is a
faithful template for the contralateral arm — the premise PhysioTwin's
self-teaching mode rests on.

Outputs:
  results/exp1_pairs.csv        one row per session pair
  results/exp1_by_exercise.csv  aggregated per exercise
  results/figures/exp1_profiles.png  example overlaid profiles
  results/figures/exp1_summary.png   distribution per exercise
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physiotwin.data import simultaneous_pairs
from physiotwin.motion import compare_pair

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    rows, profiles = [], {}
    for lw, rw, _ov in simultaneous_pairs():
        res = compare_pair(lw, rw)
        profiles[(res["subject"], res["exercise"])] = res.pop("profiles")
        rows.append(res)
        print(f"{res['subject']} {res['exercise']:32s} "
              f"r_theta={res['r_theta']:5.2f} r_speed={res['r_speed']:5.2f} "
              f"rmse={res['rmse_deg']:5.1f}deg lag={res['lag_s']:+.2f}s")

    fields = ["subject", "exercise", "condition", "overlap_s",
              "r_theta", "r_speed", "rmse_deg", "lag_s"]
    with open(os.path.join(RESULTS, "exp1_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)

    # aggregate per exercise
    exercises = sorted({r["exercise"] for r in rows})
    agg = []
    for ex in exercises:
        sel = [r for r in rows if r["exercise"] == ex]
        agg.append({
            "exercise": ex, "n_pairs": len(sel),
            "r_theta_mean": np.mean([r["r_theta"] for r in sel]),
            "r_theta_std": np.std([r["r_theta"] for r in sel]),
            "r_speed_mean": np.mean([r["r_speed"] for r in sel]),
            "rmse_deg_mean": np.mean([r["rmse_deg"] for r in sel]),
        })
    with open(os.path.join(RESULTS, "exp1_by_exercise.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(agg[0].keys()))
        w.writeheader()
        for a in agg:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v)
                        for k, v in a.items()})

    print("\n=== Per-exercise summary ===")
    for a in agg:
        print(f"{a['exercise']:32s} n={a['n_pairs']:2d} "
              f"r_theta={a['r_theta_mean']:.2f}+-{a['r_theta_std']:.2f} "
              f"r_speed={a['r_speed_mean']:.2f} rmse={a['rmse_deg_mean']:.1f}deg")

    make_figures(rows, profiles, exercises)


def make_figures(rows, profiles, exercises):
    figdir = os.path.join(RESULTS, "figures")

    # Fig A: example theta profiles, best pair of four contrasting exercises
    show = ["Bicep-Curl", "Scaption", "Wand-Flexion", "External-Rotation"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=False)
    for ax, ex in zip(axes.flat, show):
        sel = [r for r in rows if r["exercise"] == ex]
        best = max(sel, key=lambda r: r["r_theta"])
        grid, th_l, th_r = profiles[(best["subject"], best["exercise"])]
        t = (grid - grid[0]) / 1000.0
        ax.plot(t, np.degrees(th_l), label="left wrist", lw=1.2)
        ax.plot(t, np.degrees(th_r), label="right wrist", lw=1.2, alpha=0.8)
        ax.set_title(f"{ex}  (r = {best['r_theta']:.2f})", fontsize=10)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("rotation from start (deg)")
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Synchronized bilateral motion: rotation-angle profiles, "
                 "left vs right wrist", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "exp1_profiles.png"), dpi=160)

    # Fig B: r_theta distribution per exercise
    fig, ax = plt.subplots(figsize=(10, 4.5))
    data = [[r["r_theta"] for r in rows if r["exercise"] == ex]
            for ex in exercises]
    ax.boxplot(data, labels=[e.replace("-", "\n") for e in exercises],
               showmeans=True)
    ax.set_ylabel("correlation of rotation profiles (r)")
    ax.set_title("Bilateral mirror symmetry per exercise "
                 "(all subjects, synchronized dual-wrist pairs)")
    ax.axhline(0.9, color="gray", ls=":", lw=0.8)
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "exp1_summary.png"), dpi=160)


if __name__ == "__main__":
    main()
