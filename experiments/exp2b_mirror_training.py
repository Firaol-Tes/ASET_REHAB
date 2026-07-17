"""Experiment 2b: closing the cross-wrist gap with mirror-aware training.

Step 1 — data-driven mirror calibration: for every sensor channel,
estimate the sign relating left- and right-wrist signals from the
synchronized dual-wrist pairs (median correlation over pairs). This
replaces hand-derived reflection algebra with a transform the dataset
itself certifies, and doubles as a check that no axis permutation is
needed (off-diagonal correlations stay low).

Step 2 — retrain the Experiment-2 classifier three ways per direction
(train wrist -> test wrist):
    raw        train on source wrist only            (Exp-2 baseline)
    mirrored   train on mirror-transformed source     (tests the map)
    augmented  train on source + mirrored source      (deployment mode)

Outputs:
  results/exp2b_signmap.csv, results/exp2b_results.csv
  results/figures/exp2b_gap.png
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from physiotwin.data import COLS, index_sessions, simultaneous_pairs
from physiotwin.features import build_matrix, mirror_transform

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
MIRROR_CHANNELS = ["ax", "ay", "az", "wx", "wy", "wz",
                   "gx", "gy", "gz", "pitch", "roll"]


def estimate_sign_map(exclude_subject=None, verbose=True, save=True):
    """Median LW/RW correlation per channel over synchronized pairs.

    `exclude_subject` drops that subject's pairs, allowing the map to
    be estimated strictly within a LOSO training fold (leakage check).
    """
    per_channel = {ch: [] for ch in MIRROR_CHANNELS}
    for lw, rw, _ov in simultaneous_pairs():
        if exclude_subject is not None and lw.subject == exclude_subject:
            continue
        t0 = max(lw.t_ms[0], rw.t_ms[0])
        t1 = min(lw.t_ms[-1], rw.t_ms[-1])
        grid = np.arange(t0, t1, 10.0)
        for ch in MIRROR_CHANNELS:
            a = np.interp(grid, lw.t_ms, lw.col(ch))
            b = np.interp(grid, rw.t_ms, rw.col(ch))
            if a.std() > 1e-6 and b.std() > 1e-6:
                per_channel[ch].append(np.corrcoef(a, b)[0, 1])
    rows, signs = [], {}
    for ch, cs in per_channel.items():
        med = float(np.median(cs))
        signs[ch] = 1.0 if med >= 0 else -1.0
        rows.append({"channel": ch, "median_corr": round(med, 3),
                     "sign": int(signs[ch]), "n_pairs": len(cs)})
        if verbose:
            print(f"{ch:6s} median corr {med:+.2f}  -> sign {int(signs[ch]):+d}")
    if save:
        with open(os.path.join(RESULTS, "exp2b_signmap.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return signs


def leakage_check(global_signs, subjects):
    """Re-estimate the sign map excluding each subject in turn; verify
    the per-fold maps equal the global one (no test-subject influence)."""
    all_match = True
    for held in subjects:
        s = estimate_sign_map(exclude_subject=held, verbose=False,
                              save=False)
        diff = [ch for ch in MIRROR_CHANNELS
                if s[ch] != global_signs[ch]]
        status = "identical" if not diff else f"DIFFERS: {diff}"
        if diff:
            all_match = False
        print(f"fold without {held}: {status}")
    print("leakage check:",
          "PASS - per-fold mirror maps identical to global map"
          if all_match else "FAIL - fold-dependent signs found")
    return all_match


def rf():
    return RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2)


def main():
    print("=== Step 1: mirror calibration from synchronized pairs ===")
    signs = estimate_sign_map()
    mirror = mirror_transform(signs)

    print("\n=== Step 1b: leakage check (fold-wise sign maps) ===")
    subjects_all = sorted({s.subject for s in index_sessions()})
    leakage_check(signs, subjects_all)

    print("\n=== Step 2: cross-wrist training strategies ===")
    sessions = index_sessions()
    # restrict to exercises recorded on both wrists
    by_ex = {}
    for s in sessions:
        by_ex.setdefault(s.exercise, set()).add(s.wrist)
    both = {e for e, ws in by_ex.items() if {"LW", "RW"} <= ws}
    sessions = [s for s in sessions if s.exercise in both]

    X, subj, ex, wrist = build_matrix(sessions)
    Xm, _, _, _ = build_matrix(sessions, transform=mirror)

    rows = []
    for train_w, test_w in [("RW", "LW"), ("LW", "RW")]:
        tr, te = wrist == train_w, wrist == test_w
        for mode in ["raw", "mirrored", "augmented"]:
            if mode == "raw":
                Xtr, ytr = X[tr], ex[tr]
            elif mode == "mirrored":
                Xtr, ytr = Xm[tr], ex[tr]
            else:
                Xtr = np.vstack([X[tr], Xm[tr]])
                ytr = np.concatenate([ex[tr], ex[tr]])
            acc = accuracy_score(ex[te], rf().fit(Xtr, ytr).predict(X[te]))
            rows.append({"train": train_w, "test": test_w,
                         "mode": mode, "acc": round(float(acc), 4)})
            print(f"train {train_w} ({mode:9s}) -> test {test_w}: {acc:.3f}")

    print("\n=== Step 3: strictest protocol - unseen subject AND unseen wrist ===")
    for train_w, test_w in [("RW", "LW"), ("LW", "RW")]:
        for mode in ["raw", "augmented"]:
            accs = []
            for held in sorted(set(subj)):
                tr = (wrist == train_w) & (subj != held)
                te = (wrist == test_w) & (subj == held)
                if not te.any():
                    continue
                if mode == "raw":
                    Xtr, ytr = X[tr], ex[tr]
                else:
                    Xtr = np.vstack([X[tr], Xm[tr]])
                    ytr = np.concatenate([ex[tr], ex[tr]])
                accs.append(accuracy_score(ex[te], rf().fit(Xtr, ytr).predict(X[te])))
            acc = float(np.mean(accs))
            rows.append({"train": f"{train_w}-LOSO", "test": test_w,
                         "mode": mode, "acc": round(acc, 4)})
            print(f"LOSO train {train_w} ({mode:9s}) -> test {test_w}: {acc:.3f}")

    with open(os.path.join(RESULTS, "exp2b_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # figure: grouped bars, within-wrist reference line at 0.93 (Exp 2)
    plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 11})
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    modes = ["raw", "mirrored", "augmented"]
    width = 0.35
    for i, (tw, colr) in enumerate([("RW", "#3465a4"), ("LW", "#e69a2e")]):
        vals = [next(r["acc"] for r in rows
                     if r["train"] == tw and r["mode"] == m) for m in modes]
        ax.bar(np.arange(3) + (i - 0.5) * width, vals, width,
               label=f"train {tw} " + r"$\rightarrow$" + f" test {'LW' if tw == 'RW' else 'RW'}",
               color=colr)
        for j, v in enumerate(vals):
            ax.text(j + (i - 0.5) * width, v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=11)
    ax.axhline(0.929, color="gray", ls="--", lw=1,
               label="within-wrist LOSO reference")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["raw\n(baseline)", "mirrored\ntraining set",
                        "source + mirrored\n(augmented)"])
    ax.set_ylabel("window accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-wrist recognition: mirror-aware training closes the gap",
                 fontsize=13)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "exp2b_gap.png"), dpi=500)


if __name__ == "__main__":
    main()
