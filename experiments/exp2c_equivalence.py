"""Experiment 2c: is mirror-augmented cross-wrist recognition statistically
equivalent to the within-wrist reference?

Experiment 2b establishes that mirror-augmented training restores
cross-wrist accuracy to roughly the within-wrist level under the
strictest protocol (unseen subject AND unseen wrist). "Roughly equal"
is an *equivalence* claim, and a non-significant difference test does
not establish equivalence -- so this script reports both:

  1. a paired test of difference (Wilcoxon signed-rank over subjects), and
  2. a two-one-sided-tests (TOST) equivalence test against a
     pre-specified margin of 2 accuracy points.

Both arms are evaluated on the *same* held-out subject and the same
test-wrist windows, so the folds are genuinely paired:

  augmented   train on (source wrist + mirrored source wrist),
              all subjects except the held-out one
  within      train on the test wrist itself,
              all subjects except the held-out one   (reference)

Restricted, as in Exp 2b, to the ten exercises recorded on both wrists.

Outputs:
  results/exp2c_folds.csv        per-subject paired accuracies
  results/exp2c_equivalence.csv  test statistics
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from physiotwin.data import index_sessions
from physiotwin.features import build_matrix, mirror_transform
from exp2b_mirror_training import estimate_sign_map

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
MARGIN = 0.02          # pre-specified equivalence margin (2 accuracy points)


def rf():
    return RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=2)


def main():
    signs = estimate_sign_map(verbose=False, save=False)
    mirror = mirror_transform(signs)

    sessions = index_sessions()
    by_ex = {}
    for s in sessions:
        by_ex.setdefault(s.exercise, set()).add(s.wrist)
    both = {e for e, ws in by_ex.items() if {"LW", "RW"} <= ws}
    sessions = [s for s in sessions if s.exercise in both]

    X, subj, ex, wrist = build_matrix(sessions)
    Xm, _, _, _ = build_matrix(sessions, transform=mirror)
    print(f"{len(X)} windows, {len(both)} classes, "
          f"{len(set(subj))} subjects", flush=True)

    folds = []
    for train_w, test_w in [("RW", "LW"), ("LW", "RW")]:
        for held in sorted(set(subj)):
            te = (wrist == test_w) & (subj == held)
            if not te.any():
                continue
            # arm 1: mirror-augmented, trained on the *other* wrist
            tr_a = (wrist == train_w) & (subj != held)
            Xtr = np.vstack([X[tr_a], Xm[tr_a]])
            ytr = np.concatenate([ex[tr_a], ex[tr_a]])
            acc_aug = accuracy_score(ex[te], rf().fit(Xtr, ytr).predict(X[te]))
            # arm 2: within-wrist reference, trained on the test wrist
            tr_w = (wrist == test_w) & (subj != held)
            acc_within = accuracy_score(
                ex[te], rf().fit(X[tr_w], ex[tr_w]).predict(X[te]))
            folds.append({"train": train_w, "test": test_w, "subject": held,
                          "acc_augmented": round(float(acc_aug), 4),
                          "acc_within": round(float(acc_within), 4),
                          "delta": round(float(acc_aug - acc_within), 4),
                          "n_windows": int(te.sum())})
            print(f"{train_w}->{test_w} {held}: aug {acc_aug:.3f}  "
                  f"within {acc_within:.3f}  d {acc_aug - acc_within:+.3f}",
                  flush=True)

    with open(os.path.join(RESULTS, "exp2c_folds.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(folds[0].keys()))
        w.writeheader(); w.writerows(folds)

    rows = []
    groups = [("RW", "LW"), ("LW", "RW"), (None, "pooled")]
    for train_w, label in groups:
        sel = [f for f in folds
               if train_w is None or f["train"] == train_w]
        a = np.array([f["acc_augmented"] for f in sel])
        b = np.array([f["acc_within"] for f in sel])
        d = a - b
        # paired difference test
        try:
            w_stat, p_diff = stats.wilcoxon(a, b)
        except ValueError:            # all differences zero
            w_stat, p_diff = float("nan"), 1.0
        # TOST equivalence against +/- MARGIN, paired t on the differences
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        t_lo = (d.mean() + MARGIN) / se
        t_hi = (d.mean() - MARGIN) / se
        p_lo = stats.t.sf(t_lo, n - 1)      # H0: delta <= -margin
        p_hi = stats.t.cdf(t_hi, n - 1)     # H0: delta >= +margin
        p_tost = max(p_lo, p_hi)
        ci = stats.t.interval(0.90, n - 1, loc=d.mean(), scale=se)
        rows.append({
            "comparison": f"{train_w}->{label}" if train_w else "pooled",
            "n_folds": n,
            "mean_augmented": round(float(a.mean()), 4),
            "mean_within": round(float(b.mean()), 4),
            "mean_delta": round(float(d.mean()), 4),
            "ci90_lo": round(float(ci[0]), 4),
            "ci90_hi": round(float(ci[1]), 4),
            "wilcoxon_p": round(float(p_diff), 4),
            "tost_margin": MARGIN,
            "tost_p": round(float(p_tost), 4),
            "equivalent": bool(p_tost < 0.05),
        })
        print(f"\n[{rows[-1]['comparison']}] n={n}  "
              f"aug {a.mean():.4f} vs within {b.mean():.4f}  "
              f"delta {d.mean():+.4f}", flush=True)
        print(f"  90% CI on delta: [{ci[0]:+.4f}, {ci[1]:+.4f}]", flush=True)
        print(f"  Wilcoxon p={p_diff:.4f} (difference)", flush=True)
        print(f"  TOST p={p_tost:.4f} vs +/-{MARGIN:.0%} -> "
              f"{'EQUIVALENT' if p_tost < 0.05 else 'not established'}",
              flush=True)

    with open(os.path.join(RESULTS, "exp2c_equivalence.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
