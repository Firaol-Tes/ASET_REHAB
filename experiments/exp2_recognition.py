"""Experiment 2: Exercise recognition baseline.

(a) LOSO: leave-one-subject-out 12-class recognition, both wrists pooled.
    Window-level and session-level (majority vote) accuracy.
(b) Cross-wrist: train on right-wrist windows, test on left (and the
    reverse), restricted to the 10 exercises recorded on both wrists.
    The gap versus within-wrist performance quantifies how wrist-
    dependent the models are — motivation for mirror-aware training.

Outputs:
  results/exp2_loso.csv, results/exp2_crosswrist.csv
  results/figures/exp2_confusion.png
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from physiotwin.data import index_sessions
from physiotwin.features import build_matrix

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
RNG = 42


def rf():
    return RandomForestClassifier(
        n_estimators=300, random_state=RNG, n_jobs=-1, min_samples_leaf=2)


def loso(X, subj, ex, sess_id):
    subjects = sorted(set(subj))
    rows, all_true, all_pred = [], [], []
    for held in subjects:
        tr, te = subj != held, subj == held
        clf = rf().fit(X[tr], ex[tr])
        pred = clf.predict(X[te])
        acc_w = accuracy_score(ex[te], pred)
        # session-level majority vote
        s_true, s_pred = [], []
        te_sess = sess_id[te]
        for sid in set(te_sess):
            s_true.append(ex[te][te_sess == sid][0])
            vals, cnt = np.unique(pred[te_sess == sid], return_counts=True)
            s_pred.append(vals[np.argmax(cnt)])
        acc_s = accuracy_score(s_true, s_pred)
        rows.append({"subject": held, "window_acc": acc_w,
                     "session_acc": acc_s, "n_windows": int(te.sum())})
        all_true.extend(ex[te]); all_pred.extend(pred)
        print(f"LOSO {held}: window {acc_w:.3f}  session {acc_s:.3f}")
    return rows, np.asarray(all_true), np.asarray(all_pred)


def cross_wrist(X, subj, ex, wrist):
    both = {e for e in set(ex)
            if {"LW", "RW"} <= set(wrist[ex == e])}
    m = np.isin(ex, list(both))
    Xb, sb, eb, wb = X[m], subj[m], ex[m], wrist[m]
    out = []
    for train_w, test_w in [("RW", "LW"), ("LW", "RW")]:
        tr, te = wb == train_w, wb == test_w
        clf = rf().fit(Xb[tr], eb[tr])
        acc = accuracy_score(eb[te], clf.predict(Xb[te]))
        # within-wrist reference: LOSO inside the test wrist
        accs_ref = []
        for held in sorted(set(sb)):
            tr2 = te & (sb != held)
            te2 = te & (sb == held)
            if te2.sum() == 0 or tr2.sum() == 0:
                continue
            c2 = rf().fit(Xb[tr2], eb[tr2])
            accs_ref.append(accuracy_score(eb[te2], c2.predict(Xb[te2])))
        out.append({"train": train_w, "test": test_w,
                    "cross_acc": acc, "within_loso_acc": float(np.mean(accs_ref))})
        print(f"train {train_w} -> test {test_w}: cross {acc:.3f} "
              f"(within-wrist LOSO ref {np.mean(accs_ref):.3f})")
    return out


def main():
    sessions = index_sessions()
    # build matrix with a session id per window for majority voting
    X, subj, ex, wrist = build_matrix(sessions)
    sess_id = []
    for i, s in enumerate(sessions):
        n = sum(1 for _ in range(0, len(s.load()) - 256 + 1, 128))
        sess_id.extend([i] * n)
    sess_id = np.asarray(sess_id)
    assert len(sess_id) == len(X)
    print(f"{len(X)} windows, {len(set(ex))} classes, "
          f"{len(set(subj))} subjects, {X.shape[1]} features")

    rows, y_true, y_pred = loso(X, subj, ex, sess_id)
    with open(os.path.join(RESULTS, "exp2_loso.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    wacc = np.mean([r["window_acc"] for r in rows])
    sacc = np.mean([r["session_acc"] for r in rows])
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\nLOSO mean: window {wacc:.3f}, session {sacc:.3f}, macro-F1 {f1:.3f}")

    cw = cross_wrist(X, subj, ex, wrist)
    with open(os.path.join(RESULTS, "exp2_crosswrist.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(cw[0].keys()))
        w.writeheader(); w.writerows(cw)

    # confusion matrix figure (window level, all LOSO folds pooled)
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short = [l.replace("-", " ") for l in labels]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if cm[i, j] > 0.6 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"12-class exercise recognition, LOSO "
                 f"(window acc {wacc:.1%}, session acc {sacc:.1%})")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "exp2_confusion.png"), dpi=160)


if __name__ == "__main__":
    main()
