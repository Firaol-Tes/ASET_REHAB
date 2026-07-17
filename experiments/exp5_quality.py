"""Experiment 5: grading movement quality against the population envelope.

(a) False-alarm rate on genuine movement, leave-one-subject-out: the
    envelope is built from 10 subjects' repetitions and applied to the
    held-out subject's. A good grader rarely flags healthy reps.
(b) Fault detection: the DMP scaling knobs generate controlled faults
    from each exercise's primitive — too fast (tau/2), reduced range
    (goal*0.5), tremor (4 Hz oscillation added), truncated (stopped at
    60%). Detection = the faulty rep is flagged by an envelope built
    from ALL subjects.

Outputs:
  results/exp5_false_alarms.csv, results/exp5_detection.csv
  results/figures/exp5_quality.png
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physiotwin.data import index_sessions
from physiotwin.quality import FEATURES, Envelope, rep_features
from physiotwin.reps import segment_reps, session_rotvec

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FS = 100.0
DT = 1.0 / FS


def collect_rep_features():
    """(exercise, subject) -> feature dicts and raw theta segments."""
    table, reps = defaultdict(list), defaultdict(list)
    for s in index_sessions():
        rotvec, theta, _ref = session_rotvec(s)
        for a, b in segment_reps(theta):
            table[(s.exercise, s.subject)].append(rep_features(theta[a:b]))
            reps[(s.exercise, s.subject)].append(theta[a:b])
    return table, reps


def false_alarm_rates(table):
    rows = []
    exercises = sorted({ex for ex, _ in table})
    for ex in exercises:
        subjects = sorted({sb for e, sb in table if e == ex})
        flagged = total = 0
        for held in subjects:
            train = [f for (e, sb), fl in table.items()
                     if e == ex and sb != held for f in fl]
            test = table[(ex, held)]
            if len(train) < 20 or not test:
                continue
            env = Envelope(train)
            flagged += sum(1 for f in test if env.flag(f))
            total += len(test)
        rows.append({"exercise": ex, "n_reps": total,
                     "false_alarm_rate": round(flagged / total, 4)})
    return rows


def fault_variants(theta: np.ndarray) -> dict[str, np.ndarray]:
    """Controlled faults injected into one genuine repetition, keeping
    the natural texture of human movement."""
    t = np.arange(len(theta)) * DT
    return {
        "too_fast": theta[::2],
        "reduced_rom": theta[0] + 0.5 * (theta - theta[0]),
        "tremor": theta + np.radians(3.0) * np.sin(2 * np.pi * 4.0 * t),
        "truncated": theta[: max(50, int(0.6 * len(theta)))],
    }


FAULTS = list(fault_variants(np.zeros(100)).keys())


def detection_rates(table, reps_by_cell):
    """LOSO: envelope from other subjects, applied to the held-out
    subject's faulted repetitions. Returns detection rate per
    exercise x fault."""
    rows = []
    exercises = sorted({ex for ex, _ in table})
    for ex in exercises:
        subjects = sorted({sb for e, sb in table if e == ex})
        hit = {f: 0 for f in FAULTS}
        tot = 0
        for held in subjects:
            train = [f for (e, sb), fl in table.items()
                     if e == ex and sb != held for f in fl]
            if len(train) < 20:
                continue
            env = Envelope(train)
            for theta in reps_by_cell.get((ex, held), []):
                tot += 1
                for name, bad in fault_variants(theta).items():
                    if env.flag(rep_features(bad)):
                        hit[name] += 1
        row = {"exercise": ex, "n_reps": tot}
        row.update({f: round(hit[f] / tot, 4) if tot else 0.0
                    for f in FAULTS})
        rows.append(row)
    return rows


def personalized_rates(table, reps_by_cell, min_reps=8):
    """Hybrid envelope: centered on the subject's own calibration reps
    (every other rep), with the population's interval width as a floor —
    a handful of calibration reps can place the center but cannot
    reliably estimate spread. Tested on the remaining reps."""
    flagged = 0
    hit = {f: 0 for f in FAULTS}
    tot = 0
    for (ex, sb), feats in table.items():
        if len(feats) < min_reps:
            continue
        env = Envelope(feats[0::2])
        pop = Envelope([f for (e, s2), fl in table.items()
                        if e == ex and s2 != sb for f in fl])
        for f in env.lo:
            m = (env.lo[f] + env.hi[f]) / 2
            # floor: half the population spread — an individual's natural
            # variability is far smaller than the population's
            half = max(env.hi[f] - m, (pop.hi[f] - pop.lo[f]) / 4)
            env.lo[f], env.hi[f] = m - half, m + half
        thetas = reps_by_cell[(ex, sb)][1::2]
        for theta in thetas:
            tot += 1
            if env.flag(rep_features(theta)):
                flagged += 1
            for name, bad in fault_variants(theta).items():
                if env.flag(rep_features(bad)):
                    hit[name] += 1
    return {"false_alarm_rate": flagged / tot, "n_reps": tot,
            **{f: hit[f] / tot for f in FAULTS}}


def main():
    print("collecting per-repetition quality features...")
    table, reps_by_cell = collect_rep_features()
    n = sum(len(v) for v in table.values())
    print(f"{n} genuine repetitions across {len(table)} exercise-subject cells\n")

    fa = false_alarm_rates(table)
    print("=== false alarms on healthy reps (LOSO) ===")
    for r in fa:
        print(f"{r['exercise']:32s} {r['false_alarm_rate']:6.1%} of {r['n_reps']}")
    with open(os.path.join(RESULTS, "exp5_false_alarms.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, list(fa[0].keys()))
        w.writeheader(); w.writerows(fa)

    det = detection_rates(table, reps_by_cell)
    print("\n=== fault detection rates (LOSO, faults injected in human reps) ===")
    print(f"{'exercise':32s} " + " ".join(f"{k:>12s}" for k in FAULTS))
    for r in det:
        print(f"{r['exercise']:32s} " +
              " ".join(f"{r[k]:12.1%}" for k in FAULTS))
    overall = {k: sum(r[k] * r["n_reps"] for r in det) /
               sum(r["n_reps"] for r in det) for k in FAULTS}
    print(f"{'OVERALL':32s} " +
          " ".join(f"{overall[k]:12.1%}" for k in FAULTS))
    with open(os.path.join(RESULTS, "exp5_detection.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, list(det[0].keys()))
        w.writeheader(); w.writerows(det)

    pers = personalized_rates(table, reps_by_cell)
    print(f"\n=== personalized envelope (calibration reps of same subject, "
          f"n={pers['n_reps']}) ===")
    print(f"false alarms {pers['false_alarm_rate']:.1%} | " +
          " ".join(f"{k} {pers[k]:.1%}" for k in FAULTS))
    with open(os.path.join(RESULTS, "exp5_personalized.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, list(pers.keys()))
        w.writeheader(); w.writerow(pers)

    make_figure(table, det, reps_by_cell)


def make_figure(table, det, reps_by_cell):
    ex = "Bicep-Curl"
    train = [f for (e, _sb), fl in table.items() if e == ex for f in fl]
    env = Envelope(train)
    demo = max((th for (e, _s), ths in reps_by_cell.items() if e == ex
                for th in ths), key=len)

    plt.rcParams.update({"axes.labelsize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    rom = [f["rom_deg"] for f in train]
    dur = [f["dur_s"] for f in train]
    ax.scatter(dur, rom, s=12, alpha=0.35, label="genuine reps (11 subjects)")
    ax.axhspan(env.lo["rom_deg"], env.hi["rom_deg"], alpha=0.08, color="green")
    ax.axvspan(env.lo["dur_s"], env.hi["dur_s"], alpha=0.08, color="green")
    marks = {"too_fast": "v", "reduced_rom": "s", "tremor": "^",
             "truncated": "x"}
    for name, bad in fault_variants(demo).items():
        f = rep_features(bad)
        ax.scatter(f["dur_s"], f["rom_deg"], marker=marks[name], s=90,
                   label=f"fault: {name}", zorder=3)
    ax.set_xlabel("repetition duration (s)")
    ax.set_ylabel("range of motion (deg)")
    ax.set_title(f"{ex}: population envelope vs synthetic faults",
                 fontsize=13)
    ax.legend(fontsize=11, frameon=False)

    ax = axes[1]
    ax.plot(np.arange(len(demo)) * DT, np.degrees(demo - demo[0]),
            "k", lw=2, label="genuine rep")
    for name, bad in fault_variants(demo).items():
        ax.plot(np.arange(len(bad)) * DT, np.degrees(bad - bad[0]),
                lw=1.1, label=name)
    ax.set_xlabel("time (s)"); ax.set_ylabel("rotation angle (deg)")
    ax.set_title("controlled faults injected into a genuine repetition", fontsize=13)
    ax.legend(fontsize=11, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "exp5_quality.png"), dpi=500)


if __name__ == "__main__":
    main()
