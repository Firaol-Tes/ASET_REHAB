"""Experiment 5b: the virtual watch grades faulty robot executions.

Three exercises are executed on the simulated UR5e five ways each:
clean, too fast (played at human tempo instead of therapy pace),
reduced range (orientation primitive scaled to 50 %), tremor (4 Hz,
3 deg oscillation added to the primitive), and truncated (stopped at
60 %). The virtual watch reconstructs the wrist rotation-angle profile
from the executed motion, segments repetitions, and grades each one
against the population envelope of Experiment 5.

Expected: clean executions pass; faulty ones are flagged, with the
flagged feature naming the fault.

Run inside a sourced ROS 2 environment while the Gazebo sim is up
(GZ_PARTITION/ROS_DOMAIN_ID of the experiment instance):
  python3 experiments/exp5b_robot_faults.py

Outputs: results/exp5b_robot_grading.csv
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from scipy.spatial.transform import Rotation

from physiotwin import ur5e
from physiotwin.data import index_sessions
from physiotwin.quality import Envelope, rep_features
from physiotwin.reps import segment_reps, session_rotvec

from exp4a_retarget import retarget, track
from exp4b_execute import Executor, DOWNSAMPLE, LOOPS, BLEND

import rclpy

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FS = 100.0
DT = 1.0 / FS

EXERCISES = {"Bicep-Curl": 2.0, "Lateral-Delt-Raise": 2.0,
             "Wand-Extension": 2.0}   # therapy-pace slowdown


def primitive_rel_rotvec(ex):
    """Primitive re-referenced to its own start (identity at t=0)."""
    demo = np.load(os.path.join(RESULTS, "primitives", f"{ex}.npz"))["demo"]
    R = Rotation.from_rotvec(demo)
    return (R[0].inv() * R).as_rotvec()


def variants(rv: np.ndarray):
    """Faulty versions of a relative rotation-vector primitive.
    Returned speed factor multiplies the therapy-pace slowdown."""
    t = np.arange(len(rv)) * DT
    # tremor along the primitive's principal motion direction
    u = np.linalg.svd(rv - rv.mean(0), full_matrices=False)[2][0]
    return {
        "clean": (rv, 1.0),
        "too_fast": (rv, 0.5),
        "reduced_rom": (rv * 0.5, 1.0),
        "tremor": (rv + np.radians(3.0)
                   * np.sin(2 * np.pi * 4.0 * t)[:, None] * u, 1.0),
        "truncated": (rv[: int(0.6 * len(rv))], 1.0),
    }


def executed_theta(rec, slowdown):
    """Speed-compensated wrist rotation-angle profile of an execution."""
    t = (rec[:, 0] - rec[0, 0]) / slowdown
    grid = np.arange(t[0], t[-1], DT)
    q = np.column_stack([np.interp(grid, t, rec[:, 1 + j])
                         for j in range(6)])
    T0 = ur5e.fk(ur5e.HOME)
    R0 = Rotation.from_matrix(T0[:3, :3])
    R_tool = Rotation.from_matrix(
        np.stack([ur5e.fk(qi)[:3, :3] for qi in q]))
    return (R_tool * R0.inv()).magnitude()


def trim_idle(theta, fs=FS, v_thresh_dps=3.0, margin_s=0.1):
    """Trim motionless head/tail of a repetition segment. The robot's
    guarded protocol pauses between repetitions (return blend); human
    recordings cycle continuously — durations are only comparable on
    the active part."""
    v = np.abs(np.gradient(theta, 1 / fs))
    active = np.where(v > np.radians(v_thresh_dps))[0]
    if len(active) == 0:
        return theta
    m = int(margin_s * fs)
    return theta[max(0, active[0] - m): active[-1] + m]


def population_envelopes():
    table = defaultdict(list)
    for s in index_sessions():
        if s.exercise not in EXERCISES:
            continue
        _rv, theta, _ref = session_rotvec(s)
        for a, b in segment_reps(theta):
            table[s.exercise].append(rep_features(trim_idle(theta[a:b])))
    return {ex: Envelope(rows) for ex, rows in table.items()}


def build_command(qs, base_slow, speed_factor):
    one = qs[::DOWNSAMPLE]
    step = DT * base_slow * speed_factor * DOWNSAMPLE
    cmd, times, t_now = [], [], 0.0
    for loop in range(LOOPS):
        if loop > 0:
            t_now += BLEND
        for wp in one:
            t_now += step
            cmd.append(wp)
            times.append(t_now)
    return np.asarray(cmd), np.asarray(times)


def main():
    envs = population_envelopes()
    rclpy.init()
    node = Executor()
    rows = []
    for ex, base_slow in EXERCISES.items():
        rv = primitive_rel_rotvec(ex)
        for name, (rv_bad, speed) in variants(rv).items():
            pos, rots, _anchor = retarget(rv_bad)
            qs, _pes, _oes = track(pos, rots)
            cmd, times = build_command(qs, base_slow, speed)
            node.goto(qs[0], duration=8.0)
            rec = node.play(cmd, times)
            np.savez(os.path.join(RESULTS, "gz_exec",
                                  f"fault_{ex}_{name}.npz"), recorded=rec)
            theta = executed_theta(rec, base_slow * speed)
            reps = segment_reps(theta)
            flagged, feats_hit = 0, set()
            for a, b in reps:
                fl = envs[ex].flag(rep_features(trim_idle(theta[a:b])))
                if fl:
                    flagged += 1
                    feats_hit.update(fl)
            rows.append({
                "exercise": ex, "variant": name, "n_reps": len(reps),
                "flagged": flagged,
                "features": "+".join(sorted(feats_hit)) or "-",
            })
            print(f"{ex:20s} {name:12s} reps {len(reps)}  "
                  f"flagged {flagged}  [{rows[-1]['features']}]")

    with open(os.path.join(RESULTS, "exp5b_robot_grading.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
