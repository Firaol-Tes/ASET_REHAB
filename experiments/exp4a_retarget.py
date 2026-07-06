"""Experiment 4a: retargeting the learned primitives to a UR5e.

Retargeting model: the robot's tool reproduces the patient's hand path
under a single-pivot forearm model — a virtual elbow is anchored in the
robot workspace and the hand pose follows the wrist-orientation
primitive:  p(t) = anchor + R(t) v0,  R_tool(t) = R(t) R0,
with v0 a forearm-length vector in the initial pose. This is exactly
the guidance geometry of wrist-holding rehab robots.

Each 100 Hz pose trajectory is tracked with damped-least-squares IK;
we verify position/orientation tracking error, joint limits, and
velocity limits, and export joint trajectories for Gazebo playback.

Outputs:
  results/exp4a_feasibility.csv
  results/robot_traj/<exercise>.npz
  results/figures/exp4a_retarget.png
"""
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

from physiotwin import ur5e

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FS = 100.0
FOREARM = 0.25            # m, virtual forearm length


def retarget(demo_rotvec: np.ndarray):
    """Rotation-vector primitive -> (positions, rotations, anchor).

    The virtual elbow sits FOREARM above the robot's home tool pose, so
    the trajectory starts exactly at the current tool pose (hand
    hanging below the elbow, as at the start of a repetition).
    """
    R = Rotation.from_rotvec(demo_rotvec)
    R = R[0].inv() * R          # relative to the repetition's start pose
    T0 = ur5e.fk(ur5e.HOME)
    R0 = Rotation.from_matrix(T0[:3, :3])
    anchor = T0[:3, 3] + np.array([0.0, 0.0, FOREARM])
    v0 = T0[:3, 3] - anchor
    pos = anchor + R.apply(v0)
    rots = (R * R0).as_matrix()
    return pos, rots, anchor


def track(pos, rots):
    q = ur5e.HOME.copy()
    qs, pes, oes = [], [], []
    for p, Rm in zip(pos, rots):
        q, pe, oe = ur5e.ik_step(q, p, Rm)
        qs.append(q.copy()); pes.append(pe); oes.append(oe)
    return np.asarray(qs), np.asarray(pes), np.asarray(oes)


def main():
    out_dir = os.path.join(RESULTS, "robot_traj")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    example = None
    for f in sorted(glob.glob(os.path.join(RESULTS, "primitives", "*.npz"))):
        ex = os.path.splitext(os.path.basename(f))[0]
        demo = np.load(f)["demo"]
        pos, rots, anchor = retarget(demo)
        qs, pes, oes = track(pos, rots)
        qd = np.gradient(qs, 1 / FS, axis=0)
        vel_margin = float((np.abs(qd) / ur5e.QD_LIM).max())
        lim_ok = bool((qs > ur5e.Q_LIM[:, 0]).all()
                      and (qs < ur5e.Q_LIM[:, 1]).all())
        row = {
            "exercise": ex, "n_samples": len(qs),
            "pos_err_mm_max": round(float(pes.max() * 1000), 2),
            "ori_err_deg_max": round(float(np.degrees(oes.max())), 2),
            "joint_vel_frac_of_limit": round(vel_margin, 3),
            # DMP tau scaling needed so all joints respect speed limits
            "required_slowdown": round(max(1.0, vel_margin), 2),
            "within_joint_limits": lim_ok,
        }
        rows.append(row)
        print(f"{ex:32s} pos<= {row['pos_err_mm_max']:6.2f} mm  "
              f"ori<= {row['ori_err_deg_max']:5.2f} deg  "
              f"vel {row['joint_vel_frac_of_limit']:5.1%} of limit  "
              f"slowdown x{row['required_slowdown']:.2f}  "
              f"limits OK: {lim_ok}")
        np.savez(os.path.join(out_dir, f"{ex}.npz"),
                 q=qs, dt=1 / FS, tool_pos=pos)
        if ex == "Bicep-Curl":
            example = (qs, pos, pes, oes, anchor)

    with open(os.path.join(RESULTS, "exp4a_feasibility.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    make_figure(example)


def make_figure(example):
    qs, pos, pes, oes, anchor = example
    t = np.arange(len(qs)) / FS
    fig = plt.figure(figsize=(13, 4))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=2)
    ax.scatter(*anchor, color="k", s=40, label="virtual elbow")
    ax.set_title("Bicep-Curl: tool path (m)", fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(1, 3, 2)
    for i in range(6):
        ax.plot(t, np.degrees(qs[:, i]), label=f"q{i + 1}", lw=1.2)
    ax.set_title("joint trajectories", fontsize=10)
    ax.set_xlabel("time (s)"); ax.set_ylabel("joint angle (deg)")
    ax.legend(fontsize=7, ncol=3, frameon=False)

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(t, pes * 1000, label="position error (mm)")
    ax.plot(t, np.degrees(oes), label="orientation error (deg)")
    ax.set_title("IK tracking error", fontsize=10)
    ax.set_xlabel("time (s)")
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "figures", "exp4a_retarget.png"),
                dpi=160)


if __name__ == "__main__":
    main()
