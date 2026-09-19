"""Experiment 4b: execute learned exercises on the UR5e in Gazebo.

Sends each retargeted joint trajectory (Experiment 4a) to the
joint_trajectory_controller of a physics-simulated UR5e, records
/joint_states during execution, and reports joint-space tracking error.
Trajectories that exceeded UR5e speed limits at human tempo are played
at their required DMP slowdown (Exp 4a feasibility table).

Run inside a sourced ROS 2 environment while the Gazebo sim is up:
  python3 experiments/exp4b_execute.py [exercise ...]

Outputs:
  results/exp4b_tracking.csv
  results/gz_exec/<exercise>.npz     commanded + achieved joint traj
"""
import csv
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
CTRL = "/joint_trajectory_controller/follow_joint_trajectory"
DOWNSAMPLE = 5           # command waypoints at 20 Hz; controller interpolates
LOOPS = 5                # repetitions per exercise set
BLEND = 2.0              # s, slow return-to-start between repetitions


def load_slowdowns():
    out = {}
    with open(os.path.join(RESULTS, "exp4a_feasibility.csv")) as f:
        for row in csv.DictReader(f):
            out[row["exercise"]] = float(row["required_slowdown"])
    return out


class Executor(Node):
    def __init__(self):
        super().__init__("physiotwin_executor")
        self.client = ActionClient(self, FollowJointTrajectory, CTRL)
        self.recording = False
        self.samples = []
        self.create_subscription(JointState, "/joint_states", self._cb, 50)

    def _cb(self, msg):
        if self.recording and all(j in msg.name for j in JOINTS):
            idx = [msg.name.index(j) for j in JOINTS]
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.samples.append([t] + [msg.position[i] for i in idx])

    def goto(self, q, duration=4.0):
        """Move to configuration q smoothly (setup move)."""
        self._send(np.asarray([q]), np.asarray([duration]))

    def play(self, qs, times):
        self.samples = []
        self.recording = True
        self._send(qs, times)
        self.recording = False
        return np.asarray(self.samples)

    def _send(self, qs, times):
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = JOINTS
        for q, tt in zip(qs, times):
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in q]
            p.time_from_start.sec = int(tt)
            p.time_from_start.nanosec = int((tt % 1) * 1e9)
            traj.points.append(p)
        goal.trajectory = traj
        for j in JOINTS:
            tol = JointTolerance()
            tol.name = j
            tol.position = float(os.environ.get("PT_PATH_TOL", 0.5))
            goal.path_tolerance.append(tol)
        self.client.wait_for_server()
        fut = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        handle = fut.result()
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res)
        return res.result()


def main():
    wanted = sys.argv[1:] or None
    slow = load_slowdowns()
    out_dir = os.path.join(RESULTS, "gz_exec")
    os.makedirs(out_dir, exist_ok=True)

    rclpy.init()
    node = Executor()
    rows = []
    traj_dir = os.path.join(RESULTS, "robot_traj")
    for fn in sorted(os.listdir(traj_dir)):
        ex = os.path.splitext(fn)[0]
        if wanted and ex not in wanted:
            continue
        d = np.load(os.path.join(traj_dir, fn))
        qs, dt = d["q"], float(d["dt"])
        # therapy pace: at least half human speed, slower if UR5e
        # velocity limits demand it (Exp 4a feasibility)
        factor = max(2.0, slow.get(ex, 1.0))
        # a therapy set: repeat the learned repetition, as the human did,
        # with a slow return-to-start blend between repetitions
        one = qs[::DOWNSAMPLE]
        step = dt * factor * DOWNSAMPLE
        cmd_list, t_list, t_now = [], [], 0.0
        for loop in range(LOOPS):
            if loop > 0:
                t_now += BLEND
            for k, wp in enumerate(one):
                t_now += step
                cmd_list.append(wp)
                t_list.append(t_now)
        cmd = np.asarray(cmd_list)
        t_cmd = np.asarray(t_list)
        node.goto(qs[0], duration=8.0)
        rec = node.play(cmd, t_cmd)
        t_rec = rec[:, 0] - rec[0, 0]
        err = []
        for j in range(6):
            achieved = np.interp(t_cmd, t_rec, rec[:, 1 + j])
            err.append(np.abs(achieved - cmd[:, j]))
        err = np.asarray(err)
        rows.append({
            "exercise": ex, "slowdown": factor,
            "duration_s": round(float(t_cmd[-1]), 1),
            "mean_err_deg": round(float(np.degrees(err.mean())), 3),
            "max_err_deg": round(float(np.degrees(err.max())), 3),
            "n_recorded": len(rec),
        })
        print(f"{ex:32s} x{factor:.2f}  {rows[-1]['duration_s']:5.1f}s  "
              f"mean {rows[-1]['mean_err_deg']:6.3f} deg  "
              f"max {rows[-1]['max_err_deg']:6.3f} deg")
        np.savez(os.path.join(out_dir, fn), cmd=cmd, t_cmd=t_cmd,
                 recorded=rec)

    # Merge into the existing table rather than truncating it: a filtered
    # re-run (e.g. one exercise, to recapture figure frames) must not drop
    # the other exercises' rows from the published results table.
    out_csv = os.path.join(RESULTS, "exp4b_tracking.csv")
    merged = {}
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                merged[row["exercise"]] = row
    for row in rows:
        merged[row["exercise"]] = {k: str(v) for k, v in row.items()}
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(merged[k] for k in sorted(merged))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
