# PhysioTwin

Teaching a rehabilitation robot upper-limb exercises from a single consumer
smartwatch — and verifying patient performance with the same device.

A patient's (or therapist's) exercise, recorded by an Apple Watch, is
mirrored, encoded as a motion primitive, and executed by a compliant robot
arm that guides the affected limb.

Accompanying paper: *PhysioTwin: Teaching, Verifying, and Grading Robotic
Rehabilitation Exercises with a Single Consumer Smartwatch*, accepted at
ASET 2026 (Mechatronics/Industry 4.0/IoT track). Source in `paper/`.

**Scope.** All robot results are simulation-based — no physical robot was
operated. All subjects are healthy volunteers and the movement faults are
synthetic injections. This is a validated pipeline, not a clinical result.

## Dataset

Dual-wrist Apple Watch recordings (100 Hz device-motion + raw accelerometer)
of 11 subjects performing 12 prescribed upper-limb rehabilitation exercises,
most captured **simultaneously on both wrists** (107 of 109 left/right pairs
overlap in absolute time). 240 analysed sessions, ~3.0 hours of motion.
The dataset is not included in this repository (privacy); set
`PHYSIOTWIN_DATA` to its location.

Subjects are anonymized to S01–S11 by folder name at load time; real names
never enter `results/`. Note that a handful of recordings carry a stale
`user` field in their filename, from a watch that was not renamed between
participants — anonymize by folder, as `data.py` does, never by that field.

## Layout

- `src/physiotwin/` — data access (anonymized), motion/quaternion analysis
- `experiments/` — one script per paper experiment
- `results/` — generated tables and figures (anonymized subject IDs only)

## Experiments

1. `exp1_mirror.py` — bilateral mirror-symmetry validation: is the healthy
   arm's motion a faithful template for the other arm? (frame-invariant
   rotation profiles of synchronized left/right recordings)
2. `exp2_recognition.py` — 12-class exercise recognition, strict LOSO, plus
   the cross-wrist transfer gap
3. `exp2b_mirror_training.py` — data-certified mirror map and mirror-aware
   training that closes that gap; includes the per-fold leakage check
4. `exp2c_equivalence.py` — paired difference and TOST equivalence tests of
   mirror-augmented cross-wrist accuracy against the within-wrist reference
5. `exp3_primitives.py` — repetition segmentation and DMP learning
6. `exp4a_retarget.py` — kinematic retargeting of primitives to a UR5e
7. `exp4b_execute.py` — execution in Gazebo via ROS 2
   `joint_trajectory_controller` (needs a sourced ROS 2 environment and a
   running sim)
8. `exp4c_virtual_watch.py` — closed-loop verification: recognizing the
   robot's executions through a virtual watch
9. `exp5_quality.py`, `exp5b_robot_faults.py` — movement-quality grading
   against population and personalized envelopes, offline and on the robot

Experiments 1–3, 5 and 9 run from the dataset alone. Experiments 7 and 9's
robot half require ROS 2 + Gazebo; their recorded joint states are committed
under `results/gz_exec/` so the tables can be regenerated without a sim.

## Requirements

Python 3.12, NumPy, SciPy, scikit-learn, Matplotlib.
ROS 2 (Jazzy) + Gazebo + the UR5e description packages for `exp4b`/`exp5b`.
