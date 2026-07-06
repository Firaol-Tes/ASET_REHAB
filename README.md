# PhysioTwin

Teaching a rehabilitation robot upper-limb exercises from a single consumer
smartwatch — and verifying patient performance with the same device.

A patient's (or therapist's) exercise, recorded by an Apple Watch, is
mirrored, encoded as a motion primitive, and executed by a compliant robot
arm that guides the affected limb. Research paper in progress (target:
ASET 2026, Mechatronics/Industry 4.0/IoT track).

## Dataset

Dual-wrist Apple Watch recordings (100 Hz device-motion + raw accelerometer)
of 11 subjects performing 12 prescribed upper-limb rehabilitation exercises,
most captured **simultaneously on both wrists**. The dataset is not included
in this repository (privacy); set `PHYSIOTWIN_DATA` to its location.

## Layout

- `src/physiotwin/` — data access (anonymized), motion/quaternion analysis
- `experiments/` — one script per paper experiment
- `results/` — generated tables and figures (anonymized subject IDs only)

## Experiments

1. `exp1_mirror.py` — bilateral mirror-symmetry validation: is the healthy
   arm's motion a faithful template for the other arm? (frame-invariant
   rotation profiles of synchronized left/right recordings)
2. Exercise recognition baseline (LOSO + cross-wrist) — planned
3. Motion-primitive (DMP) learning — planned
4. ROS 2 + Gazebo + MoveIt 2 execution on UR5e — planned

## Requirements

Python 3.12, NumPy, SciPy, Matplotlib.
