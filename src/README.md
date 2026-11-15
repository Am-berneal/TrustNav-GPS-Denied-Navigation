# TrustNav-IMU — Track 3 "Path Finder"

6-DoF trajectory from a degrading IMU: progressive noise, bias drift, spikes, 2 s dropouts,
then complete failure with 5 s of blind dead-reckoning. Offline, single thread, one pass.

## Submission (Samsung PRISM XR Hackathon 2026, Track 03)
| Item | Where |
|---|---|
| Final output | `estimated_trajectory.txt` (TUM, 36,819 poses, 6/6 checks in `validation.json`) |
| Source code | `src/main.cpp` (submitted solver, C++17) and `src/main.py` (reference) |
| Presentation | `presentation/Thapar_University_Cipher_Guards.pptx` and `.pdf` |
| Demo video | DEMO_VIDEO_LINK |
| Live console | `dashboard/TrustNav-Mission-Console.html` (GitHub Pages: https://am-berneal.github.io/TrustNav-GPS-Denied-Navigation/dashboard/TrustNav-Mission-Console.html) |
| Package / Docker | `Dockerfile`, `metadata.json`, `tools/` |

Team Cipher Guards, Thapar Institute of Engineering & Technology: Amberneal Singh.

## Get the data
1. Download MH_01_easy.bag from Kaggle: https://www.kaggle.com/datasets/chunai/euroc-mh-01-easy-ros-bag-dataset
2. pip install rosbags && python tools/bag_to_euroc.py MH_01_easy.bag MH_01_easy
3. Ground truth: euroc_mav/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv from
   https://www.kaggle.com/datasets/drmainak/uzh-fpv-and-euroc -> MH_01_easy/mav0/state_groundtruth_estimate0/
4. Degrade + run + score:  bash tools/run_real.sh MH_01_easy

## Run
```
g++ -O2 -std=c++17 -o trustnav src/main.cpp
./trustnav mav0/imu0/data.csv estimated_trajectory.txt          # submitted solver
python3 src/main.py mav0/imu0/data.csv estimated_trajectory.txt # reference, same output
```
Docker: `docker build -t trustnav . && docker run --rm -v $PWD/mav0:/data trustnav /data/imu0/data.csv /data/estimated_trajectory.txt`

Output: TUM format `timestamp tx ty tz qx qy qz qw`, seconds, one pose per IMU frame, missing
frames filled at 200 Hz, plus 1000 poses (5 s) after the last IMU row.

Options (C++ and Python):
- `--no-fill` write poses only at the input IMU timestamps (no gap filling)
- `--blind S` length of the post-failure dead-reckoning window (default 5; `--blind 0` disables it)

`--no-fill --blind 0` gives exactly one output row per input IMU row, in case the evaluator
matches rows rather than timestamps.

## Method
1. Streaming reader; NaN / saturated / all-zero rows treated as missing.
2. Start-up calibration: quietest 1 s window in the first 10 s gives gravity direction, local g,
   gyro bias and noise floor; attitude is integrated backwards to frame 0.
3. Per-channel trust (noise floor / current noise) and spike gate (8 sigma + physical floor).
4. Quaternion strapdown with trust-weighted complementary tilt correction and gyro-bias integral
   (Mahony et al. 2008); fast re-level for 3 s after a dropout.
5. Motion prior fitted on training flights: velocity mean reversion (tau 1 s, vertical 0.5 s),
   weak workspace spring (3 s), speed bound 3 m/s, zero-velocity detector.
6. Rotor-drag velocity aiding: specific force perpendicular to thrust = -k v, k = 0.17 1/s
   (Leishman et al. 2014; measured consistently across the training flights).
7. Gaps and the blind window: coast with decaying velocity (0.5 s) and yaw rate (3 s), tilt held level.

8. Sensor-health monitor (tools/health.py): per-channel noise vs. the clean start, spike and dropout detection, event log, per-pose confidence (IMU / DROPOUT / BLIND) and a health timeline. Stress-tested on 6 jamming scenarios (tools/stress_test.py).
9. Physical guardrail: the position may never move faster than the speed bound (vmax*dt per step); found by tools/validate_output.py, improves ATE 4.25 -> 4.20 m.

## Results on real EuRoC data (MH_01_easy, real IMU)
Real ADIS16448 IMU from the EuRoC MH_01_easy bag, degraded with `tools/degrade.py` (seed 1: noise up
to x6, bias drift, 46 spikes, two 2 s dropouts, 5 s failure), scored against the real ground truth.

| metric | TrustNav-IMU | zero-information reference* |
|---|---|---|
| ATE RMSE | 4.20 m | 4.31 m |
| checkpoints (mean of 6) | 2.28 m | - |
| blind window (mean) | 5.84 m | - |
| dropout windows (mean) | 3.53 m | - |
| RPE (1 s, translation) | 0.88 m | - |
| tilt error | 3.5-12 deg | - |
| peak RSS (C++) | 3.5 MB | limit 8 MB |
| runtime (C++, incl. I/O) | 0.75 ms per IMU second | limit 20 ms |

*a constant position at the centre of the ground-truth path (best possible "do nothing" answer).
Ablations on the same run (measured before step 9): no workspace spring 25.6 m ATE, no motion prior 72.9 m, no rotor-drag
aiding 4.32 m ATE / 4.71 m in dropouts. MH_01 starts with a hand-held excitation (no quiet second),
so start-up gyro-bias calibration is skipped; the bias integral takes over in flight.
Honest reading: on real data the attitude and the efficiency are strong; room-scale position is
held to the level of a well-placed constant, not better.

## Results on regenerated IMU (development data, 7 EuRoC flights)
| | ATE | checkpoints | blind mean | target +5 s | dropouts |
|---|---|---|---|---|---|
| raw integration | 33.9 km | 30.0 km | 70.5 km | 74.7 km | 16.6 km |
| TrustNav-IMU | 3.20 m | 2.84 m | 2.18 m | 2.36 m | 3.38 m |
| hold-position reference | 3.29 m | 2.90 m | 3.00 m | 3.19 m | 3.62 m |

Leave-one-flight-out ATE 3.24 m. Tilt error 3.4 deg RMS. C++: 4.1 MB peak RSS, 0.37 ms per IMU second.
Python: 0.75 MB of allocations but ~16 MB process RSS (the interpreter alone is ~10.7 MB), so submit main.cpp.
`tools/evaluate.py` matches `evo_ape tum gt est -a` exactly.

## Stress tests (same parameters, no re-tuning)
| scenario | ATE | target +5 s | weighted |
|---|---|---|---|
| baseline noise x6 | 3.20 m | 2.36 m | 2.71 m |
| noise x3 | 3.20 m | 2.59 m | 2.77 m |
| noise x10 | 3.20 m | 2.54 m | 2.74 m |
| dropouts zeroed | 3.20 m | 2.55 m | 2.76 m |
| new jamming seed | 3.20 m | 2.39 m | 2.75 m |

Blind-window sweep: constant-velocity coasting gives 4.94 m at the target vs 2.36 m with the fitted
decay (tau_b 0.5 s). Removing the workspace spring raises ATE from 3.20 to 6.03 m. See `tools/extra.py`.

## Data honesty
The organiser IMU package was not available to us. Flight paths and attitudes are the real EuRoC
ground truth (from public evo / rpg_trajectory_evaluation repos); IMU readings were regenerated from
them with EuRoC's ADIS16448 noise model (`tools/make_imu.py`) and degraded by `tools/degrade.py`,
our replica of the brief. Re-run `tools/fit_params.py` and `tools/results.py` on the official data.
On room-scale position the result is only modestly better than standing still; the clear gains are
the blind window, attitude and short-term motion.

## Real-data validation
```
tools/run_real.sh /path/to/MH_01_easy        # degrade -> solve -> score; results in real_runs/
```
Needs `<seq>/mav0/imu0/data.csv` and `<seq>/mav0/state_groundtruth_estimate0/data.csv`
(official ASL zip from the ETH Research Collection). From the Kaggle ROS bag instead:
`pip install rosbags && python tools/bag_to_euroc.py MH_01_easy.bag MH_01_easy`, then add the
ground-truth CSV from the Kaggle dataset `drmainak/uzh-fpv-and-euroc`.
`tools/evaluate.py` accepts the EuRoC ground-truth CSV directly.

## Dashboard
`dashboard/TrustNav-Console.html` — open in a browser; 18 pages driven by the measured results.

## Stress test (real MH_01, tools/stress_test.py, C++ solver)
| scenario | ATE m | checkpoints m | blind m | dropouts m | RPE 1 s m |
|---|---|---|---|---|---|
| baseline noise x6, seed 1 | 4.20 | 2.28 | 5.84 | 3.53 | 0.88 |
| baseline noise x6, seed 2 | 4.89 | 3.38 | 2.87 | 4.83 | 0.97 |
| baseline noise x6, seed 3 | 4.61 | 2.98 | 1.68 | 4.30 | 0.95 |
| light jamming x3 | 4.20 | 2.27 | 5.84 | 3.53 | 0.88 |
| heavy jamming x10 | 4.21 | 2.28 | 5.82 | 3.55 | 0.88 |
| dropouts zeroed (not removed) | 4.20 | 2.28 | 5.84 | 3.53 | 0.88 |
| average of 3 seeds | 4.57 | 2.88 | 3.46 | 4.22 | 0.93 |
Accuracy is unchanged from x3 to x10 jamming; the variation between seeds comes from bias drift.
Health monitor on seed 1: 44 spikes and both 2 s dropouts detected.