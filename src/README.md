# TrustNav: how to run (Track 03, Path Finder)

Demo video (2:34): https://youtu.be/Md4wv-Sm9wE

**Entry point:** `src/main.py`. Python 3.8 or newer, standard library only, nothing to install. It runs on one thread, offline, in one streaming pass.

```bash
python src/main.py <degraded imu0/data.csv> estimated_trajectory.txt
```

**Input:** the degraded IMU CSV, in any layout the organisers' `degrade_imu.py` produces.
- Columns are `t, gyro, accel` (EuRoC) or `t, accel, gyro` (as the script documents). The order is detected automatically.
- The header row is optional.
- Timestamps can be in s, ms, µs or ns, and accel in m/s² or g. Both are auto-detected.

**Output:** TUM format `timestamp tx ty tz qx qy qz qw`, with the timestamp in seconds.
- There is **exactly one pose per input row**, and nothing is appended.
- The input timestamp is copied exactly (ns becomes `s.nnnnnnnnn`).
- A duplicated input row keeps its duplicated timestamp.
- Rows with invalid values get a dead-reckoned pose. That covers the zeroed or NaN 5 s blind window, stuck samples and out-of-range values.
- Dropouts (missing rows) are bridged internally.

**Budgets measured on EuRoC MH_01** (`python tools/measure.py <imu> out.txt`):
- Peak heap: 0.91 MB (tracemalloc)
- Runtime: 8.1 ms per 1 s of IMU, single thread

**Check the output:** `python tools/validate_output.py estimated_trajectory.txt <imu>` runs 6 rule checks and computes SHA-256 hashes.

Useful options (the defaults are the submitted configuration):
- `-v` prints a run summary.
- `--trace FILE` writes diagnostics for the dashboard.
- `--fill` writes extra poses inside gaps. It is off by default, because the rules want one pose per row.
- `--blind S` appends S extra seconds after the last row. The default is 0, because the rules put the blind window inside the input.
