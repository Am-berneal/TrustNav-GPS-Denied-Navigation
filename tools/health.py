"""TrustNav sensor-health monitor: jamming timeline, event log and per-pose confidence.

Reads the degraded IMU file (and optionally the solver output) and writes, into <out_dir>:
  events_log.txt  timestamped events: noise rising, spikes, dropouts, IMU failure
  health.csv      per 0.5 s: gyro/accel noise ratio vs. clean start, spikes, health 0-100
  confidence.txt  per pose of the trajectory: timestamp, confidence 0-1, state
  health.png      timeline figure for the dashboard / demo video (needs matplotlib)

Usage:
  python tools/health.py <imu0/data.csv> <out_dir> [estimated_trajectory.txt]

Standalone: does not change the solver or estimated_trajectory.txt.
"""
import os
import sys
import numpy as np

WIN = 0.5            # health window [s]
SPIKE_K = 8.0        # spike gate, in noise sigmas (same as the solver)
FLOOR = (0.6, 6.0)   # physical floor for a spike: gyro rad/s, accel m/s^2 (same as the solver)
NOISE_LEVELS = (2, 4, 6, 8)


def load_imu(path):
    d = np.loadtxt(path, delimiter=',', comments='#')
    t = d[:, 0].astype(np.int64) * 1e-9 if d[0, 0] > 1e12 else d[:, 0]
    return t, d[:, 1:7]


def highpass(x, n):
    """Remove motion: subtract a centred moving median of n samples (robust to spikes)."""
    from numpy.lib.stride_tricks import sliding_window_view
    n |= 1
    pad = np.pad(x, ((n // 2, n // 2), (0, 0)), mode='edge')
    med = np.median(sliding_window_view(pad, n, axis=0), axis=-1)
    return x - med


def robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x, 0)), 0)


def noise_sigma(x):
    """Noise level robust to spikes AND to quantised / repeated samples (where the MAD is 0):
    RMS after clipping at the 99th percentile of |x|."""
    q = np.quantile(np.abs(x), 0.99, axis=0)
    return np.sqrt(np.mean(np.clip(x, -q, q) ** 2, axis=0))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    imu, out = sys.argv[1], sys.argv[2]
    traj = sys.argv[3] if len(sys.argv) > 3 else None
    os.makedirs(out, exist_ok=True)
    t, x = load_imu(imu)
    t0 = t[0]
    dt = float(np.median(np.diff(t)))
    hp = highpass(x, max(3, int(round(0.1 / dt))))
    n_base = max(200, int(0.1 * len(t)))          # first 10% = clean reference
    base = np.maximum(noise_sigma(hp[:n_base]), 1e-6)

    events = []
    # dropouts
    gaps = np.nonzero(np.diff(t) > 1.6 * dt)[0]
    gap_list = []
    for i in gaps:
        a, b = t[i] - t0, t[i + 1] - t0
        gap_list.append((a, b))
        events.append((a, 'DROPOUT_START', 'no IMU data for %.2f s - coasting on motion model' % (b - a)))
        events.append((b, 'DROPOUT_END', 'IMU back - fast re-levelling for 3 s'))
    # windows
    edges = np.arange(0.0, t[-1] - t0 + WIN, WIN)
    idx = np.searchsorted(t - t0, edges)
    rows, crossed, recent = [], set(), []
    n_spikes_total = 0
    for k in range(len(edges) - 1):
        s, e = idx[k], idx[k + 1]
        if e - s < 5:
            rows.append((edges[k], np.nan, np.nan, 0, 0.0))
            continue
        sig = noise_sigma(hp[s:e]) / base
        rg, ra = float(sig[:3].mean()), float(sig[3:].mean())
        lim = SPIKE_K * robust_sigma(hp[s:e]) * 1.0
        floor = np.array([FLOOR[0]] * 3 + [FLOOR[1]] * 3)
        sp = np.any((np.abs(hp[s:e]) > lim) & (np.abs(hp[s:e]) > floor), 1)
        ns = int(sp.sum())
        n_spikes_total += ns
        for j in np.nonzero(sp)[0]:
            ch = int(np.argmax(np.abs(hp[s + j]) / np.maximum(lim, floor)))
            events.append((t[s + j] - t0, 'SPIKE', 'rejected outlier on %s' % ['gx', 'gy', 'gz', 'ax', 'ay', 'az'][ch]))
        r = max(rg, ra)
        recent.append(r)
        r_s = float(np.median(recent[-6:]))  # sustained for ~3 s, not a single noisy window
        for lv in NOISE_LEVELS:
            if r_s >= lv and lv not in crossed:
                crossed.add(lv)
                events.append((edges[k], 'NOISE_RISING', 'noise now x%d of clean level, sustained (gyro x%.1f, accel x%.1f) - trust lowered' % (lv, rg, ra)))
        health = 100.0 / max(1.0, r) * (0.5 if ns else 1.0)
        rows.append((edges[k], rg, ra, ns, health))
    events.append((t[-1] - t0, 'IMU_FAILURE', 'last IMU sample - dead reckoning the 5 s blind window'))
    events.sort(key=lambda e: e[0])

    with open(os.path.join(out, 'events_log.txt'), 'w') as fh:
        fh.write('# t_rel[s]  t_unix[s]  event  detail\n')
        for a, kind, msg in events:
            fh.write('%8.2f  %.3f  %-13s %s\n' % (a, a + t0, kind, msg))
    with open(os.path.join(out, 'health.csv'), 'w') as fh:
        fh.write('t_rel_s,gyro_noise_ratio,accel_noise_ratio,spikes,health_0_100\n')
        for r in rows:
            fh.write('%.2f,%.3f,%.3f,%d,%.1f\n' % r)

    # per-pose confidence: health while data flows, exponential decay while coasting
    if traj:
        tr = np.loadtxt(traj, comments='#')[:, 0]
        hw = np.array([r[4] for r in rows]) / 100.0
        last_t = t[-1]
        with open(os.path.join(out, 'confidence.txt'), 'w') as fh:
            fh.write('# timestamp confidence state\n')
            for ts in tr:
                rel = ts - t0
                gap = next(((a, b) for a, b in gap_list if a < rel < b), None)
                if ts > last_t:
                    c, st = 0.6 * np.exp(-(ts - last_t) / 3.0), 'BLIND'
                elif gap:
                    c, st = 0.6 * np.exp(-(rel - gap[0]) / 3.0), 'DROPOUT'
                else:
                    k = min(int(rel / WIN), len(hw) - 1)
                    c, st = (hw[k] if hw[k] == hw[k] else 0.0), 'IMU'
                fh.write('%.9f %.3f %s\n' % (ts, c, st))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        R = np.array([r for r in rows], float)
        fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        ax[0].plot(R[:, 0], R[:, 1], label='gyro noise ratio', lw=1)
        ax[0].plot(R[:, 0], R[:, 2], label='accel noise ratio', lw=1)
        ax[0].set_ylabel('x clean level'); ax[0].legend(loc='upper left'); ax[0].grid(alpha=.3)
        ax[1].fill_between(R[:, 0], R[:, 4], color='tab:green', alpha=.5, step='post')
        ax[1].set_ylabel('IMU health'); ax[1].set_ylim(0, 105); ax[1].set_xlabel('time [s]'); ax[1].grid(alpha=.3)
        for a, b in gap_list:
            for x_ in ax: x_.axvspan(a, b, color='tab:red', alpha=.25)
        for x_ in ax: x_.axvspan(t[-1] - t0, t[-1] - t0 + 5, color='k', alpha=.15)
        ax[0].set_title('TrustNav sensor health - red: dropouts, grey: IMU failure / blind 5 s')
        fig.tight_layout(); fig.savefig(os.path.join(out, 'health.png'), dpi=130)
    except ImportError:
        pass
    print('events: %d (spikes %d, dropouts %d) -> %s' % (len(events), n_spikes_total, len(gap_list), out))


if __name__ == '__main__':
    main()
