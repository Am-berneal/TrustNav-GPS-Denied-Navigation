"""Convert a phone IMU recording into an EuRoC imu0/data.csv that TrustNav (and the organisers'
degrade_imu.py) can read.

Supported (auto-detected from file and column names, folder or .zip):
  * Sensor Logger (Android / iOS, free): Gyroscope.csv + TotalAcceleration.csv
    (or Accelerometer.csv + Gravity.csv), columns time [ns epoch], seconds_elapsed, z, y, x
  * phyphox: "Gyroscope rotation rate" + "Acceleration with g" exports,
    columns "Time (s)", "... x (...)", "... y (...)", "... z (...)"
  * any pair of CSVs whose names contain "gyro" and "acc", with a time column and x / y / z columns

Units: gyro rad/s (deg/s is detected and converted); accel m/s^2 or g (converted). Accelerometer data
without gravity is completed with a gravity file when one is present.
The accelerometer is interpolated onto the gyroscope timestamps.

Usage:  python tools/phone_to_euroc.py <export folder or .zip> <out_dir>
        -> <out_dir>/mav0/imu0/data.csv   (EuRoC: t[ns], gyro xyz [rad/s], accel xyz [m/s^2])
"""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

G = 9.80665


def list_csvs(src):
    """name -> DataFrame for every CSV in a folder or zip."""
    out = {}
    if os.path.isdir(src):
        for root, _, files in os.walk(src):
            for f in files:
                if f.lower().endswith('.csv'):
                    out[os.path.relpath(os.path.join(root, f), src)] = pd.read_csv(os.path.join(root, f))
    else:
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                if n.lower().endswith('.csv') and not n.startswith('__MACOSX'):
                    out[n] = pd.read_csv(io.BytesIO(z.read(n)))
    return out


def find(csvs, *keys, exclude=()):
    for n, df in csvs.items():
        b = os.path.basename(n).lower().replace(' ', '')
        if all(k in b for k in keys) and not any(e in b for e in exclude):
            return n, df
    return None, None


def xyz_time(df):
    """-> t [s, float64 epoch or elapsed], N x 3 (x, y, z), using column names."""
    cols = {c: c.strip().lower() for c in df.columns}
    tcol = None
    for pref in ('time', 'timestamp', 'time (s)', 'seconds_elapsed'):
        for c, l in cols.items():
            if l == pref:
                tcol = c; break
        if tcol: break
    if tcol is None:
        tcol = next(c for c, l in cols.items() if 'time' in l)
    axis = {}
    for a in 'xyz':
        for c, l in cols.items():
            if l == a or l.startswith(a + ' ') or l.startswith(a + '(') or (' %s ' % a) in (' ' + l + ' ') \
                    or l.endswith('_' + a) or l.endswith(' ' + a):
                axis[a] = c; break
    if len(axis) < 3:
        raise ValueError('could not find x/y/z columns in %s' % list(df.columns))
    t = df[tcol].to_numpy(dtype=np.float64)
    dt = np.median(np.diff(t))
    scale = 1.0 if dt < 1.0 else 1e-3 if dt < 1e3 else 1e-6 if dt < 1e6 else 1e-9
    return t * scale, df[[axis['x'], axis['y'], axis['z']]].to_numpy(dtype=np.float64), tcol


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    csvs = list_csvs(src)
    gn, gdf = find(csvs, 'gyro', exclude=('uncalib',))
    an, adf = find(csvs, 'totalacc')
    if an is None: an, adf = find(csvs, 'accelerationwithg')
    if an is None: an, adf = find(csvs, 'acc', exclude=('uncalib', 'linear', 'withoutg'))
    if gn is None or an is None:
        sys.exit('need a gyroscope and an accelerometer CSV; found: %s' % list(csvs))
    tg, w, _ = xyz_time(gdf)
    ta, a, _ = xyz_time(adf)
    if np.median(np.abs(w)) > 0.5 and np.percentile(np.abs(w), 99) > 20:
        w = np.radians(w)  # deg/s
    norm = np.median(np.linalg.norm(a, axis=1))
    if 0.5 < norm < 2.0:
        a = a * G; norm *= G  # g -> m/s^2
    note = ''
    if norm < 3.0:  # gravity removed: add the gravity vector back if it was logged
        nn, gvdf = find(csvs, 'gravity')
        if gvdf is None:
            sys.exit('accelerometer has no gravity and no Gravity.csv was found: log TotalAcceleration / "Acceleration with g"')
        tv, gv, _ = xyz_time(gvdf)
        if 0.5 < np.median(np.linalg.norm(gv, axis=1)) < 2.0: gv = gv * G
        a = a + np.column_stack([np.interp(ta, tv, gv[:, i]) for i in range(3)])
        note = ' (+ gravity from %s)' % nn
    # common time span, accel interpolated onto gyro timestamps
    order = np.argsort(tg, kind='stable'); tg, w = tg[order], w[order]
    keep = np.r_[True, np.diff(tg) > 0]; tg, w = tg[keep], w[keep]
    m = (tg >= ta.min()) & (tg <= ta.max()); tg, w = tg[m], w[m]
    acc = np.column_stack([np.interp(tg, ta, a[:, i]) for i in range(3)])
    t_ns = np.round(tg * 1e9).astype(np.int64)
    if t_ns[0] < 10 ** 17:  # elapsed seconds only: anchor at a fixed epoch so seconds stay large
        t_ns = t_ns - t_ns[0] + 1_700_000_000_000_000_000
    d = os.path.join(out, 'mav0', 'imu0'); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'data.csv'), 'w') as fh:
        fh.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],'
                 'a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')
        for i in range(len(t_ns)):
            fh.write('%d,%.7f,%.7f,%.7f,%.6f,%.6f,%.6f\n' % (t_ns[i], *w[i], *acc[i]))
    hz = 1.0 / np.median(np.diff(tg))
    print('%s + %s%s -> %s : %d rows, %.1f s, %.0f Hz, |a| median %.2f m/s^2'
          % (gn, an, note, os.path.join(d, 'data.csv'), len(t_ns), tg[-1] - tg[0], hz, np.median(np.linalg.norm(acc, axis=1))))


if __name__ == '__main__':
    main()
