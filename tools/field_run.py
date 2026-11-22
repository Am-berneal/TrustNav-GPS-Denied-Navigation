"""One command for a phone field test.

  python tools/field_run.py <phone export folder or .zip> <waypoints.json> field_runs/<name>

1. converts the phone export to EuRoC imu0/data.csv           (tools/phone_to_euroc.py)
2. degrades it with the organisers' degrade_imu.py, if present (tools/official/degrade_imu.py)
3. runs TrustNav (src/main.py) on the clean and the degraded IMU, plus raw integration for contrast
4. checks one pose per input row                              (tools/validate_output.py)
5. scores every run against the measured floor checkpoints    (tools/field_test.py)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PY = sys.executable
RAW = ['--no-init', '--no-trust', '--no-tilt', '--no-prior', '--no-zupt', '--vmax', '0', '--drag-k', '0', '--no-blind-model']
HANDHELD = ['--drag-k', '0']  # rotor-drag aiding is drone physics: off for a hand-carried phone


def run(cmd, **kw):
    print('$', ' '.join(cmd)); return subprocess.run(cmd, check=True, **kw)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    src, wps, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    run([PY, os.path.join(HERE, 'phone_to_euroc.py'), src, out])
    clean = os.path.join(out, 'mav0', 'imu0', 'data.csv')
    main_py = os.path.join(ROOT, 'src', 'main.py')
    runs = []
    off = os.path.join(HERE, 'official', 'degrade_imu.py')
    deg = os.path.join(out, 'degraded.csv')
    if os.path.exists(off):
        run([PY, off, clean, deg, '--meta', os.path.join(out, 'degrade_meta.json')])
        est = os.path.join(out, 'est_degraded.txt')
        run([PY, main_py, deg, est] + HANDHELD)
        run([PY, os.path.join(HERE, 'validate_output.py'), est, deg, '--meta', os.path.join(out, 'meta_tmp.json'),
             '--json', os.path.join(out, 'validation.json')])
        runs.append((est, 'TrustNav, official degradation'))
    else:
        print('(no tools/official/degrade_imu.py: skipping the degraded run)')
    for args, name, f in [(HANDHELD, 'TrustNav, clean phone IMU', 'est_clean.txt'),
                          ([], 'TrustNav drone preset, clean', 'est_clean_drone.txt'),
                          (RAW, 'raw integration, clean', 'est_raw.txt')]:
        est = os.path.join(out, f)
        run([PY, main_py, clean, est] + args)
        runs.append((est, name))
    cmd = [PY, os.path.join(HERE, 'field_test.py'), clean, runs[0][0], wps, os.path.join(out, 'field'), '--label', runs[0][1]]
    for est, name in runs[1:]:
        cmd += ['--compare', est, name]
    run(cmd)


if __name__ == '__main__':
    main()
