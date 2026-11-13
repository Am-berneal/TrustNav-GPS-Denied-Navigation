"""Measure the Track 3 execution budgets for src/main.py on one IMU file.
peak memory : tracemalloc peak of the algorithm's own allocations, and process peak RSS (includes the
              Python interpreter itself, ~9 MB on CPython before any work)
runtime     : CPU time of the whole run, divided by IMU duration -> ms per 1 s of IMU (budget 20 ms)"""
import sys, time, tracemalloc, resource, json, os, subprocess
def measure(imu, out, extra=()):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    import main as M
    cfg = M.parse([imu, out] + list(extra))
    tracemalloc.start(); c0 = time.process_time()
    M.run(cfg)
    cpu = time.process_time() - c0; cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    with open(imu) as fh:
        ts = [l.split(',', 1)[0] for l in fh if l[0].isdigit()]
    dur = (int(ts[-1]) - int(ts[0])) * 1e-9
    # separate clean process for wall time and RSS (no tracemalloc overhead)
    code = ('import resource,time,sys;sys.argv=["main.py"]+%r;t=time.process_time();'
            'import runpy;runpy.run_path(%r,run_name="__main__");'
            'print(time.process_time()-t, [l for l in open("/proc/self/status") if l.startswith("VmHWM")][0].split()[1])') % ([imu, out] + list(extra), os.path.join(os.path.dirname(__file__), '..', 'src', 'main.py'))
    o = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True).stdout.split()
    base = subprocess.run([sys.executable, '-c', 'print([l for l in open("/proc/self/status") if l.startswith("VmHWM")][0].split()[1])'], capture_output=True, text=True).stdout
    return {'imu_seconds': dur, 'cpu_s': float(o[0]), 'ms_per_imu_s': 1000 * float(o[0]) / dur,
            'tracemalloc_peak_mb': peak / 2**20, 'rss_peak_mb': int(o[1]) / 1024, 'python_baseline_rss_mb': int(base) / 1024}
if __name__ == '__main__':
    print(json.dumps(measure(sys.argv[1], sys.argv[2], sys.argv[3:]), indent=1))
