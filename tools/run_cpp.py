"""Run the C++ build on every flight: accuracy, Python/C++ agreement, peak RSS (VmHWM) and CPU time."""
import subprocess, sys, json, os, resource, numpy as np
sys.path.insert(0,'tools'); from evaluate import evaluate; from fit_params import score
SEQ=['MH_01','MH_03','MH_05','V1_02','V2_01','V2_02','V2_03']
out={}
for s in SEQ:
    d=f'data/deg/{s}'; o=f'data/runs/{s}_cpp.txt'
    # wrapper reads the child's VmHWM just before it exits by polling /proc is racy; use wait4 rusage instead
    pid=os.fork()
    if pid==0:
        fd=os.open('/tmp/cpp_err.txt',os.O_WRONLY|os.O_CREAT|os.O_TRUNC); os.dup2(fd,2)
        os.execv('./trustnav',['./trustnav',f'{d}/mav0/imu0/data.csv',o])
    _,status,ru=os.wait4(pid,0)
    ev=json.load(open(f'{d}/events.json')); dur=ev['failure']
    r=evaluate(o,f'{d}/gt.txt',f'{d}/events.json'); r.pop('_series')
    py=np.loadtxt(f'data/runs/{s}_full.txt'); cp=np.loadtxt(o)
    diff=float(np.abs(py[:,1:4]-cp[:,1:4]).max()) if len(py)==len(cp) else None
    out[s]={'ate_rmse':r['ate_rmse'],'score':score(r),'rss_peak_mb':int([l for l in open('/tmp/cpp_err.txt') if 'VmHWM' in l][0].split()[-2])/1024,'cpu_s':ru.ru_utime+ru.ru_stime,
            'ms_per_imu_s':1000*(ru.ru_utime+ru.ru_stime)/dur,'rows':len(cp),'py_rows':len(py),'max_pos_diff_vs_python_m':diff}
    print(s,{k:(round(v,4) if isinstance(v,float) else v) for k,v in out[s].items()})
json.dump(out,open('data/runs/cpp.json','w'),indent=1)
