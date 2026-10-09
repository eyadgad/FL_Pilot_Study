#!/usr/bin/env python3
"""Real-data worker-isolation check without retraining or synthetic images."""
import argparse,subprocess,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);a=p.parse_args()
    snippet="""import sys,json,os,time
from pathlib import Path
from study_v7.data import verify_data
r=verify_data('mnist',sys.argv[1]);print(json.dumps({'pid':os.getpid(),'train':r['train_count'],'test':r['test_count']}),flush=True)
"""
    start=time.perf_counter()
    procs=[subprocess.Popen([sys.executable,'-c',snippet,str(Path(a.data_root).resolve())],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
    results=[]
    for proc in procs:
        out,err=proc.communicate(timeout=90)
        if proc.returncode!=0:raise RuntimeError(err)
        results.append(json.loads(out))
    assert len({r['pid'] for r in results})==2 and all((r['train'],r['test'])==(60000,10000) for r in results)
    print(json.dumps({'status':'PASS','parallel_real_data_subprocesses':2,'separate_pids':True,'seconds':round(time.perf_counter()-start,3)},indent=2))

if __name__=='__main__':main()
