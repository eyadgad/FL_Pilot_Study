#!/usr/bin/env python3
"""Sequential fault-isolated launcher; resume never overwrites completed results."""
import subprocess,sys,json,argparse,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['mnist','cifar10','both'],default='both')
    p.add_argument('--data-root',required=True)
    p.add_argument('--output-root',default='full_study_outputs')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    protocol=json.loads((ROOT/'configs'/'PROTOCOL.json').read_text())
    targets=['mnist','cifar10'] if a.dataset=='both' else [a.dataset]
    for dataset in targets:
        for scenario in protocol['scenarios']:
            name=f'{dataset}_{scenario}'
            cfg=ROOT/'configs'/f'{name}.yaml'
            for seed in protocol['seed_matrix'][name]:
                dest=Path(a.output_root)/name/f'seed_{seed}'
                if (dest/'complete_manifest.json').exists():
                    print('ALREADY_COMPLETE',name,seed,flush=True);continue
                if (dest/'train_manifest.json').exists():
                    meta=json.loads((dest/'train_manifest.json').read_text())
                    if meta.get('status')=='FAIL_TASK_GATE':
                        print('PREVIOUS_FAIL_TASK_GATE',name,seed,flush=True);continue
                    raise FileExistsError(f'Incomplete output {dest}; inspect before continuing. Never overwrite.')
                cmd=[sys.executable,str(ROOT/'scripts'/'run_one.py'),'--config',str(cfg),'--seed',str(seed),
                     '--data-root',a.data_root,'--output-root',a.output_root,'--device',a.device,'--threads',str(a.threads)]
                print('PLAN' if a.dry_run else 'RUN',name,seed,flush=True)
                if not a.dry_run:
                    subprocess.run(cmd,check=True,cwd=ROOT,env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8'})

if __name__=='__main__':main()
