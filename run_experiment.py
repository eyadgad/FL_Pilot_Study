#!/usr/bin/env python3
import argparse
from ucpa_fl.config import load_config
from ucpa_fl.runner import run_one
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--seed',type=int,default=None);a=p.parse_args()
cfg=load_config(a.config)
for seed in ([a.seed] if a.seed is not None else cfg.seeds):
    if int(seed) not in cfg.seeds: p.error(f'seed {seed} is not in frozen config')
    d,_=run_one(cfg,int(seed));print(d)
