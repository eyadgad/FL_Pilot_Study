#!/usr/bin/env python3
import argparse
from ucpa_fl.runner import run_config
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args()
for run_dir,_ in run_config(a.config): print(run_dir)
