#!/usr/bin/env python3
import argparse
from multiprocessing import freeze_support
from ucpa_fl.config import load_config
from ucpa_fl.runner import run_one


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--seed', type=int, default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    for seed in ([a.seed] if a.seed is not None else cfg.seeds):
        if int(seed) not in cfg.seeds:
            p.error(f'seed {seed} is not in frozen config')
        d, _ = run_one(cfg, int(seed))
        print(d)


if __name__ == '__main__':
    freeze_support()
    main()

