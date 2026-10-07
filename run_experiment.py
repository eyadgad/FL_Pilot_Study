#!/usr/bin/env python3
import argparse
from multiprocessing import freeze_support
from ucpa_fl.runner import run_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args()
    for run_dir, _ in run_config(a.config):
        print(run_dir)


if __name__ == '__main__':
    freeze_support()
    main()

