#!/usr/bin/env python3
import argparse
from ucpa_fl.aggregate import aggregate
p=argparse.ArgumentParser();p.add_argument('--outputs',default='./outputs');p.add_argument('--out',default='./aggregate');a=p.parse_args()
aggregate(a.outputs,a.out);print(a.out)
