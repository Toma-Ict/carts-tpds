#!/usr/bin/env python3
"""READ-ONLY: dump the 4x4 lambda / unique matrices for Fig 5 panel 2."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from demand_extractor import MoEConfig, generate_demand

dem = generate_demand(MoEConfig(skew=1.5, seed=0, num_gpus=32,
                                gpus_per_domain=8, num_experts=32))
n = dem.cfg.num_domains
for title, fn in (("lambda (raw)", dem.lam), ("unique (after dedup)", dem.unique)):
    print("\n%s" % title)
    print("      " + "".join("%8s" % ("d%d" % j) for j in range(n)))
    for i in range(n):
        cells = "".join("%8s" % ("--" if i == j else fn(i, j)) for j in range(n))
        print("  d%d  %s" % (i, cells))
