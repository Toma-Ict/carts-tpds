#!/usr/bin/env python3
# verify_headline_seeds.py -- READ-ONLY check before Fig 7 redesign promotion
# anchors: Table IV (gpd=8) means + >=C0 counts; S30 (gpd=16) means + per-seed
import csv, statistics as st
from collections import defaultdict

ANCH_G8  = {4: (10.3, 6), 8: (5.9, 6), 16: (1.0, 4)}
ANCH_G16 = {4: (9.45, 6), 8: (5.34, 6)}
S30_G16_4DOM = [24.2, 0.6, 3.8, 13.3, 14.9, 0.0]
TOL = 0.05

def load(path):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        d[int(r["num_domains"])][int(r["seed"])] = float(r["new_vs_c0"])
    return d

def report(name, data, anch):
    ok = True
    print("== %s ==" % name)
    for dom, (em, eg) in sorted(anch.items()):
        v = [data[dom][s] for s in sorted(data[dom])]
        m, ge = st.mean(v), sum(1 for x in v if x >= 0)
        good = abs(m - em) <= TOL and ge == eg and len(v) == 6
        if not good: ok = False
        print("dom=%2d mean=%+7.2f (exp %+5.2f) >=C0 %d/6 (exp %d)  %s"
              % (dom, m, em, ge, eg, "OK" if good else "MISMATCH"))
        print("   per-seed:", ["%+.2f" % x for x in v])
    return ok

g8, g16 = load("scale_multiseed_gpd8_results.csv"), load("scale_multiseed_gpd16_results.csv")
ok1 = report("gpd=8 vs Table IV", g8, ANCH_G8)
ok2 = report("gpd=16 vs S30", g16, ANCH_G16)
ok3 = all(abs(round(g16[4][s], 1) - S30_G16_4DOM[s]) < 0.051 for s in range(6))
print("gpd=16 4-dom per-seed vs S30 log:", "OK" if ok3 else "MISMATCH")
print("ALL CHECKS PASSED -- safe to promote" if (ok1 and ok2 and ok3)
      else "SOME CHECKS FAILED -- do NOT promote")
