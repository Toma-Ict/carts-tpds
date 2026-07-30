#!/usr/bin/env python3
# verify_necessity_numbers.py -- READ-ONLY check before composite-figure promotion
# (1) CA 12-cell means + seed counts vs Table V
# (2) C4 12-cell means + seed counts vs Table VI
# (3) NAIVE means at dpu=20500 vs LOCKED_OLD (make_necessity_fig_v3.py)
import csv, statistics as st
from collections import defaultdict

CSV   = "sensitivity_gpd8_results.csv"
C4CSV = "c4_always_dedup_results.csv"
BWS   = [5, 10, 20, 50]
DPUS  = [20500, 50000, 83900]
TOL   = 0.05

TABLE_V = {
 (5,20500):(10.3,6),(5,50000):(6.9,4),(5,83900):(6.9,4),
 (10,20500):(6.1,6),(10,50000):(11.8,6),(10,83900):(9.6,5),
 (20,20500):(3.4,6),(20,50000):(6.3,6),(20,83900):(10.7,6),
 (50,20500):(3.4,6),(50,50000):(3.4,6),(50,83900):(3.4,6)}
TABLE_VI = {
 (5,20500):(6.3,4),(5,50000):(6.9,4),(5,83900):(7.2,4),
 (10,20500):(4.7,4),(10,50000):(6.4,4),(10,83900):(6.8,4),
 (20,20500):(-2.6,4),(20,50000):(5.3,4),(20,83900):(6.2,4),
 (50,20500):(-32.7,0),(50,50000):(-3.4,3),(50,83900):(3.0,4)}
LOCKED_OLD = {5:6.94, 10:5.83, 20:-1.49, 50:-32.70}

def load(path, col):
    d = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(int(r["bw_gbs"]), int(r["dpu_bytes_per_us"]))].append(float(r[col]))
    return d

def report(name, data, table):
    print("\n== %s ==" % name)
    ok = True
    for bw in BWS:
        for dpu in DPUS:
            v = data[(bw, dpu)]
            m = st.mean(v)
            ge = sum(1 for x in v if x >= 0)
            em, eg = table[(bw, dpu)]
            good = abs(m - em) <= TOL and ge == eg and len(v) == 6
            if not good: ok = False
            print("bw=%2d dpu=%5d: mean=%+7.2f (exp %+6.1f)  seeds>=0: %d/%d (exp %d/6)  %s"
                  % (bw, dpu, m, em, ge, len(v), eg, "OK" if good else "MISMATCH"))
    return ok

ca  = load(CSV, "new_vs_c0")
c4  = load(C4CSV, "c4_vs_c0")
old = load(CSV, "old_vs_c0")

ok1 = report("CA vs Table V", ca, TABLE_V)
ok2 = report("C4 vs Table VI", c4, TABLE_VI)

print("\n== NAIVE (dpu=20500) vs LOCKED_OLD ==")
ok3 = True
for bw in BWS:
    m = st.mean(old[(bw, 20500)])
    e = LOCKED_OLD[bw]
    good = abs(m - e) <= TOL
    if not good: ok3 = False
    print("bw=%2d: mean=%+7.2f (exp %+7.2f)  %s" % (bw, m, e, "OK" if good else "MISMATCH"))

print("\nALL CHECKS PASSED -- numbers safe to promote" if (ok1 and ok2 and ok3)
      else "\nSOME CHECKS FAILED -- do NOT promote, investigate first")
