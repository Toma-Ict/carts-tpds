#!/usr/bin/env python3
"""
verify_ablation_numbers.py -- READ-ONLY verification of the paper's Table V
(dual-metric ablation). Writes nothing, modifies nothing, runs no simulation.

Layer-1 (max inter-domain link load) is recomputed from the same Placement
objects dual_metric_table.py builds. Layer-2 (cycles) anchors are cross-checked
against the existing sweep CSVs.

Open question this resolves: dual_metric_table.py line 46 builds C3 with
greedy_crp (NAIVE) while its Layer-2 anchor is the cost-aware Fix B run.
This script computes BOTH placements and reports whether they agree.
"""
import os, sys, csv
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from demand_extractor import MoEConfig, generate_demand
from greedy_crp import (CRPConfig, greedy_crp, CRPConfigCostAware,
                        greedy_crp_cost_aware)
from make_ablation import random_crp, force_no_dedup, CAP, C1_SEEDS

PAPER_L1 = {"C0": 252.0, "C2": 252.0, "C1": 249.8, "C3": 218.0}
PAPER_L1_PCT = {"C2": 0.0, "C1": 0.9, "C3": 13.5}
PAPER_L2 = {"C0": 392004, "C2": 392004, "C1": 353400, "C3": 362250}
PAPER_L2_PCT = {"C2": 0.0, "C1": 9.9, "C3": 7.6}

DPU_BYTES_PER_US = 20500
TOKEN_BYTES = 2048
LINK_BW_BYTES_PER_US = 5000.0
TOL_L1 = 0.05
TOL_PCT = 0.05


def max_link_load(placement):
    pairs = placement.inter_pairs()
    return max(placement.load[p] for p in pairs)


def pct_vs(base, val):
    return 100.0 * (base - val) / base


def check(label, got, want, tol, unit=""):
    ok = abs(got - want) <= tol
    print("  [%s] %-30s recomputed=%10.2f%s   paper=%10.2f%s   d=%+.3f"
          % ("PASS" if ok else "FAIL", label, got, unit, want, unit, got - want))
    return ok


def scan_csv(paths, values):
    hits = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                vals = set(row.values())
                for v in values:
                    if str(v) in vals:
                        hits.append((os.path.basename(p), v, row))
    return hits


def main():
    ok = True
    dem = generate_demand(MoEConfig(skew=1.5, seed=0))

    c0 = greedy_crp(dem, CRPConfig(relay_capacity=0))
    c2 = force_no_dedup(greedy_crp(dem, CRPConfig(relay_capacity=CAP)))
    c3n = greedy_crp(dem, CRPConfig(relay_capacity=CAP))
    c3c = greedy_crp_cost_aware(dem, CRPConfigCostAware(
        relay_capacity=CAP,
        dpu_bytes_per_us=DPU_BYTES_PER_US,
        token_bytes=TOKEN_BYTES,
        link_bw_bytes_per_us=LINK_BW_BYTES_PER_US))
    c1_seeds = list(C1_SEEDS)
    c1_loads = [float(max_link_load(random_crp(dem, CRPConfig(relay_capacity=CAP), s)))
                for s in c1_seeds]

    l1 = {"C0": float(max_link_load(c0)),
          "C2": float(max_link_load(c2)),
          "C1": sum(c1_loads) / len(c1_loads),
          "C3n": float(max_link_load(c3n)),
          "C3c": float(max_link_load(c3c))}

    print("=" * 78)
    print("TABLE V VERIFICATION  (16 GPUs, cap=%d, dpu=%d B/us, demand seed 0)"
          % (CAP, DPU_BYTES_PER_US))
    print("=" * 78)
    print("\n[1] LAYER-1  max inter-domain link load")
    ok &= check("C0 baseline", l1["C0"], PAPER_L1["C0"], TOL_L1)
    ok &= check("C2 placement only", l1["C2"], PAPER_L1["C2"], TOL_L1)
    ok &= check("C1 random+dedup (mean)", l1["C1"], PAPER_L1["C1"], TOL_L1)
    print("      C1 per-seed loads %s over seeds %s" % (c1_loads, c1_seeds))

    print("\n[2] C3 PLACEMENT VARIANT CHECK  (the open question)")
    print("      C3 via greedy_crp        (NAIVE, as dual_metric_table.py) = %.2f"
          % l1["C3n"])
    print("      C3 via cost_aware        (Fix B, as the L2 anchor)        = %.2f"
          % l1["C3c"])
    agree = abs(l1["C3n"] - l1["C3c"]) <= TOL_L1
    print("      -> variants agree: %s" % ("YES (no problem)" if agree
                                           else "NO  (paper pairs mismatched placements)"))
    ok &= check("C3 naive     vs paper 218", l1["C3n"], PAPER_L1["C3"], TOL_L1)
    ok &= check("C3 cost-aware vs paper 218", l1["C3c"], PAPER_L1["C3"], TOL_L1)

    print("\n[3] LAYER-1 PERCENTAGES vs C0")
    for cfg, key in (("C2", "C2"), ("C1", "C1"), ("C3", "C3n")):
        ok &= check("dL1 %s" % cfg, pct_vs(l1["C0"], l1[key]),
                    PAPER_L1_PCT[cfg], TOL_PCT, "%")

    print("\n[4] LAYER-2 PERCENTAGES vs C0  (arithmetic on locked anchors)")
    for cfg in ("C2", "C1", "C3"):
        ok &= check("dL2 %s" % cfg, pct_vs(PAPER_L2["C0"], PAPER_L2[cfg]),
                    PAPER_L2_PCT[cfg], TOL_PCT, "%")

    print("\n[5] LAYER-2 CYCLES CORROBORATION IN SWEEP CSVs")
    csvs = [os.path.join(_HERE, n) for n in
            ("sweep_multi_seed_results.csv", "sweep_any_n_results.csv",
             "scale_multiseed_gpd8_results.csv", "sensitivity_gpd8_results.csv")]
    for v in (PAPER_L2["C0"], PAPER_L2["C3"], PAPER_L2["C1"]):
        hits = scan_csv(csvs, [v])
        print("      %d -> %d row(s)" % (v, len(hits)))
        for fn, _, row in hits[:3]:
            print("         %s :: %s" % (fn, dict(list(row.items())[:6])))

    print("\n" + "=" * 78)
    print("RESULT: %s" % ("ALL ANCHORS MATCH" if ok else "MISMATCH FOUND - DO NOT PROMOTE"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
