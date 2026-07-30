#!/usr/bin/env python3
"""
ablation_gpd8_multiseed.py -- multi-seed dual-metric ablation at the gpd=8
headline regime (32 GPUs, 4 domains), seeds 0-5.

Extends ablation_gpd8.py (single seed) to the multi-seed discipline used for
every other headline claim, so the ablation is no longer a one-seed result.

Per-seed cap via pick_cap: cap is seed/topology-specific, never reused across
seeds. Seed 0 is anchor-gated against the locked S25 numbers (cap=541,
cost-aware gain +17.49%); if seed 0 fails, nothing is written.

Both C3 variants (naive Alg.1, cost-aware Alg.2) are carried separately at
every seed, so Layer-1 and Layer-2 are never taken from different algorithms.

Append-only: new file, reuses existing helpers unmodified.
Writes ablation_gpd8_multiseed_results.csv (per-seed rows).
"""
import os, sys, csv
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from demand_extractor import MoEConfig, generate_demand
from greedy_crp import (CRPConfig, greedy_crp,
                        CRPConfigCostAware, greedy_crp_cost_aware)
from carts_et_writer import WriterConfig
from make_2rank_workload import write_2rank_workload
from sweep_any_n import emit_ets, write_system_json, pick_cap
from scale_multiseed import run_cycles_n
from make_ablation import random_crp, force_no_dedup, C1_SEEDS

GPD = 8
NUM_GPUS = 32
SEEDS = [0, 1, 2, 3, 4, 5]
SKEW = 1.5
BW = 5
DPU = 20500
BASE = "/workspace/astra-sim/examples"
NET = "examples/network/analytical/Switch_32npus_slow.yml"
ANCHOR = dict(seed=0, cap=541, c3c_pct=17.49, tol=0.10)
CFGS = ["C0", "C2", "C1", "C3n", "C3c"]


def max_link_load(placement):
    return max(placement.load[p] for p in placement.inter_pairs())


def pct(base, x):
    return 100.0 * (base - x) / base


def emit_and_run(name, placement, wcfg, wl):
    emit_ets(os.path.join(BASE, "system/custom_collectives", name),
             name, placement, wcfg)
    write_system_json(name)
    return run_cycles_n(name, wl, NET)


def run_seed(seed, wl):
    """One seed: pick cap, build 9 placements, simulate, collapse C1 draws.
    Returns (cap, l1dict, cycdict, c1_l1_list, c1_cyc_list) or None on failure."""
    dem = generate_demand(MoEConfig(skew=SKEW, seed=seed, num_gpus=NUM_GPUS,
                                    gpus_per_domain=GPD, num_experts=NUM_GPUS))
    cap, _ = pick_cap(dem)
    wcfg = WriterConfig(dpu_bytes_per_us=DPU)
    link_bw = BW * 1000.0

    pl = {"C0": greedy_crp(dem, CRPConfig(relay_capacity=0)),
          "C2": force_no_dedup(greedy_crp(dem, CRPConfig(relay_capacity=cap))),
          "C3n": greedy_crp(dem, CRPConfig(relay_capacity=cap)),
          "C3c": greedy_crp_cost_aware(dem, CRPConfigCostAware(
              relay_capacity=cap, dpu_bytes_per_us=DPU,
              link_bw_bytes_per_us=link_bw))}
    for s in C1_SEEDS:
        pl["C1_s%d" % s] = random_crp(dem, CRPConfig(relay_capacity=cap), s)

    l1, cyc = {}, {}
    for name, p in pl.items():
        l1[name] = float(max_link_load(p))
        c = emit_and_run("ab8m_%s" % name.lower(), p, wcfg, wl)
        if c is None:
            return None
        cyc[name] = float(c)

    c1k = ["C1_s%d" % s for s in C1_SEEDS]
    c1_l1 = [l1[k] for k in c1k]
    c1_cy = [cyc[k] for k in c1k]
    l1["C1"] = sum(c1_l1) / len(c1_l1)
    cyc["C1"] = sum(c1_cy) / len(c1_cy)
    return cap, l1, cyc, c1_l1, c1_cy


def main():
    wl_dir = os.path.join(
        BASE, "workload/microbenchmarks/all_to_all/scale_%dnpus" % NUM_GPUS)
    wl = "examples/workload/microbenchmarks/all_to_all/scale_%dnpus/carts_wl" % NUM_GPUS
    write_2rank_workload(wl_dir, 1048576, num_ranks=NUM_GPUS)

    print("=" * 96)
    print("MULTI-SEED DUAL-METRIC ABLATION  gpd=%d  %d GPUs (%d dom)  "
          "bw=%d GB/s  dpu=%d B/us  seeds=%s"
          % (GPD, NUM_GPUS, NUM_GPUS // GPD, BW, DPU, SEEDS))
    print("=" * 96)

    rows, per_seed = [], {}
    for seed in SEEDS:
        res = run_seed(seed, wl)
        if res is None:
            print("RUN FAILED at seed %d -- aborting, nothing written" % seed)
            return 1
        cap, l1, cyc, c1_l1, c1_cy = res
        per_seed[seed] = (cap, l1, cyc)

        if seed == ANCHOR["seed"]:
            g = pct(cyc["C0"], cyc["C3c"])
            print("\nANCHOR (seed %d): cap=%d (locked %d)   C3c gain=%+.2f%% "
                  "(locked %+.2f%%)" % (seed, cap, ANCHOR["cap"], g,
                                        ANCHOR["c3c_pct"]))
            if cap != ANCHOR["cap"] or abs(g - ANCHOR["c3c_pct"]) > ANCHOR["tol"]:
                print("ANCHOR FAIL -- DO NOT PROMOTE. (no CSV written)")
                return 1
            print("ANCHOR OK\n")

        print("seed %d  cap=%4d | " % (seed, cap) + "  ".join(
            "%s L1=%6.1f(%+5.1f%%) L2=%8.0f(%+5.1f%%)"
            % (c, l1[c], pct(l1["C0"], l1[c]), cyc[c], pct(cyc["C0"], cyc[c]))
            for c in ("C1", "C3c")))
        for c in CFGS:
            rows.append([seed, cap, c, round(l1[c], 1),
                         round(pct(l1["C0"], l1[c]), 3), round(cyc[c], 1),
                         round(pct(cyc["C0"], cyc[c]), 3)])
        rows.append([seed, cap, "C1_draws", str(c1_l1), "", str(c1_cy), ""])

    print("\n" + "=" * 96)
    print("AGGREGATE OVER %d SEEDS" % len(SEEDS))
    print("=" * 96)
    print("%-5s %10s %8s %8s %10s %8s %8s %8s"
          % ("cfg", "dL1 mean", "min", "max", "dL2 mean", "min", "max", ">=C0"))
    print("-" * 96)
    agg = {}
    for c in CFGS:
        d1 = [pct(per_seed[s][1]["C0"], per_seed[s][1][c]) for s in SEEDS]
        d2 = [pct(per_seed[s][2]["C0"], per_seed[s][2][c]) for s in SEEDS]
        nn = sum(1 for v in d2 if v >= -1e-9)
        agg[c] = (sum(d1) / len(d1), sum(d2) / len(d2))
        print("%-5s %9.2f%% %7.2f%% %7.2f%% %9.2f%% %7.2f%% %7.2f%% %5d/%d"
              % (c, agg[c][0], min(d1), max(d1), agg[c][1], min(d2), max(d2),
                 nn, len(SEEDS)))
    print("-" * 96)

    same = sum(1 for s in SEEDS
               if abs(per_seed[s][2]["C3n"] - per_seed[s][2]["C3c"]) < 1e-9)
    print("C3n == C3c (identical cycles) on %d of %d seeds" % (same, len(SEEDS)))
    om = 0
    for s in SEEDS:
        _, l1, cyc = per_seed[s]
        o1 = sorted(["C1", "C2", "C3c"], key=lambda c: l1[c])
        o2 = sorted(["C1", "C2", "C3c"], key=lambda c: cyc[c])
        om += (o1 == o2)
        print("  seed %d  L1 order %s | L2 order %s | match %s"
              % (s, " < ".join(o1), " < ".join(o2), o1 == o2))
    print("ORDER MATCH (v0.8 6.3 causal standard): %d of %d seeds"
          % (om, len(SEEDS)))

    out = os.path.join(_HERE, "ablation_gpd8_multiseed_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "cap", "config", "layer1_max_load",
                    "layer1_pct_vs_c0", "layer2_cycles", "layer2_pct_vs_c0"])
        w.writerows(rows)
    print("\nwrote %s  (%d rows)" % (out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
