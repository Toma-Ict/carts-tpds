#!/usr/bin/env python3
"""
ablation_gpd8.py -- dual-metric ablation at the gpd=8 HEADLINE regime
(32 GPUs, 4 domains), mirroring make_ablation.py's 16-GPU / gpd=4 design.

Why: the paper's Table V ablation runs at gpd=4 (16 GPUs), near the envelope
boundary of Section VII-F. Repeating it at the sweet spot puts the ablation in
the same regime as the headline.

Reports BOTH C3 variants separately (naive Alg.1 vs cost-aware Alg.2) so
Layer-1 and Layer-2 are never taken from different algorithms -- the pairing
error found in the S22 dual-metric table.

Append-only: new file, reuses sweep_any_n / scale_multiseed / make_ablation
helpers unmodified. Anchor-gated: aborts before simulating if the config does
not reproduce the locked S25 gpd=8 seed-0 cap, and refuses to write results if
the cost-aware gain does not reproduce +17.49%.
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
SEED = 0
SKEW = 1.5
BW = 5
DPU = 20500
BASE = "/workspace/astra-sim/examples"
NET = "examples/network/analytical/Switch_32npus_slow.yml"
ANCHOR = dict(cap=541, new_vs_c0=17.49, tol=0.10)


def max_link_load(placement):
    return max(placement.load[p] for p in placement.inter_pairs())


def pct(base, x):
    return 100.0 * (base - x) / base


def emit_and_run(name, placement, wcfg, wl):
    emit_ets(os.path.join(BASE, "system/custom_collectives", name),
             name, placement, wcfg)
    write_system_json(name)
    return run_cycles_n(name, wl, NET)


def main():
    wl_dir = os.path.join(
        BASE, "workload/microbenchmarks/all_to_all/scale_%dnpus" % NUM_GPUS)
    wl = "examples/workload/microbenchmarks/all_to_all/scale_%dnpus/carts_wl" % NUM_GPUS
    write_2rank_workload(wl_dir, 1048576, num_ranks=NUM_GPUS)

    dem = generate_demand(MoEConfig(skew=SKEW, seed=SEED, num_gpus=NUM_GPUS,
                                    gpus_per_domain=GPD, num_experts=NUM_GPUS))
    cap, _ = pick_cap(dem)
    print("gpd=%d  num_gpus=%d  (%d domains)  seed=%d  bw=%d GB/s  dpu=%d B/us"
          % (GPD, NUM_GPUS, NUM_GPUS // GPD, SEED, BW, DPU))
    print("pick_cap -> %d   (locked S25 anchor: %d)" % (cap, ANCHOR["cap"]))
    if cap != ANCHOR["cap"]:
        print("ANCHOR FAIL: cap mismatch -- config differs from the locked "
              "gpd=8 sweep. Aborting before any simulation.")
        return 1

    wcfg = WriterConfig(dpu_bytes_per_us=DPU)
    link_bw = BW * 1000.0
    pl = {}
    pl["C0"] = greedy_crp(dem, CRPConfig(relay_capacity=0))
    pl["C2"] = force_no_dedup(greedy_crp(dem, CRPConfig(relay_capacity=cap)))
    pl["C3n"] = greedy_crp(dem, CRPConfig(relay_capacity=cap))
    pl["C3c"] = greedy_crp_cost_aware(dem, CRPConfigCostAware(
        relay_capacity=cap, dpu_bytes_per_us=DPU, link_bw_bytes_per_us=link_bw))
    for s in C1_SEEDS:
        pl["C1_s%d" % s] = random_crp(dem, CRPConfig(relay_capacity=cap), s)

    print("\nrunning %d configs ..." % len(pl))
    cyc, l1 = {}, {}
    for name, p in pl.items():
        l1[name] = float(max_link_load(p))
        c = emit_and_run("ab8_%s" % name.lower(), p, wcfg, wl)
        if c is None:
            print("RUN FAILED: %s" % name)
            return 1
        cyc[name] = float(c)
        print("  %-8s L1=%9.1f   L2=%10d" % (name, l1[name], c))

    c1k = ["C1_s%d" % s for s in C1_SEEDS]
    l1["C1"] = sum(l1[k] for k in c1k) / len(c1k)
    cyc["C1"] = sum(cyc[k] for k in c1k) / len(c1k)

    gain_new = pct(cyc["C0"], cyc["C3c"])
    print("\nANCHOR CHECK  cost-aware gain vs C0 = %+.2f%%   (locked %+.2f%%)"
          % (gain_new, ANCHOR["new_vs_c0"]))
    if abs(gain_new - ANCHOR["new_vs_c0"]) > ANCHOR["tol"]:
        print("ANCHOR FAIL: does not reproduce the locked gpd=8 seed-0 "
              "number. DO NOT PROMOTE. (no CSV written)")
        return 1
    print("ANCHOR OK")

    order = [("C0", "baseline"),
             ("C2", "placement only, dedup off"),
             ("C1", "random relay + dedup (%d draws)" % len(c1k)),
             ("C3n", "naive placement (Alg 1) + dedup"),
             ("C3c", "cost-aware placement (Alg 2) + dedup")]
    print("\n" + "=" * 90)
    print("DUAL-METRIC ABLATION  gpd=%d, %d GPUs, cap=%d, dpu=%d, bw=%d, seed=%d"
          % (GPD, NUM_GPUS, cap, DPU, BW, SEED))
    print("=" * 90)
    print("%-5s %-34s %10s %9s %12s %9s"
          % ("cfg", "description", "L1 load", "dL1", "L2 cycles", "dL2"))
    print("-" * 90)
    rows = []
    for k, desc in order:
        r = [k, desc, round(l1[k], 1), round(pct(l1["C0"], l1[k]), 2),
             round(cyc[k], 1), round(pct(cyc["C0"], cyc[k]), 2)]
        rows.append(r)
        print("%-5s %-34s %10.1f %8.2f%% %12.1f %8.2f%%"
              % (k, desc, r[2], r[3], r[4], r[5]))
    print("-" * 90)
    print("C1 per-draw L2 : %s" % [int(cyc[k]) for k in c1k])
    print("C1 per-draw L1 : %s" % [l1[k] for k in c1k])
    print("order L1 (best first, C0 excluded): %s"
          % " < ".join(sorted(["C1", "C2", "C3n", "C3c"], key=lambda c: l1[c])))
    print("order L2 (best first, C0 excluded): %s"
          % " < ".join(sorted(["C1", "C2", "C3n", "C3c"], key=lambda c: cyc[c])))

    out = os.path.join(_HERE, "ablation_gpd8_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "description", "layer1_max_load",
                    "layer1_pct_vs_c0", "layer2_cycles", "layer2_pct_vs_c0"])
        w.writerows(rows)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
