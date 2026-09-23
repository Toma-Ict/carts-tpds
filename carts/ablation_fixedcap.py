#!/usr/bin/env python3
"""
ablation_fixedcap.py -- bounded sanity check on the capacity budget.

The headline, ablation and robustness runs all take their relay capacity from
pick_cap(), which selects the value maximising the C1(random) - C3(greedy) gap
on each seed's own demand. That makes capacity an outcome-dependent parameter.
This script repeats the gpd=8 ablation at capacities FIXED IN ADVANCE and
shared across all seeds, to test whether the cost-aware advantage survives
when the budget is not chosen from the data.

Caps are declared before any result is inspected:
  400   clearly below the headline's per-seed values (which run 408-541)
  600   clearly above them
  10**9 unbounded, capacity never binds

Append-only: imports ablation_gpd8_multiseed unmodified and overrides only the
capacity. That module's anchor gate is cap-specific (expects cap=541) and so
is deliberately not applied here; instead C0 is recomputed per seed and every
gain is reported relative to it, exactly as in the original ablation.

Writes ablation_fixedcap_results.csv.
"""
import os, sys, csv
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ablation_gpd8_multiseed as AB
from demand_extractor import MoEConfig, generate_demand
from greedy_crp import (CRPConfig, greedy_crp,
                        CRPConfigCostAware, greedy_crp_cost_aware)
from carts_et_writer import WriterConfig
from make_2rank_workload import write_2rank_workload
from sweep_any_n import pick_cap
from make_ablation import random_crp, force_no_dedup, C1_SEEDS

FIXED_CAPS = [400, 600, 10**9]
SEEDS = AB.SEEDS
CFGS = AB.CFGS


def run_seed_fixed(seed, wl, cap, tag):
    """One seed at one fixed cap. Mirrors AB.run_seed but with cap injected."""
    dem = generate_demand(MoEConfig(skew=AB.SKEW, seed=seed,
                                    num_gpus=AB.NUM_GPUS,
                                    gpus_per_domain=AB.GPD,
                                    num_experts=AB.NUM_GPUS))
    wcfg = WriterConfig(dpu_bytes_per_us=AB.DPU)
    link_bw = AB.BW * 1000.0

    pl = {"C0": greedy_crp(dem, CRPConfig(relay_capacity=0)),
          "C2": force_no_dedup(greedy_crp(dem, CRPConfig(relay_capacity=cap))),
          "C3n": greedy_crp(dem, CRPConfig(relay_capacity=cap)),
          "C3c": greedy_crp_cost_aware(dem, CRPConfigCostAware(
              relay_capacity=cap, dpu_bytes_per_us=AB.DPU,
              link_bw_bytes_per_us=link_bw))}
    for s in C1_SEEDS:
        pl["C1_s%d" % s] = random_crp(dem, CRPConfig(relay_capacity=cap), s)

    l1, cyc = {}, {}
    for name, p in pl.items():
        l1[name] = float(AB.max_link_load(p))
        c = AB.emit_and_run("fc%s_%s" % (tag, name.lower()), p, wcfg, wl)
        if c is None:
            return None
        cyc[name] = float(c)

    c1k = ["C1_s%d" % s for s in C1_SEEDS]
    l1["C1"] = sum(l1[k] for k in c1k) / len(c1k)
    cyc["C1"] = sum(cyc[k] for k in c1k) / len(c1k)
    return l1, cyc


def main():
    wl_dir = os.path.join(
        AB.BASE,
        "workload/microbenchmarks/all_to_all/scale_%dnpus" % AB.NUM_GPUS)
    wl = ("examples/workload/microbenchmarks/all_to_all/scale_%dnpus/carts_wl"
          % AB.NUM_GPUS)
    write_2rank_workload(wl_dir, 1048576, num_ranks=AB.NUM_GPUS)

    print("=" * 96)
    print("FIXED-CAP ABLATION  gpd=%d  %d GPUs  bw=%d GB/s  dpu=%d B/us"
          % (AB.GPD, AB.NUM_GPUS, AB.BW, AB.DPU))
    print("caps fixed in advance: %s" % FIXED_CAPS)
    print("=" * 96)

    print("\nfor reference, the per-seed caps pick_cap would have chosen:")
    picked = {}
    for seed in SEEDS:
        d = generate_demand(MoEConfig(skew=AB.SKEW, seed=seed,
                                      num_gpus=AB.NUM_GPUS,
                                      gpus_per_domain=AB.GPD,
                                      num_experts=AB.NUM_GPUS))
        picked[seed], _ = pick_cap(d)
    print("  " + "  ".join("s%d=%d" % (s, picked[s]) for s in SEEDS))

    rows, store = [], {}
    for ci, cap in enumerate(FIXED_CAPS):
        print("\n" + "-" * 96)
        print("CAP = %d" % cap)
        print("-" * 96)
        for seed in SEEDS:
            res = run_seed_fixed(seed, wl, cap, "%d" % ci)
            if res is None:
                print("RUN FAILED at cap=%d seed=%d -- nothing written"
                      % (cap, seed))
                return 1
            l1, cyc = res
            store[(cap, seed)] = (l1, cyc)
            print("  seed %d | " % seed + "  ".join(
                "%s L2=%+6.2f%%" % (c, AB.pct(cyc["C0"], cyc[c]))
                for c in ("C1", "C3n", "C3c")))
            for c in CFGS:
                rows.append([cap, seed, c, round(l1[c], 1),
                             round(AB.pct(l1["C0"], l1[c]), 3),
                             round(cyc[c], 1),

                             round(AB.pct(cyc["C0"], cyc[c]), 3)])

    print("\n" + "=" * 96)
    print("AGGREGATE: mean cycles gain vs C0, per fixed cap")
    print("=" * 96)
    print("%10s | %9s %9s %9s | %9s %9s" %
          ("cap", "C1", "C3n", "C3c", "C3c min", "C3c>=C0"))
    print("-" * 96)
    for cap in FIXED_CAPS:
        m = {}
        for c in ("C1", "C3n", "C3c"):
            v = [AB.pct(store[(cap, s)][1]["C0"], store[(cap, s)][1][c])
                 for s in SEEDS]
            m[c] = v
        nn = sum(1 for v in m["C3c"] if v >= -1e-9)
        print("%10d | %8.2f%% %8.2f%% %8.2f%% | %8.2f%% %6d/%d"
              % (cap,
                 sum(m["C1"]) / len(SEEDS),
                 sum(m["C3n"]) / len(SEEDS),
                 sum(m["C3c"]) / len(SEEDS),
                 min(m["C3c"]), nn, len(SEEDS)))
    print("-" * 96)

    out = os.path.join(_HERE, "ablation_fixedcap_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cap", "seed", "config", "layer1_max_load",
                    "layer1_pct_vs_c0", "layer2_cycles", "layer2_pct_vs_c0"])
        w.writerows(rows)
    print("\nwrote %s  (%d rows)" % (out, len(rows)))
    return 0



if __name__ == "__main__":
    sys.exit(main())
