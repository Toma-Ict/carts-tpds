"""
dual_metric_table.py -- builds the CARTS_v0.8 Section 6.3 "dual-metric"
causal-evidence table: Layer-1 (network-level max inter-domain link load)
and Layer-2 (system-level completion time) across the same C0/C1/C2/C3
configs, in the same order, per the v0.8 evaluation plan.

Layer-2 numbers are the existing locked S18 headline (already run, not
re-simulated here). Layer-1 numbers are NEW: extracted directly from the
same Placement objects make_ablation.py constructs (same dem, same seed,
same cap), via Placement.load -- no new simulation, pure Python.

Reuses make_ablation.py's exact placement-construction code (same dem,
same CRPConfig/CRPConfigCostAware, same force_no_dedup), so the Layer-1
demand is guaranteed identical to what produced the S18 Layer-2 numbers.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from demand_extractor import MoEConfig, generate_demand
from greedy_crp import (CRPConfig, greedy_crp, Placement,
                         CRPConfigCostAware, greedy_crp_cost_aware)
from make_ablation import random_crp, force_no_dedup, CAP, C1_SEEDS

# Locked Layer-2 numbers from S18 (already-validated cycles-level ablation,
# Switch_16npus_slow.yml, cap=237, dpu_bytes_per_us=20500, seed=0).
LAYER2_CYCLES = {
    "C0": 392004,
    "C2": 392004,
    "C1": 353400,   # 5-seed average, per S18
    "C3": 362250,    # FixB (validated production Fix B, per S21 decision)
}


def max_link_load(placement):
    pairs = placement.inter_pairs()
    return max(placement.load[p] for p in pairs)


def main():
    dem = generate_demand(MoEConfig(skew=1.5, seed=0))

    c0 = greedy_crp(dem, CRPConfig(relay_capacity=0))
    c2 = force_no_dedup(greedy_crp(dem, CRPConfig(relay_capacity=CAP)))
    c3 = greedy_crp(dem, CRPConfig(relay_capacity=CAP))
    c1_loads = [max_link_load(random_crp(dem, CRPConfig(relay_capacity=CAP), s))
                for s in C1_SEEDS]
    c1_mean_load = sum(c1_loads) / len(c1_loads)

    layer1 = {
        "C0": max_link_load(c0),
        "C2": max_link_load(c2),
        "C1": c1_mean_load,
        "C3": max_link_load(c3),
    }

    print("DUAL-METRIC CAUSAL EVIDENCE TABLE (CARTS_v0.8 Section 6.3)")
    print("Switch_16npus_slow.yml, cap=237, dpu_bytes_per_us=20500, seed=0\n")
    print(f"{'config':>8} | {'Layer-1':>12} | {'Layer-2':>12} | {'L1 vs C0':>9} | {'L2 vs C0':>9}")
    print(f"{'':>8} | {'max load':>12} | {'cycles':>12} | {'':>9} | {'':>9}")
    print("-" * 64)
    for cfg in ["C0", "C2", "C1", "C3"]:
        l1, l2 = layer1[cfg], LAYER2_CYCLES[cfg]
        l1pct = 100.0 * (layer1["C0"] - l1) / layer1["C0"]
        l2pct = 100.0 * (LAYER2_CYCLES["C0"] - l2) / LAYER2_CYCLES["C0"]
        l1_str = f"{l1:.1f}" if isinstance(l1, float) else str(l1)
        print(f"{cfg:>8} | {l1_str:>12} | {l2:>12} | {l1pct:>+8.2f}% | {l2pct:>+8.2f}%")

    print("\nOrder check (best to worst, lower = better, C0 excluded as baseline):")
    l1_order = sorted(["C1", "C2", "C3"], key=lambda c: layer1[c])
    l2_order = sorted(["C1", "C2", "C3"], key=lambda c: LAYER2_CYCLES[c])
    print(f"  Layer-1 order: {' < '.join(l1_order)}")
    print(f"  Layer-2 order: {' < '.join(l2_order)}")
    print(f"  Orders match: {l1_order == l2_order}")


if __name__ == "__main__":
    main()
