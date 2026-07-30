"""
gpd4_gated_check.py -- cycles-level confirmation that the Bug-2 gate
(greedy_crp_cost_aware_gated) is a true no-op at the 4-domain headline
config (gpd=4, cap=237, seed=0), where the load-level check (validate_gate.py)
showed maxload exactly unchanged (252->252). This is the critical safety
check before trusting that S6/S18's headline numbers are gate-safe --
the gpd=2 case showed load-level-unchanged does NOT guarantee
cycles-level-unchanged (per-rank scheduling effects the load proxy misses),
so this must be verified directly, not assumed.

Reuses sweep_any_n.py's validated harness unchanged.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(os.path.dirname(_HERE)) if os.path.basename(_HERE) == "carts" else None

from demand_extractor import MoEConfig, generate_demand
from greedy_crp import (CRPConfig, greedy_crp, CRPConfigCostAware,
                         greedy_crp_cost_aware, greedy_crp_cost_aware_gated)
from carts_et_writer import WriterConfig
from sweep_any_n import (run_variant, BASE, NUM_GPUS, DPU_SWEEP,
                          LINK_BW_BYTES_PER_US)
from make_2rank_workload import write_2rank_workload

GPD = 4
CAP = 237
SEED = 0
SKEW = 1.5
DBPU = DPU_SWEEP[0]  # 20500, S5/S6 headline DPU-cost point


def main():
    wl_dir = os.path.join(BASE, "workload/microbenchmarks/all_to_all/carts_16npus")
    write_2rank_workload(wl_dir, 1048576, num_ranks=NUM_GPUS)

    dem = generate_demand(MoEConfig(skew=SKEW, seed=SEED, num_gpus=NUM_GPUS,
                                    gpus_per_domain=GPD))
    print(f"gpus_per_domain={GPD} ({NUM_GPUS // GPD} domains), seed={SEED}, "
          f"skew={SKEW}, cap={CAP} (S6/S18-validated headline operating point)")

    c0_placement = greedy_crp(dem, CRPConfig(relay_capacity=0))
    wcfg0 = WriterConfig(dpu_bytes_per_us=DBPU)
    c0 = run_variant("gpd4_gate_c0", c0_placement, wcfg0)

    crp = CRPConfigCostAware(relay_capacity=CAP, dpu_bytes_per_us=DBPU,
                             link_bw_bytes_per_us=LINK_BW_BYTES_PER_US)
    wcfg = WriterConfig(dpu_bytes_per_us=DBPU)

    p_old = greedy_crp_cost_aware(dem, crp)
    c_old = run_variant("gpd4_gate_old", p_old, wcfg)

    p_gated = greedy_crp_cost_aware_gated(dem, crp)
    c_gated = run_variant("gpd4_gate_new", p_gated, wcfg)

    print(f"\n{'config':>14} | {'cycles':>8}")
    print("-" * 27)
    for name, c in [("C0", c0), ("OLD (ungated)", c_old), ("NEW (gated)", c_gated)]:
        print(f"{name:>14} | {c}")

    if None not in (c0, c_old, c_gated):
        old_vs_c0 = 100.0 * (c0 - c_old) / c0
        new_vs_c0 = 100.0 * (c0 - c_gated) / c0
        gate_delta = 100.0 * (c_old - c_gated) / c_old
        print(f"\nOLD vs C0        : {old_vs_c0:+.2f}%")
        print(f"NEW(gated) vs C0 : {new_vs_c0:+.2f}%")
        print(f"Gate delta       : {gate_delta:+.2f}%  (0.00% = true no-op, as load-level predicted)")
        print(f"\n(load-level showed maxload exactly unchanged 252->252 at this "
              f"config; this is the cycles-level confirmation at dpu_bytes_per_us={DBPU})")
    else:
        print("\n[WARN] one or more runs failed to parse cycles -- see stderr above")


if __name__ == "__main__":
    main()
