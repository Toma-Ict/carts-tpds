"""
num_experts_cycles_check.py -- single-point cycles-level confirmation of the
num_experts load-level finding (num_experts_sweep.py).

Goal: the load-level sweep showed gain% stays flat (15-20% range) across
num_experts in {16,32,64,128,256} under block mapping. That result is
solver-side (max_link_load proxy), same caveat as S8. This script confirms
the SAME conclusion holds at the cycles level (real ASTRA-sim run, OLD vs
Fix B) at ONE representative point: num_experts=64 (4 experts/gpu, near the
mean of the load-level table), seed=0. Not a full table -- a spot-check.

Reuses sweep_any_n.py's validated harness (emit_ets/run_variant/pick_cap)
unchanged. num_gpus=16/gpus_per_domain=4 fixed -> reuses existing
carts_16npus workload + Switch_16npus_slow.yml network files, no new
topology needed. Only new piece: the MoEConfigBlock subclass (same one
validated in num_experts_sweep.py, with the same 252/218 anchor at ne=16).
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(os.path.dirname(_HERE))  # sweep_any_n.BIN is a relative path that
                                  # assumes CWD=/workspace/astra-sim, not carts/

from demand_extractor import MoEConfig, generate_demand
from greedy_crp import CRPConfig, greedy_crp, CRPConfigCostAware, greedy_crp_cost_aware
from carts_et_writer import WriterConfig
from sweep_any_n import (pick_cap, run_variant, write_system_json,
                          run_cycles, BASE, NUM_GPUS, DPU_SWEEP,
                          LINK_BW_BYTES_PER_US)
from make_2rank_workload import write_2rank_workload

SKEW = 1.5
NUM_EXPERTS = 64
SEED = 0
DBPU = DPU_SWEEP[0]  # 20500, the pessimistic/low-DPU regime (matches S5/S6 headline)


class MoEConfigBlock(MoEConfig):
    """Same block mapping validated in num_experts_sweep.py. Reduces to
    identity at num_experts==num_gpus (anchor: raw=252 dedup=218, seed=0)."""
    def gpu_of_expert(self, e):
        epg = max(1, self.num_experts // self.num_gpus)
        return e // epg


def main():
    wl_dir = os.path.join(BASE, "workload/microbenchmarks/all_to_all/carts_16npus")
    write_2rank_workload(wl_dir, 1048576, num_ranks=NUM_GPUS)

    cfg = MoEConfigBlock(skew=SKEW, seed=SEED, num_gpus=NUM_GPUS,
                          gpus_per_domain=4, num_experts=NUM_EXPERTS)
    dem = generate_demand(cfg)
    cap, cap_rows = pick_cap(dem)
    print(f"num_experts={NUM_EXPERTS} (experts/gpu={NUM_EXPERTS // NUM_GPUS}), "
          f"seed={SEED}, skew={SKEW}, auto-picked cap={cap}")

    c0_placement = greedy_crp(dem, CRPConfig(relay_capacity=0))
    wcfg0 = WriterConfig(dpu_bytes_per_us=DBPU)
    c0 = run_variant("ne64_c0", c0_placement, wcfg0)

    p_old = greedy_crp(dem, CRPConfig(relay_capacity=cap))
    wcfg = WriterConfig(dpu_bytes_per_us=DBPU)
    c_old = run_variant("ne64_old", p_old, wcfg)

    p_new = greedy_crp_cost_aware(dem, CRPConfigCostAware(
        relay_capacity=cap, dpu_bytes_per_us=DBPU,
        link_bw_bytes_per_us=LINK_BW_BYTES_PER_US))
    c_new = run_variant("ne64_new", p_new, wcfg)

    print(f"\n{'config':>10} | {'cycles':>8}")
    print("-" * 23)
    for name, c in [("C0", c0), ("OLD", c_old), ("NEW (FixB)", c_new)]:
        print(f"{name:>10} | {c}")

    if None not in (c0, c_old, c_new):
        old_vs_c0 = 100.0 * (c0 - c_old) / c0
        new_vs_c0 = 100.0 * (c0 - c_new) / c0
        fixb = 100.0 * (c_old - c_new) / c_old
        print(f"\nOLD vs C0   : {old_vs_c0:+.2f}%")
        print(f"NEW vs C0   : {new_vs_c0:+.2f}%")
        print(f"FixB delta  : {fixb:+.2f}%")
        print(f"\n(load-level table predicted ~17.45% mean potential gain at "
              f"num_experts=64; this is the cycles-level NEW-vs-C0 number, "
              f"net of DPU overhead at dpu_bytes_per_us={DBPU})")
    else:
        print("\n[WARN] one or more runs failed to parse cycles -- see stderr above")


if __name__ == "__main__":
    main()
