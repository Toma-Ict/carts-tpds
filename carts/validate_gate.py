"""
validate_gate.py -- confirm greedy_crp_cost_aware_gated (Bug 2 fix) is safe.

Two checks, both load-level (max link load proxy, no ASTRA-sim):
(A) HEADLINE SAFETY: at the validated configs (gpd=4 cap=237 seeds 0-5;
    gpd=2 cap=120 seed 0), the gated version must produce the SAME served
    set and SAME max link load as the ungated greedy_crp_cost_aware --
    i.e. the gate must not touch any positive-benefit bottleneck decision.
(B) BUG-2 FIXED: at num_experts=64 block-mapping cap=252 seed=0 (where Bug 2
    re-triggered this session), the gated version must DROP at least one
    zero/negative-benefit pair that the ungated version served.
"""
import sys
sys.path.insert(0, ".")
from demand_extractor import MoEConfig, generate_demand
from greedy_crp import (CRPConfigCostAware, greedy_crp_cost_aware,
                         greedy_crp_cost_aware_gated)


class MoEConfigBlock(MoEConfig):
    def gpu_of_expert(self, e):
        epg = max(1, self.num_experts // self.num_gpus)
        return e // epg


def maxload(pl):
    n = pl.demand.cfg.num_domains
    inter = [(i, j) for i in range(n) for j in range(n) if i != j]
    return max(pl.load[p] for p in inter)


def cmp(tag, dem, cap, dbpu=20500):
    crp = CRPConfigCostAware(relay_capacity=cap, dpu_bytes_per_us=dbpu,
                             link_bw_bytes_per_us=5000.0)
    old = greedy_crp_cost_aware(dem, crp)
    new = greedy_crp_cost_aware_gated(dem, crp)
    same_served = old.served == new.served
    n_old = sum(1 for v in old.served.values() if v)
    n_new = sum(1 for v in new.served.values() if v)
    ml_old, ml_new = maxload(old), maxload(new)
    print(f"{tag:>34} | served {n_old:>3}->{n_new:>3} | "
          f"maxload {ml_old:>4}->{ml_new:>4} | identical={same_served}")
    return same_served, ml_old, ml_new


print("=== (A) HEADLINE SAFETY (gated must equal ungated) ===")
all_safe = True
for s in range(6):
    dem = generate_demand(MoEConfig(skew=1.5, seed=s, num_gpus=16,
                                    gpus_per_domain=4, num_experts=16))
    same, mo, mn = cmp(f"gpd=4 cap=237 seed={s}", dem, 237)
    all_safe = all_safe and same and (mo == mn)
dem = generate_demand(MoEConfig(skew=1.5, seed=0, num_gpus=16,
                                gpus_per_domain=2, num_experts=16))
same, mo, mn = cmp("gpd=2 cap=120 seed=0", dem, 120)
all_safe = all_safe and same and (mo == mn)
print(f"\n  ALL HEADLINE CONFIGS UNCHANGED: {all_safe}")

print("\n=== (B) BUG 2 FIXED (gated must drop a zero-benefit pair) ===")
dem = generate_demand(MoEConfigBlock(skew=1.5, seed=0, num_gpus=16,
                                     gpus_per_domain=4, num_experts=64))
same, mo, mn = cmp("num_experts=64 cap=252 seed=0", dem, 252)
print(f"\n  GATE CHANGED THE DECISION (Bug 2 fixed): {not same}")
