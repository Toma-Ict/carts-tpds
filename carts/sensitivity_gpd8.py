"""
sensitivity_gpd8.py -- Lock the S24 headline (gpd=8, num_gpus=32, ~+10% mean,
+17.49% seed0) by showing it is robust across the bandwidth-limited regime,
not a single network-config / single DPU-cost artifact.

Two axes, at the S24 sweet spot (gpd=8, num_gpus=32 = 4 domains, seeds 0-5):
  - link bandwidth: 5 / 10 / 20 / 50 GB/s. Per S2a the gain is bandwidth-
    limited -> expected strong at low bw, fading as links get fast. This
    bounds the regime and answers "is it just 5 GB/s?".
  - DPU cost: 20500 / 50000 / 83900 (the S9 estimated BF3 range, low->high).

Coupling rule (critical, S-this-session): the network yaml's bandwidth and
CRPConfigCostAware.link_bw_bytes_per_us MUST move together (5.0 GB/s ==
5000 bytes/us), else Fix B's net-benefit ordering judges the wrong regime.
pick_cap is pure load-level (no bw/dpu) -> cap picked once per seed, reused
across the whole bw x dpu grid (clean, comparable).

Append-only: does NOT modify scale_multiseed.py or any validated file.
Run from repo root via os.chdir.
"""
import os, sys, time, csv

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(os.path.dirname(_HERE))   # -> /workspace/astra-sim

from demand_extractor import MoEConfig, generate_demand
from greedy_crp import (CRPConfig, greedy_crp,
                        CRPConfigCostAware, greedy_crp_cost_aware)
from carts_et_writer import WriterConfig
from make_2rank_workload import write_2rank_workload
from sweep_any_n import emit_ets, write_system_json, pick_cap
from scale_multiseed import run_cycles_n   # parametrized binary runner (wl/net args)

GPD = 8
NUM_GPUS = 32          # 4 domains -- the S24 sweet spot
SEEDS = [0, 1, 2, 3, 4, 5]
SKEW = 1.5
BW_LIST = [5, 10, 20, 50]            # GB/s ; 5 == existing _slow yaml
DPU_LIST = [20500, 50000, 83900]     # S9 estimated BF3 range

# bandwidth (GB/s) -> topology yaml (all created from Switch_32npus_slow.yml)
BW_YAML = {5:  "examples/network/analytical/Switch_32npus_slow.yml",
           10: "examples/network/analytical/Switch_32npus_bw10.yml",
           20: "examples/network/analytical/Switch_32npus_bw20.yml",
           50: "examples/network/analytical/Switch_32npus_bw50.yml"}

# anchor: bw=5, dpu=20500, seed0 must reproduce S24 gpd=8/32 seed0 (+17.49%, cap=541)
ANCHOR = dict(bw=5, dpu=20500, seed=0, new_vs_c0=17.49, cap=541)

BASE = "/workspace/astra-sim/examples"


def precompute_caps():
    """pick_cap is pure load-level (no bw/dpu) -> one cap per seed, reused
    across the entire bw x dpu grid. Computed once."""
    caps = {}
    for seed in SEEDS:
        dem = generate_demand(MoEConfig(skew=SKEW, seed=seed, num_gpus=NUM_GPUS,
                                        gpus_per_domain=GPD, num_experts=NUM_GPUS))
        cap, _ = pick_cap(dem)
        caps[seed] = cap
    return caps


def run_cell(seed, cap, bw, dpu, wl):
    """C0/OLD/NEW cycles at one (bw, dpu) point. link_bw_bytes_per_us is
    SYNCED to the yaml bandwidth (bw GB/s == bw*1000 bytes/us) so Fix B's
    net-benefit ordering judges the SAME regime the simulator runs. C0
    (cap=0, no served pairs, no DPU COMP) is dpu-independent by construction
    -- a built-in sanity check (C0 must be identical across dpu at fixed bw)."""
    net = BW_YAML[bw]
    link_bw = bw * 1000.0
    dem = generate_demand(MoEConfig(skew=SKEW, seed=seed, num_gpus=NUM_GPUS,
                                    gpus_per_domain=GPD, num_experts=NUM_GPUS))
    wcfg = WriterConfig(dpu_bytes_per_us=dpu)

    p_c0 = greedy_crp(dem, CRPConfig(relay_capacity=0))
    p_old = greedy_crp(dem, CRPConfig(relay_capacity=cap))
    p_new = greedy_crp_cost_aware(dem, CRPConfigCostAware(
        relay_capacity=cap, dpu_bytes_per_us=dpu, link_bw_bytes_per_us=link_bw))

    out = {}
    for tag, pl in [("c0", p_c0), ("old", p_old), ("new", p_new)]:
        name = f"sg8_{tag}"   # sequential emit->run, name reuse is safe
        emit_ets(os.path.join(BASE, "system/custom_collectives", name), name, pl, wcfg)
        write_system_json(name)
        out[tag] = run_cycles_n(name, wl, net)
    return out["c0"], out["old"], out["new"]


def main():
    wl_dir = os.path.join(BASE, f"workload/microbenchmarks/all_to_all/scale_{NUM_GPUS}npus")
    wl = f"examples/workload/microbenchmarks/all_to_all/scale_{NUM_GPUS}npus/carts_wl"
    write_2rank_workload(wl_dir, 1048576, num_ranks=NUM_GPUS)

    caps = precompute_caps()
    print("=" * 100)
    print(f"sensitivity sweep | gpd={GPD} num_gpus={NUM_GPUS} ({NUM_GPUS//GPD} dom) "
          f"| seeds={SEEDS} | bw={BW_LIST} GB/s | dpu={DPU_LIST}")
    print(f"per-seed caps (regime-independent): {caps}")
    print("=" * 100)
    print(f"{'bw':>4} | {'dpu':>7} | {'NEWvsC0 mean':>12} | {'min':>7} | {'max':>7} "
          f"| {'NEW>=C0':>8} | {'OLDvsC0 mean':>12}")
    print("-" * 100)

    rows = []
    anchor_seen = None
    for bw in BW_LIST:
        for dpu in DPU_LIST:
            nv, ov = [], []
            for seed in SEEDS:
                c0, old, new = run_cell(seed, caps[seed], bw, dpu, wl)
                if None in (c0, old, new):
                    print(f"{bw:>4} | {dpu:>7} | RUN FAILED seed={seed}"); continue
                nvc0 = 100.0 * (c0 - new) / c0
                ovc0 = 100.0 * (c0 - old) / c0
                nv.append(nvc0); ov.append(ovc0)
                rows.append([bw, dpu, seed, caps[seed], c0, old, new,
                             round(nvc0, 4), round(ovc0, 4)])
                if bw == ANCHOR["bw"] and dpu == ANCHOR["dpu"] and seed == ANCHOR["seed"]:
                    anchor_seen = nvc0
            if nv:
                ge0 = sum(1 for x in nv if x >= -1e-9)
                print(f"{bw:>4} | {dpu:>7} | {sum(nv)/len(nv):>+11.2f}% | {min(nv):>+6.2f}% "
                      f"| {max(nv):>+6.2f}% | {ge0:>6}/{len(nv)} | {sum(ov)/len(ov):>+11.2f}%")

    if anchor_seen is not None:
        ok = abs(anchor_seen - ANCHOR["new_vs_c0"]) <= 0.5
        print(f"\n[ANCHOR {'OK' if ok else 'WARN'}] bw=5 dpu=20500 seed0 NEWvsC0="
              f"{anchor_seen:+.2f}% vs S24 {ANCHOR['new_vs_c0']:+.2f}%")

    csv_path = os.path.join(_HERE, "sensitivity_gpd8_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bw_gbs", "dpu_bytes_per_us", "seed", "cap", "c0", "old", "new",
                    "new_vs_c0", "old_vs_c0"])
        w.writerows(rows)
    print(f"\n[saved] {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
