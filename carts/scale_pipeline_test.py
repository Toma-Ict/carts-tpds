"""
scale_pipeline_test.py -- Real CARTS-pipeline scale test (S13 follow-up,
Q1-feasibility investigation). Unlike sweep_any_n.py (num_gpus=16 FIXED,
gpus_per_domain varies), this script holds gpus_per_domain=4 FIXED and
varies num_gpus (32, 64, 128) -- i.e. scales num_domains while keeping the
per-domain ratio at the already-validated gpd=4 configuration.

Measures both Python-side pipeline time (demand generation + placement
solve) and ASTRA-sim binary runtime, separately -- the bottleneck could be
either stage at larger scale, not necessarily the simulator.

Reuses emit_ets/write_system_json from sweep_any_n.py as-is (confirmed
parameter-clean, no hardcoded N). Does NOT reuse sweep_any_n.py's
run_cycles (it hardcodes WL/NET to the 16-npu paths) -- reimplemented here
with wl/net as parameters instead.
"""
import os
import sys
import time
import json
import re
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from demand_extractor import MoEConfig, generate_demand
from greedy_crp import CRPConfig, greedy_crp, CRPConfigCostAware, greedy_crp_cost_aware
from carts_et_writer import WriterConfig
from make_2rank_workload import write_2rank_workload
from sweep_any_n import emit_ets, write_system_json, pick_cap

GPUS_PER_DOMAIN = 4
SCALE_LIST = [32, 64, 128]
SKEW = 1.5
SEED = 0
DPU_BYTES_PER_US = 20500  # low-DPU regime, matches FixB-gain convention elsewhere
LINK_BW_BYTES_PER_US = 5000.0

BASE = "/workspace/astra-sim/examples"
BIN = "./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
MEM = "examples/remote_memory/analytical/no_memory_expansion.json"


def run_cycles_n(name, wl, net):
    """Parametrized version of sweep_any_n.py's run_cycles -- takes wl/net
    as arguments instead of reading module-level globals, since this script
    needs a different workload/network path per num_gpus value."""
    sysjson = f"examples/system/custom_collectives/{name}.json"
    p = subprocess.run([BIN, f"--workload-configuration={wl}",
                        f"--system-configuration={sysjson}",
                        f"--network-configuration={net}",
                        f"--remote-memory-configuration={MEM}"],
                       capture_output=True, text=True, timeout=600)
    nums = [int(m.split()[0]) for m in re.findall(r"\d+ cycles", p.stdout)]
    if not nums:
        sys.stderr.write(f"[{name}] no cycles parsed. tail:\n{p.stdout[-300:]}\n")
        return None
    return max(nums)


def run_one_scale(num_gpus):
    net = f"examples/network/analytical/Switch_{num_gpus}npus_slow.yml"
    net_full = os.path.join(BASE, "..", net)
    if not os.path.exists(net_full):
        print(f"[n={num_gpus}] SKIP -- {net} does not exist, create it first")
        return None

    wl_dir = os.path.join(BASE, f"workload/microbenchmarks/all_to_all/scale_{num_gpus}npus")
    wl = f"examples/workload/microbenchmarks/all_to_all/scale_{num_gpus}npus/carts_wl"

    t_demand_0 = time.time()
    write_2rank_workload(wl_dir, 1048576, num_ranks=num_gpus)
    cfg = MoEConfig(skew=SKEW, seed=seed, num_gpus=num_gpus,
                     gpus_per_domain=GPUS_PER_DOMAIN, num_experts=num_gpus)
    dem = generate_demand(cfg)
    t_demand = time.time() - t_demand_0

    t_solve_0 = time.time()
    placement = greedy_crp_cost_aware(dem, CRPConfigCostAware(
        relay_capacity=10**9,  # cap=inf for this feasibility check -- not tuning the ablation here
        dpu_bytes_per_us=DPU_BYTES_PER_US, link_bw_bytes_per_us=LINK_BW_BYTES_PER_US))
    t_solve = time.time() - t_solve_0

    t_write_0 = time.time()
    wcfg = WriterConfig(dpu_bytes_per_us=DPU_BYTES_PER_US)
    name = f"scale_{num_gpus}npus"
    emit_ets(os.path.join(BASE, "system/custom_collectives", name), name, placement, wcfg)
    write_system_json(name)
    t_write = time.time() - t_write_0

    t_sim_0 = time.time()
    cycles = run_cycles_n(name, wl, net)
    t_sim = time.time() - t_sim_0

    return dict(num_gpus=num_gpus, num_domains=dem.cfg.num_domains,
                t_demand=t_demand, t_solve=t_solve, t_write=t_write,
                t_sim=t_sim, cycles=cycles)


def main_v1():
    print("=" * 100)
    print(f"CARTS pipeline scale test: gpus_per_domain={GPUS_PER_DOMAIN} (fixed), "
          f"num_gpus in {SCALE_LIST}, seed={SEED}")
    print("=" * 100)
    print(f"{'num_gpus':>9} | {'domains':>8} | {'t_demand':>9} | {'t_solve':>9} | "
          f"{'t_write':>9} | {'t_sim':>9} | {'cycles':>10}")
    print("-" * 100)

    rows = []
    for n in SCALE_LIST:
        r = run_one_scale(n)
        if r is None:
            continue
        rows.append(r)
        print(f"{r['num_gpus']:>9} | {r['num_domains']:>8} | {r['t_demand']:>8.3f}s | "
              f"{r['t_solve']:>8.3f}s | {r['t_write']:>8.3f}s | {r['t_sim']:>8.3f}s | "
              f"{str(r['cycles']):>10}")

    csv_path = os.path.join(_HERE, "scale_pipeline_results.csv")
    with open(csv_path, "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["num_gpus", "num_domains", "t_demand", "t_solve",
                    "t_write", "t_sim", "cycles"])
        for r in rows:
            w.writerow([r['num_gpus'], r['num_domains'], r['t_demand'],
                        r['t_solve'], r['t_write'], r['t_sim'], r['cycles']])
    print(f"\n[saved] {csv_path} ({len(rows)} rows)")


def run_one_scale_v2(num_gpus, seed=SEED):
    """C0-vs-OLD-vs-NEW gain%% comparison at a given num_gpus, gpus_per_domain
    fixed at GPUS_PER_DOMAIN (S13 follow-up, extending S7's domain-count
    curve to larger scale). Reuses pick_cap() from sweep_any_n.py (confirmed
    parameter-clean, no num_gpus=16 assumption)."""
    net = f"examples/network/analytical/Switch_{num_gpus}npus_slow.yml"
    net_full = os.path.join(BASE, "..", net)
    if not os.path.exists(net_full):
        print(f"[n={num_gpus}] SKIP -- {net} does not exist, create it first")
        return None

    wl_dir = os.path.join(BASE, f"workload/microbenchmarks/all_to_all/scale_{num_gpus}npus")
    wl = f"examples/workload/microbenchmarks/all_to_all/scale_{num_gpus}npus/carts_wl"
    write_2rank_workload(wl_dir, 1048576, num_ranks=num_gpus)

    cfg = MoEConfig(skew=SKEW, seed=seed, num_gpus=num_gpus,
                     gpus_per_domain=GPUS_PER_DOMAIN, num_experts=num_gpus)
    dem = generate_demand(cfg)

    t_cap_0 = time.time()
    cap, cap_rows = pick_cap(dem)
    t_cap = time.time() - t_cap_0

    wcfg = WriterConfig(dpu_bytes_per_us=DPU_BYTES_PER_US)

    p_c0 = greedy_crp(dem, CRPConfig(relay_capacity=0))
    p_old = greedy_crp(dem, CRPConfig(relay_capacity=cap))
    p_new = greedy_crp_cost_aware(dem, CRPConfigCostAware(
        relay_capacity=cap, dpu_bytes_per_us=DPU_BYTES_PER_US,
        link_bw_bytes_per_us=LINK_BW_BYTES_PER_US))

    results = {}
    for tag, placement in [("c0", p_c0), ("old", p_old), ("new", p_new)]:
        name = f"scale_{num_gpus}npus_{tag}"
        emit_ets(os.path.join(BASE, "system/custom_collectives", name), name, placement, wcfg)
        write_system_json(name)
        results[tag] = run_cycles_n(name, wl, net)

    return dict(num_gpus=num_gpus, num_domains=dem.cfg.num_domains, cap=cap,
                t_cap=t_cap, c0=results["c0"], old=results["old"], new=results["new"])


def main():
    print("=" * 100)
    print(f"CARTS gain%% test: gpus_per_domain={GPUS_PER_DOMAIN} (fixed), "
          f"num_gpus in {SCALE_LIST}, seed={SEED}, dpu_bytes_per_us={DPU_BYTES_PER_US}")
    print("=" * 100)
    print(f"{'num_gpus':>9} | {'domains':>8} | {'cap':>6} | {'C0':>10} | "
          f"{'OLD':>10} | {'NEW':>10} | {'OLDvsC0':>9} | {'NEWvsC0':>9} | {'FixBdelta':>9}")
    print("-" * 100)

    rows = []
    for n in SCALE_LIST:
        r = run_one_scale_v2(n)
        if r is None:
            continue
        rows.append(r)
        c0, old, new = r['c0'], r['old'], r['new']
        if c0 is None or old is None or new is None:
            print(f"{r['num_gpus']:>9} | RUN FAILED")
            continue
        old_vs_c0 = 100.0 * (c0 - old) / c0
        new_vs_c0 = 100.0 * (c0 - new) / c0
        fixb = 100.0 * (old - new) / old if old != 0 else 0.0
        print(f"{r['num_gpus']:>9} | {r['num_domains']:>8} | {r['cap']:>6} | "
              f"{c0:>10} | {old:>10} | {new:>10} | {old_vs_c0:>+8.2f}% | "
              f"{new_vs_c0:>+8.2f}% | {fixb:>+8.2f}%")

    csv_path = os.path.join(_HERE, "scale_gain_results.csv")
    with open(csv_path, "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["num_gpus", "num_domains", "cap", "c0", "old", "new"])
        for r in rows:
            w.writerow([r['num_gpus'], r['num_domains'], r['cap'],
                        r['c0'], r['old'], r['new']])
    print(f"\n[saved] {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()


