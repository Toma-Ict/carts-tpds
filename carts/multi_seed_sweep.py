import os
import sys
import json
import re
import subprocess
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from demand_extractor import MoEConfig, generate_demand
from greedy_crp import (CRPConfig, greedy_crp,
                         CRPConfigCostAware, greedy_crp_cost_aware)
from carts_et_writer import ETBuilder, WriterConfig, dpu_duration_micros
from make_2rank_workload import write_2rank_workload

CAP = 237
SEEDS = [0, 1, 2, 3, 4, 5]
DPU_LOW = 20500
DPU_HIGH = 100000
LINK_BW_BYTES_PER_US = 5000.0
BASE = "/workspace/astra-sim/examples"
SYSDIR = os.path.join(BASE, "system/custom_collectives")
BIN = "./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
WL = "examples/workload/microbenchmarks/all_to_all/carts_16npus/carts_wl"
NET = "examples/network/analytical/Switch_16npus_slow.yml"
MEM = "examples/remote_memory/analytical/no_memory_expansion.json"
def emit_ets(out_dir, prefix, placement, wcfg):
    cfg = placement.demand.cfg
    D, G, N = cfg.num_domains, cfg.gpus_per_domain, cfg.num_gpus
    gw = {d: d * G for d in range(D)}
    builders = {r: ETBuilder() for r in range(N)}
    pairs = [(i, j) for i in range(D) for j in range(D) if i != j]
    tag = {p: k for k, p in enumerate(pairs)}
    for (di, dj) in pairs:
        load = placement.load[(di, dj)]
        size = load * wcfg.token_bytes
        t, gsrc, gdst = tag[(di, dj)], gw[di], gw[dj]
        bs, bd = builders[gsrc], builders[gdst]
        if placement.served[(di, dj)]:
            c = bs.comp(dpu_duration_micros(size, wcfg))
            bs.send(size, comm_dst=gdst, tag=t, dep=c)
        else:
            bs.send(size, comm_dst=gdst, tag=t)
        bd.recv(size, comm_src=gsrc, tag=t)
    os.makedirs(out_dir, exist_ok=True)
    for r in range(N):
        b = builders[r]
        if len(b._nodes) == 0:
            b.comp(1)
        b.write(os.path.join(out_dir, f"{prefix}.{r}.et"))

def write_system_json(name):
    prefix = f"examples/system/custom_collectives/{name}/{name}"
    j = {"scheduling-policy": "FIFO", "preferred-dataset-splits": 1,
         "all-to-all-implementation-custom": [prefix], "local-mem-bw": 50}
    with open(os.path.join(SYSDIR, name + ".json"), "w") as f:
        json.dump(j, f, indent=2)
def run_cycles(name):
    sysjson = f"examples/system/custom_collectives/{name}.json"
    p = subprocess.run([BIN, f"--workload-configuration={WL}",
                        f"--system-configuration={sysjson}",
                        f"--network-configuration={NET}",
                        f"--remote-memory-configuration={MEM}"],
                       capture_output=True, text=True)
    nums = [int(m.split()[0]) for m in re.findall(r"\d+ cycles", p.stdout)]
    if not nums:
        sys.stderr.write(f"[{name}] no cycles parsed. tail:\n{p.stdout[-300:]}\n")
        return None
    return max(nums)

def run_one(name, placement, wcfg):
    emit_ets(os.path.join(SYSDIR, name), name, placement, wcfg)
    write_system_json(name)
    return run_cycles(name)
def main():
    wl_dir = os.path.join(BASE, "workload/microbenchmarks/all_to_all/carts_16npus")
    write_2rank_workload(wl_dir, 1048576, num_ranks=16)
    print("=" * 100)
    print("MULTI-SEED ROBUSTNESS CHECK (4 domains, cap=237)")
    print("=" * 100)
    print(f"{'seed':>5} | {'C0':>8} | {'C3@high(100k)':>14} | {'gain%':>7} | "
          f"{'C3@low(20.5k)':>14} | {'FixB gain%':>11}")
    print("-" * 100)
    for seed in SEEDS:
        dem = generate_demand(MoEConfig(skew=1.5, seed=seed))
        p_c0 = greedy_crp(dem, CRPConfig(relay_capacity=0))
        p_c3_high = greedy_crp(dem, CRPConfig(relay_capacity=CAP))
        wcfg_high = WriterConfig(dpu_bytes_per_us=DPU_HIGH)
        p_c3_low_costaware = greedy_crp_cost_aware(dem, CRPConfigCostAware(
            relay_capacity=CAP, dpu_bytes_per_us=DPU_LOW, link_bw_bytes_per_us=LINK_BW_BYTES_PER_US))
        p_c3_low_old = greedy_crp(dem, CRPConfig(relay_capacity=CAP))
        wcfg_low = WriterConfig(dpu_bytes_per_us=DPU_LOW)

        c0 = run_one(f"ms{seed}_c0", p_c0, wcfg_high)
        c3_high = run_one(f"ms{seed}_c3high", p_c3_high, wcfg_high)
        c3_low_old = run_one(f"ms{seed}_c3low_old", p_c3_low_old, wcfg_low)
        c3_low_new = run_one(f"ms{seed}_c3low_new", p_c3_low_costaware, wcfg_low)

        if any(v is None for v in [c0, c3_high, c3_low_old, c3_low_new]):
            print(f"{seed:>5} | FAILED (see stderr)")
            continue

        gain_pct = 100.0 * (c0 - c3_high) / c0
        fixb_pct = 100.0 * (c3_low_old - c3_low_new) / c3_low_old
        print(f"{seed:>5} | {c0:>8} | {c3_high:>14} | {gain_pct:>+6.2f}% | "
              f"{c3_low_new:>14} | {fixb_pct:>+10.2f}%")
    print("-" * 100)
    print("Reading: 'gain%' = baseline CARTS gain (C0 vs C3) at the placeholder DPU cost.")
    print("'FixB gain%' = cost-aware vs link-only placement at the low-DPU regime.")

if __name__ == "__main__":
    main()
