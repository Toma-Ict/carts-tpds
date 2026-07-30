import os
import sys
import json
import re
import subprocess
import random as _random
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from demand_extractor import MoEConfig, generate_demand
from greedy_crp import (CRPConfig, greedy_crp, Placement,
                         CRPConfigCostAware, greedy_crp_cost_aware)
from carts_et_writer import ETBuilder, WriterConfig, dpu_duration_micros
from make_2rank_workload import write_2rank_workload

CAP = 237
DPU_SWEEP = [20500, 50000, 83900, 100000]
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

def run_variant(dem, dbpu, tag, placement, wcfg):
    name = f"swv2_{tag}"
    emit_ets(os.path.join(SYSDIR, name), name, placement, wcfg)
    write_system_json(name)
    return run_cycles(name)
def main():
    dem = generate_demand(MoEConfig(skew=1.5, seed=0))
    wl_dir = os.path.join(BASE, "workload/microbenchmarks/all_to_all/carts_16npus")
    write_2rank_workload(wl_dir, 1048576, num_ranks=16)
    print("=" * 86)
    print("ALGORITHM A/B SWEEP: old greedy_crp (link-load only) vs new cost-aware greedy")
    print("=" * 86)
    print(f"{'dpu_bytes_per_us':>16} | {'OLD cycles':>10} | {'NEW cycles':>10} | {'delta':>8} | winner")
    print("-" * 86)
    for dbpu in DPU_SWEEP:
        wcfg = WriterConfig(dpu_bytes_per_us=dbpu)
        p_old = greedy_crp(dem, CRPConfig(relay_capacity=CAP))
        p_new = greedy_crp_cost_aware(dem, CRPConfigCostAware(
            relay_capacity=CAP, dpu_bytes_per_us=dbpu, link_bw_bytes_per_us=LINK_BW_BYTES_PER_US))
        c_old = run_variant(dem, dbpu, f"old_{dbpu}", p_old, wcfg)
        c_new = run_variant(dem, dbpu, f"new_{dbpu}", p_new, wcfg)
        if c_old is None or c_new is None:
            print(f"{dbpu:>16,} | FAILED")
            continue
        delta_pct = 100.0 * (c_old - c_new) / c_old
        winner = "NEW (cost-aware)" if c_new < c_old else ("OLD (link-only)" if c_new > c_old else "TIE")
        print(f"{dbpu:>16,} | {c_old:>10} | {c_new:>10} | {delta_pct:>+7.2f}% | {winner}")
    print("-" * 86)
    print("Reading: positive delta% means the cost-aware algorithm reduces cycles vs the")
    print("original link-load-only greedy, at that DPU throughput.")

if __name__ == "__main__":
    main()
