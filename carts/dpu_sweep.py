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
from greedy_crp import CRPConfig, greedy_crp, Placement
from carts_et_writer import ETBuilder, WriterConfig, dpu_duration_micros
from make_2rank_workload import write_2rank_workload

CAP = 237
C1_SEEDS = range(5)
DPU_SWEEP = [15000, 20500, 50000, 83900, 100000]
BASE = "/workspace/astra-sim/examples"
SYSDIR = os.path.join(BASE, "system/custom_collectives")
BIN = "./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
WL = "examples/workload/microbenchmarks/all_to_all/carts_16npus/carts_wl"
NET = "examples/network/analytical/Switch_16npus_slow.yml"
MEM = "examples/remote_memory/analytical/no_memory_expansion.json"
def random_crp(demand, crp, seed):
    n = demand.cfg.num_domains
    cap_left = {d: crp.relay_capacity for d in range(n)}
    x = {d: 0 for d in range(n)}
    served, load = {}, {}
    inter = [(i, j) for i in range(n) for j in range(n) if i != j]
    _random.Random(seed).shuffle(inter)
    for (di, dj) in inter:
        raw = demand.lam(di, dj)
        u = demand.unique(di, dj)
        if raw == 0:
            served[(di, dj)] = False
            load[(di, dj)] = 0
            continue
        r = di
        if cap_left[r] >= u:
            cap_left[r] -= u
            x[r] = 1
            served[(di, dj)] = True
            load[(di, dj)] = u
        else:
            served[(di, dj)] = False
            load[(di, dj)] = raw
    return Placement(demand, crp, served, load, x, cap_left)
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
def run_one_dpu_value(dem, dbpu, tag):
    wcfg = WriterConfig(dpu_bytes_per_us=dbpu)
    placements = {
        f"sw{tag}_c0": greedy_crp(dem, CRPConfig(relay_capacity=0)),
        f"sw{tag}_c2": greedy_crp(dem, CRPConfig(relay_capacity=0)),
        f"sw{tag}_c3": greedy_crp(dem, CRPConfig(relay_capacity=CAP)),
    }
    for s in C1_SEEDS:
        placements[f"sw{tag}_c1_s{s}"] = random_crp(dem, CRPConfig(relay_capacity=CAP), s)
    for name, pl in placements.items():
        emit_ets(os.path.join(SYSDIR, name), name, pl, wcfg)
        write_system_json(name)
    res = {name: run_cycles(name) for name in placements}
    c0, c2, c3 = res[f"sw{tag}_c0"], res[f"sw{tag}_c2"], res[f"sw{tag}_c3"]
    c1v = [res[f"sw{tag}_c1_s{s}"] for s in C1_SEEDS]
    if any(v is None for v in [c0, c2, c3] + c1v):
        return None
    c1mean = sum(c1v) / len(c1v)
    return {"c0": c0, "c2": c2, "c3": c3, "c1mean": c1mean,
            "c1lo": min(c1v), "c1hi": max(c1v),
            "c3_vs_c1_pct": 100.0 * (c1mean - c3) / c1mean,
            "c3_vs_c2_pct": 100.0 * (c2 - c3) / c2}
def main():
    dem = generate_demand(MoEConfig(skew=1.5, seed=0))
    wl_dir = os.path.join(BASE, "workload/microbenchmarks/all_to_all/carts_16npus")
    write_2rank_workload(wl_dir, 1048576, num_ranks=16)
    print("=" * 78)
    print("DPU COST SENSITIVITY SWEEP  (cap=237, Switch_16npus_slow.yml)")
    print("=" * 78)
    print(f"{'dpu_bytes_per_us':>16} | {'GB/s':>6} | {'C3 vs C1':>10} | {'C3 vs C2':>10} | sign")
    print("-" * 78)
    for i, dbpu in enumerate(DPU_SWEEP):
        r = run_one_dpu_value(dem, dbpu, i)
        if r is None:
            print(f"{dbpu:>16,} | {dbpu/1000:>6.1f} | FAILED (see stderr)")
            continue
        sign = "C3 WINS" if r["c3_vs_c1_pct"] > 0 else "C1 wins (flip)"
        print(f"{dbpu:>16,} | {dbpu/1000:>6.1f} | {r['c3_vs_c1_pct']:>+9.2f}% | "
              f"{r['c3_vs_c2_pct']:>+9.2f}% | {sign}")
    print("-" * 78)
    print("Reading: 'C3 WINS' means CARTS placement beats random once coupled with dedup --")
    print("the coupling claim holds at that DPU cost. 'flip' means the placeholder-era result")
    print("persists at that throughput. Report the crossover point honestly either way.")

if __name__ == "__main__":
    main()
