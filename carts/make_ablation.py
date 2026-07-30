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
from greedy_crp import CRPConfig, greedy_crp, Placement, CRPConfigCostAware, greedy_crp_cost_aware, greedy_crp_cost_aware_v2
from carts_et_writer import ETBuilder, WriterConfig, dpu_duration_micros
from make_2rank_workload import write_2rank_workload
CAP = 237
C1_SEEDS = range(5)
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
def main():
    wcfg = WriterConfig(dpu_bytes_per_us=20500)  # low-DPU regime, see S9a -- needed for Fix B's effect to be visible (S5: ties at placeholder 100,000)
    dem = generate_demand(MoEConfig(skew=1.5, seed=0))
    wl_dir = os.path.join(BASE, "workload/microbenchmarks/all_to_all/carts_16npus")
    write_2rank_workload(wl_dir, 1048576, num_ranks=16)
    placements = {
        "carts16_c0": greedy_crp(dem, CRPConfig(relay_capacity=0)),
        "carts16_c2": force_no_dedup(greedy_crp(dem, CRPConfig(relay_capacity=CAP))),
        "carts16_c3": greedy_crp(dem, CRPConfig(relay_capacity=CAP)),
        "carts16_c3_fixb": greedy_crp_cost_aware(dem, CRPConfigCostAware(
            relay_capacity=CAP, dpu_bytes_per_us=20500, link_bw_bytes_per_us=5000.0)),
    }
    for s in C1_SEEDS:
        placements[f"carts16_c1_s{s}"] = random_crp(dem, CRPConfig(relay_capacity=CAP), s)
    for name, pl in placements.items():
        emit_ets(os.path.join(SYSDIR, name), name, pl, wcfg)
        write_system_json(name)
    print(f"generated {len(placements)} configs; running binary...\n")
    res = {name: run_cycles(name) for name in placements}
    c0, c2, c3 = res["carts16_c0"], res["carts16_c2"], res["carts16_c3"]
    c3_fixb = res["carts16_c3_fixb"]
    c1v = [res[f"carts16_c1_s{s}"] for s in C1_SEEDS]
    c1mean = sum(c1v) / len(c1v)
    def pct(x):
        return 100.0 * (c0 - x) / c0
    print("=" * 60)
    print(f"ABLATION  (Switch_16npus_slow.yml, cap={CAP}, dpu_bytes_per_us=20500)")
    print("=" * 60)
    print(f"  C0 baseline         : {c0:>9} cycles")
    print(f"  C2 place, no dedup  : {c2:>9} cycles   ({pct(c2):+.2f}% vs C0)")
    print(f"  C1 random + dedup   : {c1mean:>9.0f} cycles   ({pct(c1mean):+.2f}% vs C0)  range {min(c1v)}-{max(c1v)}")
    print(f"  C3 full CARTS (OLD) : {c3:>9} cycles   ({pct(c3):+.2f}% vs C0)")
    print(f"  C3 full CARTS (FixB): {c3_fixb:>9} cycles   ({pct(c3_fixb):+.2f}% vs C0)")
    print("-" * 60)
    print(f"  C3 vs C1 (placement value, OLD) : {100.0*(c1mean-c3)/c1mean:+.2f}%")
    print(f"  C3 vs C2 (dedup value, OLD)     : {100.0*(c2-c3)/c2:+.2f}%")
    print(f"  C3 vs C1 (placement value, FixB): {100.0*(c1mean-c3_fixb)/c1mean:+.2f}%")
    print(f"  C3 vs C2 (dedup value, FixB)    : {100.0*(c2-c3_fixb)/c2:+.2f}%")
def force_no_dedup(placement):
    """Returns a NEW Placement (does not mutate the input) with every pair
    forced to served=False and load=raw (lam), discarding any dedup that
    the underlying placement would have applied. Used to construct a
    correct C2 (CARTS-optimized placement, dedup OFF) from a placement
    solved at the real cap -- fixes the original C2 bug where
    relay_capacity=0 made placement itself trivial (identical to C0),
    rather than testing placement-with-dedup-disabled specifically.
    Appended per code-discipline: does not modify Placement, greedy_crp(),
    or emit_ets()."""
    dem = placement.demand
    pairs = placement.inter_pairs()
    new_served = {p: False for p in pairs}
    new_load = {p: dem.lam(*p) for p in pairs}
    return Placement(demand=placement.demand, crp=placement.crp,
                      served=new_served, load=new_load,
                      x=placement.x, cap_left=placement.cap_left)


if __name__ == "__main__":
    main()


