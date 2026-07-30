"""
sweep_any_n.py — Reusable any-N gpus_per_domain sweep (generalizes
gpd_test.py / gpd2_fixb_test.py, session D follow-up, CARTS_STATE.md S13 item 1).

For each gpus_per_domain in a list (num_gpus fixed at 16, per S7 isolation
methodology): auto-picks an operating-point relay_capacity via max C1-C3 gap
(capacity_sweep.py logic, generalized), then runs OLD (greedy_crp) vs NEW
(greedy_crp_cost_aware) across a DPU-cost sweep, on real ASTRA-sim.

NOTE (current scope): num_gpus is hardcoded to 16 because WL/NET paths below
point at carts_16npus / Switch_16npus_slow.yml specifically (see CARTS_STATE.md
S3, file inventory). Asserted explicitly - do not silently rerun at another
num_gpus without first creating the matching workload+network files.
"""
import os
import sys
import csv
import json
import re
import random
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from demand_extractor import MoEConfig, generate_demand
from greedy_crp import (CRPConfig, greedy_crp,
                         CRPConfigCostAware, greedy_crp_cost_aware)
from carts_et_writer import ETBuilder, WriterConfig, dpu_duration_micros
from make_2rank_workload import write_2rank_workload

NUM_GPUS = 16
LINK_BW_BYTES_PER_US = 5000.0
DPU_SWEEP = [20500, 50000, 83900, 100000]
SKEW = 1.5

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
        if load == 0:
            continue
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


def run_variant(name, placement, wcfg):
    emit_ets(os.path.join(SYSDIR, name), name, placement, wcfg)
    write_system_json(name)
    return run_cycles(name)


def max_link_load(demand, cap, order, dedup, seed=0):
    n = demand.cfg.num_domains
    cap_left = {d: cap for d in range(n)}
    load = {}
    inter = [(i, j) for i in range(n) for j in range(n) if i != j]
    if order == "greedy":
        inter.sort(key=lambda p: demand.lam(*p), reverse=True)
    elif order == "random":
        random.Random(seed).shuffle(inter)
    for (di, dj) in inter:
        raw = demand.lam(di, dj)
        if raw == 0:
            load[(di, dj)] = 0
            continue
        u = demand.unique(di, dj)
        if dedup and cap_left[di] >= u:
            cap_left[di] -= u
            load[(di, dj)] = u
        else:
            load[(di, dj)] = raw
    return max(load[p] for p in inter)


def c1_random(demand, cap, trials=50):
    vals = [max_link_load(demand, cap, "random", True, seed=s) for s in range(trials)]
    return sum(vals) / len(vals)


def pick_cap(demand):
    """Auto cap-pick: sweep candidate caps, return the one maximizing the
    C1(random)-C3(greedy+dedup) gap. Generalizes capacity_sweep.py's manual
    inspection into an automatic selection rule (session D follow-up, choice 1b).
    cap is seed/topology-specific (CARTS_STATE.md gotcha #10) -- always re-derive
    per gpus_per_domain, never reuse a cap across configs."""
    n = demand.cfg.num_domains
    inter = [(i, j) for i in range(n) for j in range(n) if i != j]
    out_unique = {d: sum(demand.unique(d, j) for j in range(n) if j != d)
                  for d in range(n)}
    cap_hi = max(out_unique.values())
    if cap_hi == 0:
        return 0, []
    caps = [0] + [int(cap_hi * f) for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    caps += [cap_hi]
    caps = sorted(set(caps))
    rows = []
    best_cap, best_gap = None, -1.0
    for cap in caps:
        c3 = max_link_load(demand, cap, "greedy", True)
        c1a = c1_random(demand, cap)
        gap = c1a - c3
        rows.append((cap, c1a, c3, gap))
        if gap > best_gap:
            best_gap, best_cap = gap, cap
    if best_gap <= 0:
        # No cap shows any C1-C3 separation at the load-accounting level
        # (e.g. gpus_per_domain=1, raw==unique everywhere -- dedup saves no
        # bytes at any cap). Default to cap=0 (no dedup attempted) rather than
        # cap_hi: cap_hi was tried and found HARMFUL here, because greedy_crp()
        # serves pairs whenever capacity allows, with no net-benefit check --
        # see CARTS_STATE.md S14b (Fix B serve-gate gap, discovered this session).
        best_cap = 0
    return best_cap, rows


def sweep_one_gpd(gpus_per_domain, seed=0):
    assert NUM_GPUS == 16, (
        f"NUM_GPUS={NUM_GPUS} but WL/NET paths are hardcoded to carts_16npus / "
        f"Switch_16npus_slow.yml. Create matching workload+network files before "
        f"changing NUM_GPUS, or this will silently run the wrong topology."
    )
    cfg = MoEConfig(skew=SKEW, seed=seed, num_gpus=NUM_GPUS,
                     gpus_per_domain=gpus_per_domain, num_experts=NUM_GPUS)
    dem = generate_demand(cfg)
    cap, cap_rows = pick_cap(dem)

    c0_placement = greedy_crp(dem, CRPConfig(relay_capacity=0))
    wcfg0 = WriterConfig(dpu_bytes_per_us=DPU_SWEEP[0])
    tag0 = f"sweepN_gpd{gpus_per_domain}_c0"
    c0 = run_variant(tag0, c0_placement, wcfg0)

    results = []
    for dbpu in DPU_SWEEP:
        wcfg = WriterConfig(dpu_bytes_per_us=dbpu)
        p_old = greedy_crp(dem, CRPConfig(relay_capacity=cap))
        p_new = greedy_crp_cost_aware(dem, CRPConfigCostAware(
            relay_capacity=cap, dpu_bytes_per_us=dbpu,
            link_bw_bytes_per_us=LINK_BW_BYTES_PER_US))
        c_old = run_variant(f"sweepN_gpd{gpus_per_domain}_old_{dbpu}", p_old, wcfg)
        c_new = run_variant(f"sweepN_gpd{gpus_per_domain}_new_{dbpu}", p_new, wcfg)
        results.append((dbpu, c0, c_old, c_new))
    return cap, cap_rows, results


def main():
    wl_dir = os.path.join(BASE, "workload/microbenchmarks/all_to_all/carts_16npus")
    write_2rank_workload(wl_dir, 1048576, num_ranks=NUM_GPUS)

    gpd_list = [8, 4, 2, 1]
    seed = 0
    all_rows = []

    for gpd in gpd_list:
        ndomains = NUM_GPUS // gpd
        if ndomains < 4:
            print(f"[skip] gpus_per_domain={gpd} -> {ndomains} domains "
                  f"(degenerate: placement decision is trivial, CARTS_STATE.md S7)")
            continue
        cap, cap_rows, results = sweep_one_gpd(gpd, seed=seed)
        print("=" * 100)
        print(f"gpus_per_domain={gpd} ({ndomains} domains), auto-picked cap={cap}, seed={seed}")
        print("=" * 100)
        print(f"{'dpu_bytes_per_us':>16} | {'C0':>8} | {'OLD':>8} | {'NEW':>8} | "
              f"{'OLDvsC0':>9} | {'NEWvsC0':>9} | {'FixB delta':>10}")
        print("-" * 100)
        for (dbpu, c0, c_old, c_new) in results:
            if c0 is None or c_old is None or c_new is None:
                print(f"{dbpu:>16,} | RUN FAILED")
                continue
            old_vs_c0 = 100.0 * (c0 - c_old) / c0
            new_vs_c0 = 100.0 * (c0 - c_new) / c0
            fixb = 100.0 * (c_old - c_new) / c_old
            print(f"{dbpu:>16,} | {c0:>8} | {c_old:>8} | {c_new:>8} | "
                  f"{old_vs_c0:>+8.2f}% | {new_vs_c0:>+8.2f}% | {fixb:>+9.2f}%")
            all_rows.append([gpd, ndomains, seed, cap, dbpu, c0, c_old, c_new,
                              old_vs_c0, new_vs_c0, fixb])
        print()

    csv_path = os.path.join(_HERE, "sweep_any_n_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpus_per_domain", "num_domains", "seed", "cap",
                    "dpu_bytes_per_us", "c0_cycles", "old_cycles", "new_cycles",
                    "old_vs_c0_pct", "new_vs_c0_pct", "fixb_delta_pct"])
        w.writerows(all_rows)
    print(f"[saved] {csv_path} ({len(all_rows)} rows)")


def main_multi_seed():
    """Multi-seed extension (follow-up session, S13 item 2). Reuses
    sweep_one_gpd() unchanged. Checks whether the gpus_per_domain
    characterization (S7, seed=0 only) holds up across seeds, mirroring
    S6's 6-seed discipline at 4 domains. Scope: gpd in [4, 2] only --
    [8, 1] already characterized as degenerate / zero-effect, no new
    information expected there."""
    wl_dir = os.path.join(BASE, "workload/microbenchmarks/all_to_all/carts_16npus")
    write_2rank_workload(wl_dir, 1048576, num_ranks=NUM_GPUS)

    gpd_list = [4, 2]
    seeds = [0, 1, 2, 3, 4, 5]
    all_rows = []
    summary = {gpd: [] for gpd in gpd_list}

    for gpd in gpd_list:
        ndomains = NUM_GPUS // gpd
        print("=" * 100)
        print(f"MULTI-SEED: gpus_per_domain={gpd} ({ndomains} domains), seeds={seeds}")
        print("=" * 100)
        for seed in seeds:
            cap, cap_rows, results = sweep_one_gpd(gpd, seed=seed)
            for (dbpu, c0, c_old, c_new) in results:
                if c0 is None or c_old is None or c_new is None:
                    print(f"  seed={seed} dbpu={dbpu} | RUN FAILED")
                    continue
                old_vs_c0 = 100.0 * (c0 - c_old) / c0
                new_vs_c0 = 100.0 * (c0 - c_new) / c0
                fixb = 100.0 * (c_old - c_new) / c_old
                all_rows.append([gpd, ndomains, seed, cap, dbpu, c0, c_old, c_new,
                                  old_vs_c0, new_vs_c0, fixb])
                if dbpu == DPU_SWEEP[0]:
                    summary[gpd].append((seed, c0, cap, old_vs_c0, fixb))
        print(f"{'seed':>5} | {'C0':>8} | {'cap':>5} | {'OLDvsC0%':>9} | {'FixBgain%':>10}")
        print("-" * 50)
        for (seed, c0, cap, old_vs_c0, fixb) in summary[gpd]:
            print(f"{seed:>5} | {c0:>8} | {cap:>5} | {old_vs_c0:>+8.2f}% | {fixb:>+9.2f}%")
        print()

    csv_path = os.path.join(_HERE, "sweep_multi_seed_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpus_per_domain", "num_domains", "seed", "cap",
                    "dpu_bytes_per_us", "c0_cycles", "old_cycles", "new_cycles",
                    "old_vs_c0_pct", "new_vs_c0_pct", "fixb_delta_pct"])
        w.writerows(all_rows)
    print(f"[saved] {csv_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    if "--multi-seed" in sys.argv:
        main_multi_seed()
    else:
        main()


