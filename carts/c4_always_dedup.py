"""
c4_always_dedup.py -- C4 "always-dedup" baseline (HierMoE-style fixed dedup,
no placement decision, no capacity) over the S25 12-cell bw x DPU grid,
seeds 0-5, gpd=8, num_gpus=32.

C4 = dedup EVERY pair, capacity-free: greedy_crp at relay_capacity=10**9
serves all pairs (S3a cap=inf collapse, used deliberately). The cap-FCFS
variant of "always dedup" is definitionally Algorithm 1 (NAIVE); its numbers
already exist as old_vs_c0 in sensitivity_gpd8_results.csv -- not re-run.

Validation anchors:
  (i)  C0 re-run per (bw, seed) must match the existing sensitivity CSV
       exactly (C0 is dpu-independent, S25-verified);
  (ii) C4 placement must serve every pair with raw>0 (asserted).

Append-only: modifies no validated file. Run from carts dir:
  python3 c4_always_dedup.py
"""
import os, sys, csv, time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(os.path.dirname(_HERE))  # -> /workspace/astra-sim

from demand_extractor import MoEConfig, generate_demand
from greedy_crp import CRPConfig, greedy_crp
from carts_et_writer import WriterConfig
from make_2rank_workload import write_2rank_workload
from sweep_any_n import emit_ets, write_system_json
from scale_multiseed import run_cycles_n

GPD, NUM_GPUS, SKEW = 8, 32, 1.5
SEEDS = [0, 1, 2, 3, 4, 5]
BW_LIST = [5, 10, 20, 50]
DPU_LIST = [20500, 50000, 83900]
CAP_INF = 10**9
BW_YAML = {5:  "examples/network/analytical/Switch_32npus_slow.yml",
           10: "examples/network/analytical/Switch_32npus_bw10.yml",
           20: "examples/network/analytical/Switch_32npus_bw20.yml",
           50: "examples/network/analytical/Switch_32npus_bw50.yml"}
BASE = "/workspace/astra-sim/examples"
REF_CSV = os.path.join(_HERE, "sensitivity_gpd8_results.csv")


def load_ref():
    ref = {}
    with open(REF_CSV) as f:
        for r in csv.DictReader(f):
            k = (int(r["bw_gbs"]), int(r["dpu_bytes_per_us"]), int(r["seed"]))
            ref[k] = r
    return ref


def main():
    ref = load_ref()
    wl_dir = os.path.join(BASE,
        f"workload/microbenchmarks/all_to_all/scale_{NUM_GPUS}npus")
    wl = (f"examples/workload/microbenchmarks/all_to_all/"
          f"scale_{NUM_GPUS}npus/carts_wl")
    write_2rank_workload(wl_dir, 1048576, num_ranks=NUM_GPUS)
    rows, t0 = [], time.time()
    for seed in SEEDS:
        dem = generate_demand(MoEConfig(skew=SKEW, seed=seed,
                              num_gpus=NUM_GPUS, gpus_per_domain=GPD,
                              num_experts=NUM_GPUS))
        n = dem.cfg.num_domains
        p_c0 = greedy_crp(dem, CRPConfig(relay_capacity=0))
        p_c4 = greedy_crp(dem, CRPConfig(relay_capacity=CAP_INF))
        for i in range(n):
            for j in range(n):
                if i != j and dem.lam(i, j) > 0:
                    assert p_c4.served[(i, j)], f"C4 unserved {(i,j)} s{seed}"
        for bw in BW_LIST:
            net = BW_YAML[bw]
            name0 = f"c4b_{NUM_GPUS}_c0"
            emit_ets(os.path.join(BASE, "system/custom_collectives", name0),
                     name0, p_c0, WriterConfig(dpu_bytes_per_us=DPU_LIST[0]))
            write_system_json(name0)
            c0 = run_cycles_n(name0, wl, net)
            r0 = ref.get((bw, DPU_LIST[0], seed))
            if r0 is not None:
                assert c0 == int(r0["c0"]), \
                    f"C0 anchor mismatch bw={bw} s{seed}: {c0} vs {r0['c0']}"
            for dpu in DPU_LIST:
                name4 = f"c4b_{NUM_GPUS}_c4"
                emit_ets(os.path.join(BASE, "system/custom_collectives",
                         name4), name4, p_c4,
                         WriterConfig(dpu_bytes_per_us=dpu))
                write_system_json(name4)
                c4 = run_cycles_n(name4, wl, net)
                g = 100.0 * (c0 - c4) / c0
                rr = ref.get((bw, dpu, seed), {})
                rows.append(dict(bw_gbs=bw, dpu_bytes_per_us=dpu, seed=seed,
                                 c0=c0, c4=c4, c4_vs_c0=round(g, 4),
                                 old_vs_c0=rr.get("old_vs_c0", ""),
                                 new_vs_c0=rr.get("new_vs_c0", "")))
                print(f"bw={bw:>2} dpu={dpu:>5} s{seed} c0={c0} c4={c4} "
                      f"C4vsC0={g:+.2f}% (NAIVE={rr.get('old_vs_c0','')}, "
                      f"CA={rr.get('new_vs_c0','')})")
    out = os.path.join(_HERE, "c4_always_dedup_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[saved] {out} ({len(rows)} rows, {time.time()-t0:.1f}s)\n")
    print(f"{'bw':>4} | {'dpu':>6} | {'C4 mean':>9} | {'C4>=C0':>6} | {'CA mean':>8}")
    for bw in BW_LIST:
        for dpu in DPU_LIST:
            cl = [r for r in rows if r["bw_gbs"] == bw
                  and r["dpu_bytes_per_us"] == dpu]
            m = sum(r["c4_vs_c0"] for r in cl) / len(cl)
            ge = sum(1 for r in cl if r["c4_vs_c0"] >= 0)
            ca = [float(r["new_vs_c0"]) for r in cl if r["new_vs_c0"] != ""]
            cam = sum(ca) / len(ca) if ca else float("nan")
            print(f"{bw:>4} | {dpu:>6} | {m:>+8.2f}% | {ge}/6   | {cam:>+7.2f}%")


if __name__ == "__main__":
    main()
