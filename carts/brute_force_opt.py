#!/usr/bin/env python3
"""brute_force_opt.py -- APPEND-ONLY, READ-ONLY w.r.t. all validated files.

Measures the exact optimum of CRP objective (4a) against the cost-aware
greedy, at the headline topologies. Exact (not a bound) because capacity is
per-gateway and load(gw,dj) depends only on gw's own served set, so the
global min-max decomposes into independent per-gateway subproblems.

No ASTRA-sim. Objective (4a) is solver-side. Writes one CSV, nothing else.
"""
import csv, itertools, sys
import demand_extractor as D
from greedy_crp import CRPConfigCostAware, greedy_crp_cost_aware

CAP_CSV = "scale_multiseed_gpd8_results.csv"
OUT_CSV = "brute_force_opt_results.csv"

TOKEN_BYTES = 2048
DPU_BPUS    = 20500
LINK_BPUS   = 5000.0
SKEW        = 1.5
TOP_K       = 2


def load_caps():
    """Per-(num_gpus, seed) capacity, read from the locked headline CSV."""
    caps = {}
    with open(CAP_CSV) as f:
        for r in csv.DictReader(f):
            caps[(int(r["num_gpus"]), int(r["seed"]))] = int(r["cap"])
    return caps


def build_demand(num_gpus, gpd, seed, tokens_per_gpu):
    cfg = D.MoEConfig(num_gpus=num_gpus, gpus_per_domain=gpd,
                      num_experts=num_gpus, top_k=TOP_K,
                      tokens_per_gpu=tokens_per_gpu, skew=SKEW, seed=seed)
    out = D.generate_demand(cfg)
    return out[0] if isinstance(out, tuple) else out


def gateway_optimum(res, gw, n, cap):
    """Exhaustive over served subsets of gw's outgoing pairs.
    Returns (best_max_load, best_subset_size) or (None, None) if infeasible."""
    pairs = [(gw, dj) for dj in range(n) if dj != gw and res.lam(gw, dj) > 0]
    if not pairs:
        return 0, 0
    raw = [res.lam(a, b) for (a, b) in pairs]
    uni = [res.unique(a, b) for (a, b) in pairs]
    best, best_k = None, None
    for mask in range(1 << len(pairs)):
        used = sum(uni[i] for i in range(len(pairs)) if mask >> i & 1)
        if used > cap:
            continue
        m = max(uni[i] if mask >> i & 1 else raw[i] for i in range(len(pairs)))
        if best is None or m < best:
            best, best_k = m, bin(mask).count("1")
    return best, best_k


def run_one(num_gpus, gpd, seed, cap, tokens_per_gpu):
    res = build_demand(num_gpus, gpd, seed, tokens_per_gpu)
    n = num_gpus // gpd
    crp = CRPConfigCostAware(relay_capacity=cap, dpu_bytes_per_us=DPU_BPUS,
                             token_bytes=TOKEN_BYTES,
                             link_bw_bytes_per_us=LINK_BPUS)
    pl = greedy_crp_cost_aware(res, crp)
    greedy_obj = pl.max_load_after()
    raw_obj = pl.max_load_before()

    per_gw, infeasible = [], False
    for gw in range(n):
        b, _k = gateway_optimum(res, gw, n, cap)
        if b is None:
            infeasible = True
            break
        per_gw.append(b)
    opt_obj = None if infeasible else max(per_gw)

    return dict(num_gpus=num_gpus, num_domains=n, seed=seed, cap=cap,
                raw_max=raw_obj, greedy_obj=greedy_obj, opt_obj=opt_obj,
                gap_abs=None if opt_obj is None else greedy_obj - opt_obj,
                ratio=None if not opt_obj else round(greedy_obj / opt_obj, 6),
                is_optimal=None if opt_obj is None else (greedy_obj == opt_obj))


def anchor_gate(tokens_per_gpu, caps):
    """Seed 0, 32 GPUs, gpd=8 must reproduce the locked Fig. 5 numbers."""
    res = build_demand(32, 8, 0, tokens_per_gpu)
    l20, u20 = res.lam(2, 0), res.unique(2, 0)
    cap = caps[(32, 0)]
    r = run_one(32, 8, 0, cap, tokens_per_gpu)
    print(f"ANCHOR tokens_per_gpu={tokens_per_gpu}: lam(2,0)={l20} "
          f"(want 632)  unique(2,0)={u20} (want 461)  "
          f"cap={cap}  greedy_obj={r['greedy_obj']} (want 466)")
    return l20 == 632 and u20 == 461 and r["greedy_obj"] == 466


if __name__ == "__main__":
    caps = load_caps()
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    if not anchor_gate(T, caps):
        print("ANCHOR FAILED -- not writing CSV. "
              "Retry with a different tokens_per_gpu, e.g.:")
        print("   python3 brute_force_opt.py 128")
        sys.exit(1)
    print("ANCHOR OK\n")
    rows = []
    for num_gpus, gpd in ((32, 8), (64, 8), (128, 8)):
        for seed in range(6):
            key = (num_gpus, seed)
            if key not in caps:
                continue
            rows.append(run_one(num_gpus, gpd, seed, caps[key], T))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'GPUs':>5} {'dom':>4} {'seed':>5} {'raw':>6} {'greedy':>7} "
          f"{'opt':>6} {'gap':>5} {'ratio':>7}  optimal")
    for r in rows:
        print(f"{r['num_gpus']:>5} {r['num_domains']:>4} {r['seed']:>5} "
              f"{r['raw_max']:>6} {r['greedy_obj']:>7} {str(r['opt_obj']):>6} "
              f"{str(r['gap_abs']):>5} {str(r['ratio']):>7}  {r['is_optimal']}")
    ok = sum(1 for r in rows if r["is_optimal"])
    print(f"\nGreedy attains the exact optimum on {ok}/{len(rows)} runs.")
