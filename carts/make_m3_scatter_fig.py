#!/usr/bin/env python3
"""
make_m3_scatter_fig.py -- M3 correlation scatter (Table II -> figure).
Self-contained: inlines the exact proxy computation from the archived
m3_corr_full.py, reads sensitivity_gpd8_results.csv (72-cell grid), computes
Pearson r, ASSERTS against locked Table II (max-link 0.56, total-load 0.64,
DPU-bound max-link 0.00), then plots cycles-gain vs max-link proxy-gain,
colored transfer-bound vs DPU-bound. DPU-bound points are jittered +-0.3% on x
for visibility (disclosed in caption); the bw=5/dpu=20500/seed=1 forensic case
is highlighted with a star. Append-only wrt other files.
Run:  CHAKRA_ROOT=/workspace/astra-sim/extern/graph_frontend python3 make_m3_scatter_fig.py
"""
import csv, os, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import demand_extractor as D
from greedy_crp import CRPConfigCostAware, greedy_crp_cost_aware

CSV = "sensitivity_gpd8_results.csv"
FORENSIC = (5, 20500, 1)   # (bw, dpu, seed): 0% max-link proxy, +16.6% cycles
_cache = {}

def proxies(bw, dpu, seed, cap):
    key = (bw, dpu, seed)
    if key in _cache: return _cache[key]
    cfg = D.MoEConfig(num_gpus=32, gpus_per_domain=8, num_experts=32,
                      top_k=2, tokens_per_gpu=64, skew=1.5, seed=seed)
    res = D.generate_demand(cfg)
    crp = CRPConfigCostAware(relay_capacity=cap, dpu_bytes_per_us=dpu,
                             token_bytes=2048, link_bw_bytes_per_us=bw * 1000.0)
    p = greedy_crp_cost_aware(res, crp); pr = p.inter_pairs()
    mb = max(res.lam(*q) for q in pr); ma = max(p.load[q] for q in pr)
    tb = sum(res.lam(*q) for q in pr); ta = sum(p.load[q] for q in pr)
    out = ((mb - ma) / mb * 100.0, (tb - ta) / tb * 100.0)
    _cache[key] = out; return out

def corr(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** .5; db = sum((y - mb) ** 2 for y in b) ** .5
    return num / (da * db) if da * db else 0.0

def fit(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return b, my - b * mx


def main():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            bw = int(r['bw_gbs']); dpu = int(r['dpu_bytes_per_us'])
            seed = int(r['seed']); cap = int(r['cap'])
            ml, tl = proxies(bw, dpu, seed, cap)
            rows.append(dict(bw=bw, dpu=dpu, seed=seed, cyc=float(r['new_vs_c0']),
                             ml=ml, tl=tl, tbound=(bw * 1000 < dpu)))
    n = len(rows)
    r_ml = corr([x['cyc'] for x in rows], [x['ml'] for x in rows])
    r_tl = corr([x['cyc'] for x in rows], [x['tl'] for x in rows])
    tb = [x for x in rows if x['tbound']]; db = [x for x in rows if not x['tbound']]
    r_tb = corr([x['cyc'] for x in tb], [x['ml'] for x in tb])
    r_dpu = corr([x['cyc'] for x in db], [x['ml'] for x in db])
    print(f"n={n}  full max-link r={r_ml:+.3f}  total-load r={r_tl:+.3f}  "
          f"transfer-bound r={r_tb:+.3f}  DPU-bound r={r_dpu:+.3f}")
    assert n == 72, f"expected 72 cells, got {n} -- wrong CSV?"
    assert abs(r_ml - 0.56) < 0.03, f"max-link r {r_ml:.3f} != 0.56"
    assert abs(r_tl - 0.64) < 0.03, f"total-load r {r_tl:.3f} != 0.64"
    assert abs(r_dpu - 0.00) < 0.03, f"DPU-bound r {r_dpu:.3f} != 0.00"

    fx = next(x for x in rows if (x['bw'], x['dpu'], x['seed']) == FORENSIC)
    print(f"forensic cell {FORENSIC}: max-link proxy={fx['ml']:.2f}%  cycles={fx['cyc']:.2f}%")

    rng = random.Random(0)
    def jit(x):  # visual-only x-jitter for the piled-up DPU-bound cluster
        return x + rng.uniform(-1.2, 1.2)

    plt.rcParams.update({"font.size": 8})
    plt.figure(figsize=(3.4, 2.7))
    plt.scatter([x['ml'] for x in tb], [x['cyc'] for x in tb], s=22, c="#2a78d6",
                marker="o", alpha=0.8, edgecolors="none",
                label=f"transfer-bound (r={r_tb:+.2f})")
    plt.scatter([jit(x['ml']) for x in db], [x['cyc'] for x in db], s=30,
                facecolors="none", edgecolors="#e34948", linewidths=1.0, alpha=0.65,
                marker="^", label=f"DPU-bound (r={r_dpu:+.2f})")
    xs = [x['ml'] for x in tb]; b, a = fit(xs, [x['cyc'] for x in tb])
    xr = [min(xs), max(xs)]
    plt.plot(xr, [a + b * x for x in xr], "-", c="#2a78d6", lw=1.2, zorder=1)
    plt.scatter([fx['ml']], [fx['cyc']], s=130, marker="*", c="#eda100",
                edgecolors="#854f0b", linewidths=0.6, zorder=5)
    plt.annotate("0% proxy,\n+16.6% cycles", xy=(fx['ml'], fx['cyc']),
                 xytext=(fx['ml'] + 4, fx['cyc'] + 1.5), fontsize=6.5,
                 color="#5f5e5a",
                 arrowprops=dict(arrowstyle="->", color="#5f5e5a", lw=0.6))
    plt.xlabel("max-link load reduction  (proxy, %)")
    plt.ylabel("simulated cycles gain  (%)")
    plt.legend(frameon=False, fontsize=7, loc="lower right")
    plt.xlim(left=-3)
    plt.grid(True, lw=0.4, alpha=0.5)
    plt.tight_layout(pad=0.3)
    os.makedirs("paper_figures", exist_ok=True)
    plt.savefig("paper_figures/m3_scatter.pdf", bbox_inches="tight")
    print("wrote paper_figures/m3_scatter.pdf")


if __name__ == "__main__":
    main()
