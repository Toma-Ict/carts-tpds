#!/usr/bin/env python3
"""
make_m3_scatter_fig_v2.py -- Fig. 4 (M3 correlation scatter), legend fix (S44).

Supersedes make_m3_scatter_fig.py, which is kept unmodified. In the 12
DPU-bound cells the max-link proxy is exactly zero, so Pearson r is undefined
there; v1's corr() returns 0.0 when a variance is zero, which is how
"DPU-bound (r=+0.00)" reached the legend. v2 labels that series
"DPU-bound (proxy = 0)" and asserts the zero variance directly instead of
asserting r_dpu == 0.00.

Everything else is identical to v1: proxies, corr and fit are imported from
v1 (not reimplemented), and the plot code, colours, jitter seed and the
forensic star are unchanged. Writes m3_scatter_v2.pdf (v1's PDF is not
overwritten).
Run:  CHAKRA_ROOT=/workspace/astra-sim/extern/graph_frontend python3 make_m3_scatter_fig_v2.py
"""
import csv, random, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_m3_scatter_fig import CSV, FORENSIC, proxies, corr, fit

OUT = "m3_scatter_v2.pdf"


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
    var_dpu = statistics.pvariance([x['ml'] for x in db])
    print(f"n={n}  full max-link r={r_ml:+.3f}  total-load r={r_tl:+.3f}  "
          f"transfer-bound r={r_tb:+.3f}  DPU-bound n={len(db)} "
          f"proxy variance={var_dpu} (r undefined)")
    assert n == 72, f"expected 72 cells, got {n} -- wrong CSV?"
    assert abs(r_ml - 0.56) < 0.03, f"max-link r {r_ml:.3f} != 0.56"
    assert abs(r_tl - 0.64) < 0.03, f"total-load r {r_tl:.3f} != 0.64"
    assert len(db) == 12, f"expected 12 DPU-bound cells, got {len(db)}"
    assert var_dpu == 0.0, f"DPU-bound proxy variance {var_dpu} != 0 -- legend no longer valid"

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
                marker="^", label="DPU-bound (proxy = 0)")
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
    plt.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
