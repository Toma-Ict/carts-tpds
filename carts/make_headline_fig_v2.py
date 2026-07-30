#!/usr/bin/env python3
# make_headline_fig_v2.py -- Fig 7 restyle (append-only).
# vs make_paper_figures.py::fig_headline():
#   per-seed -> gray x (matches fig_necessity_v2), mean -> CARTS blue,
#   bold zero line, +10.3% annotation, assert-gated (Table IV anchors).
import csv, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "scale_multiseed_gpd8_results.csv"
OUTFILE = "fig_headline_v2.pdf"
CARTS_BLUE = "#2A7FB8"
GRAY = "#444444"
TOL = 0.1
LOCKED = {4: 10.3, 8: 5.9, 16: 1.0}

def collect():
    by_dom = defaultdict(list)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            by_dom[int(r["num_domains"])].append(float(r["new_vs_c0"]))
    return by_dom

def main():
    by_dom = collect()
    doms = sorted(by_dom)
    means = {d: st.mean(by_dom[d]) for d in doms}
    for d, exp in LOCKED.items():
        assert abs(means[d] - exp) <= TOL, (
            "LOCKED-ANCHOR FAIL dom=%d: got %.3f, locked %.1f"
            % (d, means[d], exp))
    print("assert OK: multi-seed means match Table IV anchors")

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    positions = {4: 1, 8: 2, 16: 3}
    for d, pos in positions.items():
        vals = by_dom[d]
        n = len(vals)
        offs = [(i - (n - 1) / 2.0) * 0.05 for i in range(n)]
        ax.scatter([pos + o for o in offs], vals, marker="x",
                   color=GRAY, s=14, alpha=0.6, zorder=2)
    ax.axhline(0, color=GRAY, linewidth=1.0, zorder=1)
    xs = [positions[d] for d in doms]
    ys = [means[d] for d in doms]
    ax.plot(xs, ys, "-o", color=CARTS_BLUE, markersize=5,
            linewidth=1.5, zorder=4)
    ax.annotate("$+10.3\\,\\%$ multi-seed mean",
                xy=(1, means[4]), xytext=(1.35, 16.5),
                fontsize=6.5, color=CARTS_BLUE,
                arrowprops=dict(arrowstyle="->", color=CARTS_BLUE,
                                lw=0.8))
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["4", "8", "16"])
    ax.set_xlabel("number of domains (gpd = 8 fixed)")
    ax.set_ylabel("gain vs C0 (%)")
    ax.set_xlim(0.6, 3.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUTFILE)
    plt.close(fig)
    print("wrote", OUTFILE)
    print("means:", [(d, round(means[d], 3)) for d in doms])

if __name__ == "__main__":
    main()
