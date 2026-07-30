#!/usr/bin/env python3
# make_necessity_fig_v3.py -- Fig 8 re-render (v3, append-only).
# vs make_necessity_fig_v2.py: adds the C4 always-dedup baseline line
# (c4_always_dedup_results.csv, dpu=20500 column) as dashed dark-gray;
# annotation updated (C4 and NAIVE coincide at bw=50: both -32.70%);
# assert-gated against v2's locked anchors PLUS the new S29 C4 anchors.
import csv, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV    = "sensitivity_gpd8_results.csv"
C4CSV  = "c4_always_dedup_results.csv"
OUTFILE = "fig_necessity_v3.pdf"
NAIVE_RED  = "#D9534F"
CARTS_BLUE = "#2A7FB8"
C4_DARK    = "#3B3B3B"
GRAY       = "#444444"
TOL = 0.05
LOCKED_NEW = {5: 10.29, 10: 6.10, 20: 3.40, 50: 3.38}
LOCKED_OLD = {5: 6.94,  10: 5.83, 20: -1.49, 50: -32.70}
LOCKED_C4  = {5: 6.30,  10: 4.73, 20: -2.60, 50: -32.70}


def collect():
    new_by_bw, old_by_bw = defaultdict(list), defaultdict(list)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if int(r["dpu_bytes_per_us"]) != 20500:
                continue
            bw = int(r["bw_gbs"])
            new_by_bw[bw].append(float(r["new_vs_c0"]))
            old_by_bw[bw].append(float(r["old_vs_c0"]))
    c4_by_bw = defaultdict(list)
    with open(C4CSV) as f:
        for r in csv.DictReader(f):
            if int(r["dpu_bytes_per_us"]) != 20500:
                continue
            c4_by_bw[int(r["bw_gbs"])].append(float(r["c4_vs_c0"]))
    return new_by_bw, old_by_bw, c4_by_bw


def check(means, locked, name):
    for bw, exp in locked.items():
        got = means[bw]
        assert abs(got - exp) <= TOL, (
            "LOCKED-ANCHOR FAIL %s bw=%d: got %.3f, locked %.2f"
            % (name, bw, got, exp))
    print("assert OK: %s means match locked anchors" % name)


def main():
    new_by_bw, old_by_bw, c4_by_bw = collect()
    bws = sorted(new_by_bw)
    new_means = {bw: st.mean(new_by_bw[bw]) for bw in bws}
    old_means = {bw: st.mean(old_by_bw[bw]) for bw in bws}
    c4_means  = {bw: st.mean(c4_by_bw[bw]) for bw in bws}
    check(new_means, LOCKED_NEW, "CARTS(new)")
    check(old_means, LOCKED_OLD, "naive(old)")
    check(c4_means,  LOCKED_C4,  "always-dedup(c4)")
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for bw in bws:                      # per-seed spread (naive, as v2)
        vals = old_by_bw[bw]
        n = len(vals)
        offs = [(i - (n - 1) / 2.0) * 0.02 for i in range(n)]
        ax.scatter([bw * (1 + o) for o in offs], vals, marker="x",
                   color=GRAY, s=14, alpha=0.6, zorder=2)
    ax.axhline(0, color=GRAY, linewidth=1.0, zorder=1)
    ax.plot(bws, [c4_means[b] for b in bws], "--s", color=C4_DARK,
            markersize=4, linewidth=1.3,
            label="Always-dedup (C4)", zorder=3)
    ax.plot(bws, [old_means[b] for b in bws], "-^", color=NAIVE_RED,
            markersize=5, linewidth=1.5, label="Naive placement", zorder=4)
    ax.plot(bws, [new_means[b] for b in bws], "-o", color=CARTS_BLUE,
            markersize=5, linewidth=1.5, label="CARTS (cost-aware)",
            zorder=5)
    ax.annotate("both collapse once link is fast\n($-32.7\\,\\%$)",
                xy=(50, old_means[50]), xytext=(13, -26),
                fontsize=6.5, color=NAIVE_RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=NAIVE_RED,
                                lw=0.8))
    ax.set_xscale("log", base=2)
    ax.set_xticks(bws)
    ax.set_xticklabels([str(b) for b in bws])
    ax.set_xlabel("bandwidth (GB/s), DPU = 20.5 GB/s", fontsize=8)
    ax.set_ylabel("gain vs C0 (%)")
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUTFILE)
    plt.close(fig)
    print("wrote", OUTFILE)
    print("CARTS means:", [round(new_means[b], 3) for b in bws])
    print("naive means:", [round(old_means[b], 3) for b in bws])
    print("C4 means:   ", [round(c4_means[b], 3) for b in bws])


if __name__ == "__main__":
    main()
