#!/usr/bin/env python3
"""
make_m2_roofline_fig.py -- M2 bandwidth-bridge figure (NEW gain vs bandwidth,
one line per DPU rate). Reads sensitivity_gpd8_results.csv, aggregates mean
new_vs_c0 per (bw, dpu) over 6 seeds, ASSERTS the 12 cell-means against locked
Table IV, and plots gain vs bandwidth (log-x) with one line per DPU rate --
showing the transfer-bound plateau and the DPU-rate-dependent crossover.
Append-only.  Run:  python3 make_m2_roofline_fig.py
"""
import csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "sensitivity_gpd8_results.csv"
BWS = [5, 10, 20, 50]
DPUS = [20500, 50000, 83900]

# locked Table IV: mean NEW-vs-C0 gain (%), rows=bw, cols=dpu
TABLE_IV = {
    (5, 20500): 10.3, (5, 50000): 6.9,  (5, 83900): 6.9,
    (10, 20500): 6.1, (10, 50000): 11.8, (10, 83900): 9.6,
    (20, 20500): 3.4, (20, 50000): 6.3,  (20, 83900): 10.7,
    (50, 20500): 3.4, (50, 50000): 3.4,  (50, 83900): 3.4,
}


def main():
    agg = defaultdict(list)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            bw = int(r['bw_gbs']); dpu = int(r['dpu_bytes_per_us'])
            agg[(bw, dpu)].append(float(r['new_vs_c0']))
    means = {k: sum(v) / len(v) for k, v in agg.items()}

    print(f"{'bw':>4} {'dpu':>6} {'mean%':>7} {'TableIV':>8} {'diff':>6} {'n':>3}")
    for bw in BWS:
        for dpu in DPUS:
            m = means[(bw, dpu)]; t = TABLE_IV[(bw, dpu)]
            print(f"{bw:>4} {dpu:>6} {m:>7.2f} {t:>8.1f} {m - t:>6.2f} {len(agg[(bw, dpu)]):>3}")
            assert abs(m - t) < 0.15, f"cell (bw={bw},dpu={dpu}) {m:.2f} != Table IV {t}"
    print("all 12 cell-means match Table IV")

    plt.rcParams.update({"font.size": 8})
    styles = {20500: ("-", "o", "#2a78d6", "20.5"),
              50000: ("--", "s", "#1baf7a", "50"),
              83900: (":", "^", "#eda100", "83.9")}
    plt.figure(figsize=(3.4, 2.7))
    for dpu in DPUS:
        ls, mk, col, lab = styles[dpu]
        ys = [means[(bw, dpu)] for bw in BWS]
        plt.plot(BWS, ys, ls, marker=mk, color=col, lw=1.6, ms=5,
                 label=f"DPU = {lab} GB/s")
    plt.xscale("log")
    plt.xticks(BWS, [str(b) for b in BWS])
    plt.minorticks_off()
    plt.axhline(0, color="#888780", lw=0.6, ls="-")
    plt.xlabel("link bandwidth (GB/s, log scale)")
    plt.ylabel("mean completion-time gain vs C0 (%)")
    plt.ylim(0, 13)
    plt.legend(frameon=False, fontsize=7, loc="upper right", title="dedup rate")
    plt.grid(True, lw=0.4, alpha=0.5)
    plt.tight_layout(pad=0.3)
    os.makedirs("paper_figures", exist_ok=True)
    plt.savefig("paper_figures/m2_roofline.pdf", bbox_inches="tight")
    print("wrote paper_figures/m2_roofline.pdf")


if __name__ == "__main__":
    main()
