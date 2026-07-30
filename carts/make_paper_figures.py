"""
make_paper_figures.py -- generate the 4 CARTS-paper evaluation/mechanism
figures directly from the validated result CSVs. No numbers are hand-typed;
every point plotted is read from disk.

Reads (all already present in this dir, from prior sweeps):
    scale_multiseed_gpd8_results.csv   -> fig_headline.pdf
    sensitivity_gpd8_results.csv       -> fig_necessity.pdf
    scale_multiseed_gpd4_results.csv   -> fig_envelope.pdf
    sweep_multi_seed_results.csv       -> fig_gpdeffect.pdf

Run from the carts dir:  python3 make_paper_figures.py
Outputs 4 PDFs (vector, embeddable via \includegraphics) in ./paper_figures/
"""
import csv, os, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "paper_figures"
os.makedirs(OUT, exist_ok=True)

# ---- shared style: keep close to the paper's pgfplots look ----
plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.figsize": (3.4, 2.3),   # single-column IEEE width
})
BLUE = "#3060A8"
RED = "#B03030"
GRAY = "#808080"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def jitter(n, spread=0.16):
    """n evenly spaced offsets centered at 0, for de-overlapping scatter."""
    if n == 1:
        return [0.0]
    step = 2 * spread / (n - 1)
    return [-spread + i * step for i in range(n)]


# ============================================================
# Fig: headline gain vs domains, gpd=8 (Table III's source)
# ============================================================
def fig_headline():
    rows = read_csv("scale_multiseed_gpd8_results.csv")
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[int(r["num_domains"])].append(float(r["new_vs_c0"]))

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    positions = {4: 1, 8: 2, 16: 3}
    means = []
    for dom, pos in positions.items():
        vals = by_dom[dom]
        offs = jitter(len(vals))
        ax.scatter([pos + o for o in offs], vals, facecolors="none",
                   edgecolors=BLUE, s=18, linewidths=0.9, alpha=0.85, zorder=3)
        means.append((pos, st.mean(vals)))
    xs, ys = zip(*means)
    ax.plot(xs, ys, "-o", color=RED, markersize=5, linewidth=1.4, zorder=4)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["4", "8", "16"])
    ax.set_xlabel("number of domains (gpd = 8 fixed)")
    ax.set_ylabel("completion-time gain vs C0 (%)")
    ax.set_xlim(0.6, 3.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUT}/fig_headline.pdf")
    plt.close(fig)
    print("fig_headline:", [(d, round(st.mean(v), 3)) for d, v in by_dom.items()])


# ============================================================
# Fig: necessity, NEW vs OLD across bandwidth, DPU = 20.5 GB/s
# (sensitivity_gpd8_results.csv's source)
# ============================================================
def fig_necessity():
    rows = read_csv("sensitivity_gpd8_results.csv")
    new_by_bw, old_by_bw = defaultdict(list), defaultdict(list)
    for r in rows:
        if int(r["dpu_bytes_per_us"]) != 20500:
            continue
        bw = int(r["bw_gbs"])
        new_by_bw[bw].append(float(r["new_vs_c0"]))
        old_by_bw[bw].append(float(r["old_vs_c0"]))

    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    bws = sorted(new_by_bw)
    for bw in bws:
        vals = old_by_bw[bw]
        jf = [bw * (1 + 0.02 * o) for o in jitter(len(vals), spread=1.0)]
        ax.scatter(jf, vals, marker="x", color=GRAY, s=14, alpha=0.6, zorder=2)
    new_means = [st.mean(new_by_bw[bw]) for bw in bws]
    old_means = [st.mean(old_by_bw[bw]) for bw in bws]
    ax.plot(bws, old_means, "-^", color=RED, markersize=5, linewidth=1.4,
            label="OLD mean", zorder=4)
    ax.plot(bws, new_means, "-o", color=BLUE, markersize=5, linewidth=1.4,
            label="NEW mean", zorder=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(bws); ax.set_xticklabels([str(b) for b in bws])
    ax.set_xlabel("bandwidth (GB/s), DPU = 20.5 GB/s", fontsize=8)
    ax.set_ylabel("gain vs C0 (%)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUT}/fig_necessity.pdf")
    plt.close(fig)
    print("fig_necessity NEW:", [round(m, 3) for m in new_means],
          "OLD:", [round(m, 3) for m in old_means])


# ============================================================
# Fig: operating envelope, gain vs domains, gpd=4
# (scale_multiseed_gpd4_results.csv's source)
# ============================================================
def fig_envelope():
    rows = read_csv("scale_multiseed_gpd4_results.csv")
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[int(r["num_domains"])].append(float(r["new_vs_c0"]))

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    positions = {32: 1, 64: 2, 128: 3}
    means = []
    for dom, pos in positions.items():
        vals = by_dom[dom]
        offs = jitter(len(vals))
        ax.scatter([pos + o for o in offs], vals, facecolors="none",
                   edgecolors=BLUE, s=18, linewidths=0.9, alpha=0.85, zorder=3)
        means.append((pos, st.mean(vals)))
    xs, ys = zip(*means)
    ax.plot(xs, ys, "-o", color=RED, markersize=5, linewidth=1.4, zorder=4)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["32", "64", "128"])
    ax.set_xlabel("number of domains (gpd = 4 fixed)")
    ax.set_ylabel("completion-time gain vs C0 (%)")
    ax.set_xlim(0.6, 3.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUT}/fig_envelope.pdf")
    plt.close(fig)
    print("fig_envelope:", [(d, round(st.mean(v), 4)) for d, v in by_dom.items()])


# ============================================================
# Fig: gpd effect at fixed N=16, DPU = 20.5 GB/s (M1 evidence)
# (sweep_multi_seed_results.csv's source)
# ============================================================
def fig_gpdeffect():
    rows = read_csv("sweep_multi_seed_results.csv")
    by_gpd = defaultdict(list)
    for r in rows:
        if int(r["dpu_bytes_per_us"]) != 20500:
            continue
        by_gpd[int(r["gpus_per_domain"])].append(float(r["new_vs_c0_pct"]))

    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    positions = {2: 1, 4: 2}
    means = []
    for gpd, pos in positions.items():
        vals = by_gpd[gpd]
        offs = jitter(len(vals))
        ax.scatter([pos + o for o in offs], vals, facecolors="none",
                   edgecolors=BLUE, s=18, linewidths=0.9, alpha=0.85, zorder=3)
        means.append((pos, st.mean(vals)))
    xs, ys = zip(*means)
    ax.plot(xs, ys, "-o", color=RED, markersize=5, linewidth=1.4, zorder=4)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["gpd = 2", "gpd = 4"])
    ax.set_ylabel("completion-time gain vs C0 (%)")
    ax.set_xlim(0.5, 2.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUT}/fig_gpdeffect.pdf")
    plt.close(fig)
    print("fig_gpdeffect:", [(g, round(st.mean(v), 3)) for g, v in by_gpd.items()])


if __name__ == "__main__":
    fig_headline()
    fig_necessity()
    fig_envelope()
    fig_gpdeffect()
    print(f"\nDone. 4 PDFs written to ./{OUT}/")
