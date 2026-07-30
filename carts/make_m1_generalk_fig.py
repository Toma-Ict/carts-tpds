#!/usr/bin/env python3
"""
make_m1_generalk_fig.py -- M1 general-k figure (gamma vs gpd, k=2/3/4).
Reuses the VALIDATED exact-gamma computation from general_k_validate.py
(exact_m_distribution, gain_from_distribution, in_domain_set); does NOT
reimplement it. Computes the 12 (k,gpd) gamma values at the E=16 seed-0
isolation config, asserts against the locked M1-doc anchors, and writes
paper_figures/m1_generalk.pdf.
Append-only: imports general_k_validate + demand_extractor, edits neither.
Run:  python3 make_m1_generalk_fig.py
"""
import os, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import demand_extractor as D
import general_k_validate as G

K_VALUES = (2, 3, 4)
GPD_VALUES = (1, 2, 4, 8)
E, N, TPG, SKEW = G.E, G.N, G.TPG, G.SKEW

# locked anchors: M1_load_level_dedup_model.md section 6 (exact gamma %, seed 0)
ANCHORS = {
    (2, 8): 31.86, (2, 4): 11.64, (2, 2): 3.60, (2, 1): 0.00,
    (3, 8): 48.88, (3, 4): 22.80, (3, 2): 7.68, (3, 1): 0.00,
    (4, 8): 58.74, (4, 4): 32.75, (4, 2): 12.10, (4, 1): 0.00,
}


def gamma_at(k, gpd):
    cfg = D.MoEConfig(num_gpus=N, gpus_per_domain=gpd, num_experts=E,
                      top_k=k, tokens_per_gpu=TPG, skew=SKEW, seed=0)
    weights = D._weights(cfg, random.Random(0))
    doms = {j: G.in_domain_set(gpd, j) for j in range(N // gpd)}
    jstar = max(doms, key=lambda j: sum(weights[e] for e in doms[j]))
    dist = G.exact_m_distribution(weights, doms[jstar], k)
    return G.gain_from_distribution(dist) * 100.0


def main():
    curves = {}
    print(f"{'k':>3} {'gpd':>4} {'computed%':>10} {'anchor%':>9} {'diff':>7}")
    for k in K_VALUES:
        ys = []
        for gpd in GPD_VALUES:
            g = gamma_at(k, gpd); a = ANCHORS[(k, gpd)]
            print(f"{k:>3} {gpd:>4} {g:>10.4f} {a:>9.2f} {g - a:>7.3f}")
            assert abs(g - a) < 0.1, f"mismatch k={k} gpd={gpd}: {g:.4f} vs {a}"
            ys.append(g)
        curves[k] = ys
    os.makedirs("paper_figures", exist_ok=True)
    plt.rcParams.update({"font.size": 8})
    styles = {2: ("-", "o", "#2a78d6"), 3: ("--", "^", "#1baf7a"), 4: (":", "s", "#eda100")}
    x = list(range(len(GPD_VALUES)))
    plt.figure(figsize=(3.4, 2.6))
    for k in K_VALUES:
        ls, mk, col = styles[k]
        plt.plot(x, curves[k], ls, marker=mk, color=col, lw=1.6, ms=5, label=f"k = {k}")
    plt.xticks(x, [str(g) for g in GPD_VALUES])
    plt.xlabel("gpus per domain (g)")
    plt.ylabel(r"per-link dedup gain  $\gamma$  (%)")
    plt.ylim(0, 65)
    plt.annotate(r"$\gamma=0$ (Thm 2)", xy=(0, 0), xytext=(0.2, 7), fontsize=7,
                 color="#5f5e5a", arrowprops=dict(arrowstyle="->", color="#5f5e5a", lw=0.7))
    plt.legend(frameon=False, loc="upper left")
    plt.grid(True, axis="y", lw=0.4, alpha=0.5)
    plt.tight_layout(pad=0.3)
    plt.savefig("paper_figures/m1_generalk.pdf", bbox_inches="tight")
    print("\nwrote paper_figures/m1_generalk.pdf")


if __name__ == "__main__":
    main()
