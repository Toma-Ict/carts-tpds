"""
num_experts_sweep_gpd8.py -- num_experts sensitivity at the gpd=8 HEADLINE config.

Why this exists: num_experts_sweep.py (S19) ran at num_gpus=16 / gpus_per_domain=4,
which is the OLD headline config. The paper's headline is num_gpus=32 /
gpus_per_domain=8 (4 domains). Citing S19's numbers in Sec. V-A would state a
claim about the headline config using data from a different one. This script
re-runs the identical measurement at the headline config.

Append-only: does not modify num_experts_sweep.py, demand_extractor.py,
capacity_sweep.py. Reuses MoEConfigBlock from num_experts_sweep.py unchanged.

experts_per_gpu spans 1,2,4,8,16 exactly as S19 did, so E in {32,...,512} here
(E in {16,...,256} at N=16). E < N is excluded: it leaves GPUs holding no
expert, a different regime from the one the claim is about.

Pure Python, load-level (cap=inf), seeds 0-5. No ASTRA-sim, no CHAKRA_ROOT.

Two anchor gates before any row is trusted:
  A1  N=16, gpd=4, E=16, seed 0 -> raw=252, dedup=218   (S7 / S19 regression)
  A2  N=32, gpd=8, E=32, seed 0 -> raw_max=632          (S38 locked C0 L1)
"""
import os, sys, csv

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from demand_extractor import generate_demand
from capacity_sweep import max_link_load
from num_experts_sweep import MoEConfigBlock

NUM_GPUS = 32
GPUS_PER_DOMAIN = 8
SKEW = 1.5
EXPERT_COUNTS = [32, 64, 128, 256, 512]
SEEDS = [0, 1, 2, 3, 4, 5]


def measure(num_gpus, gpus_per_domain, num_experts, seed):
    cfg = MoEConfigBlock(skew=SKEW, seed=seed, num_gpus=num_gpus,
                         gpus_per_domain=gpus_per_domain,
                         num_experts=num_experts)
    dem = generate_demand(cfg)
    n = dem.cfg.num_domains
    inter = [(i, j) for i in range(n) for j in range(n) if i != j]
    raw_max = max(dem.lam(*p) for p in inter)
    dedup_max = max_link_load(dem, 10**9, "greedy", True)
    gain = 100.0 * (raw_max - dedup_max) / raw_max if raw_max else 0.0
    dfrac = 100.0 * dem.dedup_fraction(inter_only=True)
    return raw_max, dedup_max, gain, dfrac


def main():
    # ---- A1: S19 regression anchor (old config must still reproduce) ----
    r, d, _, _ = measure(16, 4, 16, 0)
    assert r == 252 and d == 218, (
        f"ANCHOR A1 FAIL: N=16 gpd=4 E=16 seed0 expected raw=252 dedup=218, "
        f"got raw={r} dedup={d}. Stop -- shared code changed behavior.")
    print("[anchor A1 OK] N=16 gpd=4 E=16 seed0 -> raw=252 dedup=218 (S7/S19)")

    # ---- A2: headline-config anchor (S38 locked C0 Layer-1) ----
    r, d, g, _ = measure(NUM_GPUS, GPUS_PER_DOMAIN, NUM_GPUS, 0)
    assert r == 632, (
        f"ANCHOR A2 FAIL: N=32 gpd=8 E=32 seed0 expected raw_max=632 "
        f"(S38 locked C0 L1), got {r}. Stop -- config does not match headline.")
    print(f"[anchor A2 OK] N=32 gpd=8 E=32 seed0 -> raw_max=632 (S38 C0 L1); "
          f"cap=inf dedup_max={d}, gain={g:.2f}%\n")

    print(f"NUM_EXPERTS SENSITIVITY at the HEADLINE config "
          f"(num_gpus={NUM_GPUS}, gpus_per_domain={GPUS_PER_DOMAIN} -> "
          f"{NUM_GPUS // GPUS_PER_DOMAIN} domains, skew={SKEW}, block mapping)")
    print(f"load-level full-dedup potential (cap=inf), seeds "
          f"{SEEDS[0]}-{SEEDS[-1]}\n")
    print(f"{'n_exp':>6} | {'e/gpu':>5} | {'seed':>4} | {'raw':>5} | "
          f"{'dedup':>5} | {'gain%':>6} | {'dfrac%':>6}")
    print("-" * 60)

    rows, summ = [], {}
    for ne in EXPERT_COUNTS:
        epg = ne // NUM_GPUS
        gains = []
        for s in SEEDS:
            raw_max, dedup_max, gain, dfrac = measure(
                NUM_GPUS, GPUS_PER_DOMAIN, ne, s)
            gains.append(gain)
            rows.append([ne, epg, s, raw_max, dedup_max, gain, dfrac])
            print(f"{ne:>6} | {epg:>5} | {s:>4} | {raw_max:>5} | "
                  f"{dedup_max:>5} | {gain:>5.2f}% | {dfrac:>5.2f}%")
        summ[ne] = (epg, sum(gains) / len(gains), min(gains), max(gains))
        print("-" * 60)

    print("\nSUMMARY (gain% across seeds):")
    print(f"{'n_exp':>6} | {'e/gpu':>5} | {'mean':>7} | {'min':>7} | {'max':>7}")
    print("-" * 48)
    all_means = []
    for ne in EXPERT_COUNTS:
        epg, mean, lo, hi = summ[ne]
        all_means.append(mean)
        print(f"{ne:>6} | {epg:>5} | {mean:>6.2f}% | {lo:>6.2f}% | {hi:>6.2f}%")

    print(f"\nBAND across expert counts: {min(all_means):.2f}% -- "
          f"{max(all_means):.2f}%  (this is the number for the Sec. V-A "
          f"sentence; do NOT write 15-20% unless this band says so)")

    csv_path = os.path.join(_HERE, "num_experts_sweep_gpd8_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_experts", "experts_per_gpu", "seed",
                    "raw_max", "dedup_max", "gain_pct", "dedup_frac_pct"])
        w.writerows(rows)
    print(f"\n[saved] {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
