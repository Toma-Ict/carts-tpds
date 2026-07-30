"""
num_experts_sweep.py -- num_experts sensitivity check (load-level).
Goal: test whether CARTS's dedup gain is robust to expert count, or an
artifact of the validated num_experts==num_gpus(==16) setup. Holds
num_gpus=16, gpus_per_domain=4 (-> 4 domains, the headline config) FIXED
per S7 isolation methodology; varies only num_experts in {16,32,64,128,256}
(experts_per_gpu = 1,2,4,8,16). skew=1.5 to match all headline scripts
(make_ablation / capacity_sweep / multi_seed_sweep).
Mapping: gpu_of_expert(e)=e (original) breaks for num_experts>num_gpus
(expert index exceeds GPU count). Fixed via a BLOCK mapping subclass
(contiguous, matches DeepSpeed/Megatron expert-parallel placement):
gpu_of_expert(e) = e // (num_experts // num_gpus). At num_experts==16 this
reduces to e//1 == e -- identical to the original, so all S5/S6/S18
headline numbers are preserved (asserted via the 252/218 anchor below).
demand_extractor.py is NOT modified (append-only / subclass discipline).
Pure Python, load-level only -- no ASTRA-sim, no CHAKRA_ROOT needed.
Multi-seed (0-5) per S6 discipline. Cycles-level confirmation is a
separate follow-up once the load-level shape is known.
"""
import os, sys, csv
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from demand_extractor import MoEConfig, generate_demand
from capacity_sweep import max_link_load
NUM_GPUS = 16
GPUS_PER_DOMAIN = 4
SKEW = 1.5
EXPERT_COUNTS = [16, 32, 64, 128, 256]
SEEDS = [0, 1, 2, 3, 4, 5]


class MoEConfigBlock(MoEConfig):
    """Block (contiguous) expert->GPU mapping. Generalizes the original
    identity gpu_of_expert(e)=e to num_experts>num_gpus without modifying
    demand_extractor.py. Reduces EXACTLY to identity when num_experts==num_gpus."""
    def gpu_of_expert(self, e):
        epg = max(1, self.num_experts // self.num_gpus)
        return e // epg


def measure(num_experts, seed):
    cfg = MoEConfigBlock(skew=SKEW, seed=seed, num_gpus=NUM_GPUS,
                         gpus_per_domain=GPUS_PER_DOMAIN, num_experts=num_experts)
    dem = generate_demand(cfg)
    n = dem.cfg.num_domains
    inter = [(i, j) for i in range(n) for j in range(n) if i != j]
    raw_max = max(dem.lam(*p) for p in inter)
    dedup_max = max_link_load(dem, 10**9, "greedy", True)
    gain = 100.0 * (raw_max - dedup_max) / raw_max if raw_max else 0.0
    dfrac = 100.0 * dem.dedup_fraction(inter_only=True)
    return raw_max, dedup_max, gain, dfrac


def main():
    # Regression anchor: block mapping at num_experts==num_gpus must reproduce
    # the locked 4-domain seed=0 numbers (S7 / capacity_sweep.py: 252 / 218).
    r0, d0, _, _ = measure(NUM_GPUS, 0)
    assert r0 == 252 and d0 == 218, (
        f"ANCHOR FAIL: expected raw=252 dedup=218 at num_experts={NUM_GPUS} "
        f"seed=0, got raw={r0} dedup={d0}. Block subclass changed validated "
        f"behavior -- stop and investigate before trusting any row below.")
    print(f"[anchor OK] num_experts={NUM_GPUS} seed=0 -> raw=252 dedup=218 "
          f"(matches S7 / validated headline)\n")

    print(f"NUM_EXPERTS SENSITIVITY  (num_gpus={NUM_GPUS}, "
          f"gpus_per_domain={GPUS_PER_DOMAIN} -> {NUM_GPUS // GPUS_PER_DOMAIN} "
          f"domains, skew={SKEW}, block mapping)")
    print(f"load-level full-dedup potential (cap=inf), seeds "
          f"{SEEDS[0]}-{SEEDS[-1]}\n")
    print(f"{'n_exp':>6} | {'e/gpu':>5} | {'seed':>4} | {'raw':>5} | "
          f"{'dedup':>5} | {'gain%':>6} | {'dfrac%':>6}")
    print("-" * 60)

    rows = []
    summ = {}
    for ne in EXPERT_COUNTS:
        epg = ne // NUM_GPUS
        gains = []
        for s in SEEDS:
            raw_max, dedup_max, gain, dfrac = measure(ne, s)
            gains.append(gain)
            rows.append([ne, epg, s, raw_max, dedup_max, gain, dfrac])
            print(f"{ne:>6} | {epg:>5} | {s:>4} | {raw_max:>5} | "
                  f"{dedup_max:>5} | {gain:>5.2f}% | {dfrac:>5.2f}%")
        summ[ne] = (epg, sum(gains) / len(gains), min(gains), max(gains))
        print("-" * 60)

    print("\nSUMMARY (gain% across seeds):")
    print(f"{'n_exp':>6} | {'e/gpu':>5} | {'mean':>7} | {'min':>7} | {'max':>7}")
    print("-" * 48)
    for ne in EXPERT_COUNTS:
        epg, mean, lo, hi = summ[ne]
        print(f"{ne:>6} | {epg:>5} | {mean:>6.2f}% | {lo:>6.2f}% | {hi:>6.2f}%")

    csv_path = os.path.join(_HERE, "num_experts_sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_experts", "experts_per_gpu", "seed",
                    "raw_max", "dedup_max", "gain_pct", "dedup_frac_pct"])
        w.writerows(rows)
    print(f"\n[saved] {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
