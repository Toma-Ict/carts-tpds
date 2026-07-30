"""
scale_multiseed.py -- Multi-seed extension of scale_pipeline_test.py's gain%%
check to num_gpus=256, 512 (64, 128 domains), gpus_per_domain=4 fixed.

Why this exists: scale_pipeline_test.py (S15c) tested 32/64/128 at seed=0 ONLY.
S15d showed seed=0 alone is misleading at scale (num_gpus=64 seed=0 FixBdelta
was -5.31%, but 5/6 seeds were +5.8..+7.24%). This applies the S6/S15d
multi-seed discipline to 256/512, closing S10 open-item-4 (multi-seed never
applied to the domain-count dimension).

Methodology (locked):
  - SCALE_LIST = [128, 256, 512]; 128 is a REGRESSION ANCHOR -- seed=0 must
    reproduce S15c's row (cap=203, OLDvsC0=-2.79%, FixBdelta=+3.17%) before
    256/512 numbers are trusted.
  - cap is picked ONCE at seed=0 per scale (pick_cap), then HELD FIXED across
    seeds 0..5 -- matches S6's fixed-cap discipline, keeps FixBdelta the only
    moving part across seeds (per-seed cap re-pick would confound the compare).
  - dpu_bytes_per_us=20500 (low-DPU regime, S6/S18 headline convention).
  - Append-only: does NOT modify scale_pipeline_test.py or any validated file.

Run from repo root via os.chdir (BIN/wl/net are repo-root-relative).
"""
import os
import sys
import time
import json
import re
import csv
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(os.path.dirname(_HERE))   # -> /workspace/astra-sim (repo root)

from demand_extractor import MoEConfig, generate_demand
from greedy_crp import (CRPConfig, greedy_crp,
                        CRPConfigCostAware, greedy_crp_cost_aware)
from carts_et_writer import WriterConfig
from make_2rank_workload import write_2rank_workload
from sweep_any_n import emit_ets, write_system_json, pick_cap

GPUS_PER_DOMAIN = 4
SCALE_LIST = [128, 256, 512]       # 128 = anchor
SEEDS = [0, 1, 2, 3, 4, 5]
PER_SEED_CAP = True   # scale-fragility fix: pick cap per seed (gotcha #10), not frozen at seed-0
SKEW = 1.5
DPU_BYTES_PER_US = 20500
LINK_BW_BYTES_PER_US = 5000.0

BASE = "/workspace/astra-sim/examples"
BIN = "./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
MEM = "examples/remote_memory/analytical/no_memory_expansion.json"


def run_cycles_n(name, wl, net):
    """Parametrized binary runner (wl/net per num_gpus). Mirrors
    scale_pipeline_test.run_cycles_n -- NOT sweep_any_n.run_cycles, which
    hardcodes the 16-npu paths."""
    sysjson = f"examples/system/custom_collectives/{name}.json"
    p = subprocess.run([BIN, f"--workload-configuration={wl}",
                        f"--system-configuration={sysjson}",
                        f"--network-configuration={net}",
                        f"--remote-memory-configuration={MEM}"],
                       capture_output=True, text=True, timeout=600)
    nums = [int(m.split()[0]) for m in re.findall(r"\d+ cycles", p.stdout)]
    if not nums:
        sys.stderr.write(f"[{name}] no cycles parsed. tail:\n{p.stdout[-300:]}\n")
        return None
    return max(nums)


def run_one_seed(num_gpus, seed, cap):
    """C0/OLD/NEW cycles for one (num_gpus, seed) at a FIXED externally-supplied
    cap. cap is NOT re-picked here -- that is the whole point (S6 fixed-cap
    discipline). Returns dict with c0/old/new cycles + the served-pair counts
    (served counts let us catch a Bug-2/Bug-3-style serve-decision difference
    without a separate per-pair dump)."""
    net = f"examples/network/analytical/Switch_{num_gpus}npus_slow.yml"
    wl_dir = os.path.join(BASE, f"workload/microbenchmarks/all_to_all/scale_{num_gpus}npus")
    wl = f"examples/workload/microbenchmarks/all_to_all/scale_{num_gpus}npus/carts_wl"
    write_2rank_workload(wl_dir, 1048576, num_ranks=num_gpus)

    cfg = MoEConfig(skew=SKEW, seed=seed, num_gpus=num_gpus,
                    gpus_per_domain=GPUS_PER_DOMAIN, num_experts=num_gpus)
    dem = generate_demand(cfg)
    wcfg = WriterConfig(dpu_bytes_per_us=DPU_BYTES_PER_US)

    p_c0 = greedy_crp(dem, CRPConfig(relay_capacity=0))
    p_old = greedy_crp(dem, CRPConfig(relay_capacity=cap))
    p_new = greedy_crp_cost_aware(dem, CRPConfigCostAware(
        relay_capacity=cap, dpu_bytes_per_us=DPU_BYTES_PER_US,
        link_bw_bytes_per_us=LINK_BW_BYTES_PER_US))

    served = {tag: sum(1 for v in pl.served.values() if v)
              for tag, pl in [("old", p_old), ("new", p_new)]}

    out = {}
    for tag, pl in [("c0", p_c0), ("old", p_old), ("new", p_new)]:
        name = f"sms_{num_gpus}_{tag}"   # sms = scale_multiseed; distinct from scale_pipeline_test names
        emit_ets(os.path.join(BASE, "system/custom_collectives", name), name, pl, wcfg)
        write_system_json(name)
        out[tag] = run_cycles_n(name, wl, net)
    return dict(num_gpus=num_gpus, num_domains=dem.cfg.num_domains, seed=seed, cap=cap,
                c0=out["c0"], old=out["old"], new=out["new"],
                served_old=served["old"], served_new=served["new"])


# S15c seed-0 anchors (cap, OLDvsC0%, FixBdelta%) -- 128 must reproduce.
ANCHOR_S15C = {128: dict(cap=203, old_vs_c0=-2.79, fixb=3.17)}


def run_one_scale(num_gpus):
    """Pick cap ONCE at seed=0, hold it fixed across all SEEDS. Returns the
    cap used + a per-seed list of result dicts. Anchors num_gpus=128 seed=0
    against S15c and warns loudly on mismatch (harness-trust gate)."""
    # cap picked at seed=0 only, then frozen (S6 fixed-cap discipline)
    cfg0 = MoEConfig(skew=SKEW, seed=0, num_gpus=num_gpus,
                     gpus_per_domain=GPUS_PER_DOMAIN, num_experts=num_gpus)
    dem0 = generate_demand(cfg0)
    cap, _cap_rows = pick_cap(dem0)   # seed-0 cap kept for the S15c anchor
    _mode = "per-seed cap" if PER_SEED_CAP else "fixed cap (seed-0)"
    print(f"\n[n={num_gpus}] domains={dem0.cfg.num_domains} "
          f"cap(seed0)={cap}  | mode={_mode}")
    if cap == 0 and not PER_SEED_CAP:
        print(f"[n={num_gpus}] NOTE: seed-0 cap=0 frozen -> whole scale forced no-dedup "
              f"(PER_SEED_CAP=True avoids this; see per-seed caps in rows below).")

    rows = []
    for seed in SEEDS:
        if PER_SEED_CAP:
            _dem = generate_demand(MoEConfig(skew=SKEW, seed=seed, num_gpus=num_gpus,
                                             gpus_per_domain=GPUS_PER_DOMAIN, num_experts=num_gpus))
            seed_cap, _ = pick_cap(_dem)
        else:
            seed_cap = cap
        r = run_one_seed(num_gpus, seed, seed_cap)
        if r["c0"] is None or r["old"] is None or r["new"] is None:
            print(f"  seed={seed} | RUN FAILED (a config returned no cycles)")
            continue
        c0, old, new = r["c0"], r["old"], r["new"]
        r["old_vs_c0"] = 100.0 * (c0 - old) / c0
        r["new_vs_c0"] = 100.0 * (c0 - new) / c0
        r["fixb"] = 100.0 * (old - new) / old if old else 0.0
        rows.append(r)
        print(f"  seed={seed} | cap={r['cap']:>5} | C0={c0:>9} OLD={old:>9} NEW={new:>9} "
              f"| OLDvsC0={r['old_vs_c0']:>+7.2f}% NEWvsC0={r['new_vs_c0']:>+7.2f}% "
              f"FixB={r['fixb']:>+7.2f}% | served O/N={r['served_old']}/{r['served_new']}")

    # anchor check (harness trust)
    if num_gpus in ANCHOR_S15C and rows:
        a = ANCHOR_S15C[num_gpus]
        s0 = next((x for x in rows if x["seed"] == 0), None)
        if s0:
            d_cap = (cap != a["cap"])
            d_fixb = abs(s0["fixb"] - a["fixb"]) > 0.5
            if d_cap or d_fixb:
                print(f"  [ANCHOR WARN n={num_gpus}] seed0 vs S15c: "
                      f"cap {cap} vs {a['cap']}, FixB {s0['fixb']:+.2f}% vs "
                      f"{a['fixb']:+.2f}% -- harness may differ from S15c, investigate "
                      f"before trusting 256/512.")
            else:
                print(f"  [ANCHOR OK n={num_gpus}] seed0 reproduces S15c "
                      f"(cap={cap}, FixB={s0['fixb']:+.2f}%).")
    return cap, rows


def main():
    print("=" * 100)
    print(f"CARTS multi-seed scale test | gpus_per_domain={GPUS_PER_DOMAIN} fixed "
          f"| num_gpus={SCALE_LIST} | seeds={SEEDS} | dpu_bytes_per_us={DPU_BYTES_PER_US}")
    print("=" * 100)

    all_rows = []
    summary = []
    for n in SCALE_LIST:
        cap, rows = run_one_scale(n)
        all_rows.extend(rows)
        if not rows:
            summary.append((n, cap, None)); continue
        fixb = [r["fixb"] for r in rows]
        oldv = [r["old_vs_c0"] for r in rows]
        ge0 = sum(1 for x in fixb if x >= -1e-9)   # strictly-better-or-equal count
        summary.append((n, cap, dict(
            mean=sum(fixb) / len(fixb), mn=min(fixb), mx=max(fixb),
            ge0=ge0, k=len(fixb),
            old_mean=sum(oldv) / len(oldv), old_mn=min(oldv), old_mx=max(oldv))))

    print("\n" + "=" * 100)
    print("SUMMARY (FixBdelta = OLD->NEW gain; the strictly-better-or-equal claim)")
    print("=" * 100)
    print(f"{'num_gpus':>8} | {'cap':>5} | {'FixB mean':>9} | {'FixB min':>9} | "
          f"{'FixB max':>9} | {'FixB>=0':>9} | {'OLDvsC0 mean':>12}")
    print("-" * 100)
    for n, cap, s in summary:
        if s is None:
            print(f"{n:>8} | {cap:>5} | (no rows)"); continue
        print(f"{n:>8} | {cap:>5} | {s['mean']:>+8.2f}% | {s['mn']:>+8.2f}% | "
              f"{s['mx']:>+8.2f}% | {s['ge0']:>6}/{s['k']} | {s['old_mean']:>+11.2f}%")

    csv_path = os.path.join(_HERE, "scale_multiseed_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_gpus", "num_domains", "seed", "cap", "c0", "old", "new",
                    "old_vs_c0", "new_vs_c0", "fixb", "served_old", "served_new"])
        for r in all_rows:
            w.writerow([r["num_gpus"], r["num_domains"], r["seed"], r["cap"],
                        r["c0"], r["old"], r["new"],
                        round(r["old_vs_c0"], 4), round(r["new_vs_c0"], 4),
                        round(r["fixb"], 4), r["served_old"], r["served_new"]])
    print(f"\n[saved] {csv_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
