"""
fixc_seed1_retest.py -- retest Fix C (greedy_crp_cost_aware_v3) against the
M3 section-4 forensic config (N=32, gpd=8, 4 domains, skew=1.5, bw=5 GB/s,
dpu=20500), seeds 0-5, per CARTS_Part9_Phase3-4_record.md section 9.7 item 3
("Fix C on seed 1 -- test against the 9.4/seed-1 case independently") and
the follow-up decision (this session) to check generalization before
locking a claim, per the project's multi-seed discipline (M3 section 8,
S15d).

This is a RETEST, not a redesign (confirmed scope this session): v3 is used
exactly as it stands from S17 (reverted, reference-only, not in the active
pipeline). v3 identifies its own global-bottleneck pair per seed via
argmax(raw_loads) -- this script does NOT hardcode (0,3); that pair is
seed=1-specific (M3 section 4) and would be wrong for other seeds.

Compares three configs per seed:
  C0    -- no relay, no dedup (greedy_crp, relay_capacity=0)
  NEW   -- production Fix B (greedy_crp_cost_aware)
  FIXC  -- v3, global-bottleneck-first (greedy_crp_cost_aware_v3)

Reports per seed: the bottleneck pair v3 selected, whether NEW serves it,
and system-max cycles gain for NEW and FIXC vs C0 (per-rank max, matching
CARTS_STATE.md gotcha #11).

Append-only: imports only, modifies nothing in any validated file.
Run from /workspace/astra-sim:
    CHAKRA_ROOT=/workspace/astra-sim/extern/graph_frontend python3 carts/fixc_seed1_retest.py
"""
import os, sys, re, subprocess
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import demand_extractor as D
from greedy_crp import (CRPConfig, CRPConfigCostAware, greedy_crp,
                         greedy_crp_cost_aware, greedy_crp_cost_aware_v3)
from carts_et_writer import WriterConfig
from sweep_any_n import emit_ets, write_system_json, pick_cap

NUM_GPUS, GPD, NUM_EXPERTS, TOP_K, TPG, SKEW = 32, 8, 32, 2, 64, 1.5
DPU_BYTES_PER_US = 20500
BW_GB_S = 5
LINK_BW_BYTES_PER_US = BW_GB_S * 1000.0
SEEDS = range(6)

BASE = "/workspace/astra-sim/examples"
BIN = "./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
WL  = "examples/workload/microbenchmarks/all_to_all/scale_32npus/carts_wl"
NET = "examples/network/analytical/Switch_32npus_slow.yml"
MEM = "examples/remote_memory/analytical/no_memory_expansion.json"


def run_cycles_n(name, wl, net):
    sysjson = f"examples/system/custom_collectives/{name}.json"
    p = subprocess.run([BIN, f"--workload-configuration={wl}",
                        f"--system-configuration={sysjson}",
                        f"--network-configuration={net}",
                        f"--remote-memory-configuration={MEM}"],
                       capture_output=True, text=True, timeout=600)
    nums = [int(m.split()[0]) for m in re.findall(r"\d+ cycles", p.stdout)]
    if not nums:
        sys.stderr.write(f"[{name}] no cycles parsed. tail:\n{p.stdout[-300:]}\n")
        return None, p.stdout
    return max(nums), p.stdout


def per_rank_cycles(stdout_text):
    """sys[rank] N cycles -- CARTS_STATE.md gotcha #11: always check per-rank
    breakdown before trusting an aggregate."""
    out = {}
    for m in re.finditer(r"sys\[(\d+)\].*?(\d+)\s+cycles", stdout_text):
        out[int(m.group(1))] = int(m.group(2))
    return out


def main():
    wcfg = WriterConfig(dpu_bytes_per_us=DPU_BYTES_PER_US)
    SYSDIR = os.path.join(BASE, "system/custom_collectives")

    print(f"{'seed':>4} {'bneck':>8} {'raw':>5} {'uniq':>5} {'NEWserv':>8} {'FIXCserv':>9} "
          f"{'C0':>8} {'NEW':>8} {'FIXC':>8} {'gainNEW':>9} {'gainFIXC':>9} {'better':>8}")
    rows = []
    for seed in SEEDS:
        cfg = D.MoEConfig(num_gpus=NUM_GPUS, gpus_per_domain=GPD,
                           num_experts=NUM_EXPERTS, top_k=TOP_K,
                           tokens_per_gpu=TPG, skew=SKEW, seed=seed)
        res = D.generate_demand(cfg)
        cap, _ = pick_cap(res)
        domains = cfg.num_domains

        # v3's own bottleneck selection -- NOT hardcoded, matches its
        # internal argmax(raw_loads) over all directed pairs (greedy_crp.py
        # line 318), so this is exactly what v3 will force-serve this seed.
        all_pairs = [(i, j) for i in range(domains) for j in range(domains) if i != j]
        raw_loads = {p: res.lam(*p) for p in all_pairs}
        bottleneck_pair = max(all_pairs, key=lambda p: raw_loads[p])
        raw_b = res.lam(*bottleneck_pair)
        unique_b = res.unique(*bottleneck_pair)

        p_c0 = greedy_crp(res, CRPConfig(relay_capacity=0))
        crp_new = CRPConfigCostAware(relay_capacity=cap, dpu_bytes_per_us=DPU_BYTES_PER_US,
                                      link_bw_bytes_per_us=LINK_BW_BYTES_PER_US)
        p_new = greedy_crp_cost_aware(res, crp_new)
        crp_fixc = CRPConfigCostAware(relay_capacity=cap, dpu_bytes_per_us=DPU_BYTES_PER_US,
                                       link_bw_bytes_per_us=LINK_BW_BYTES_PER_US)
        p_fixc = greedy_crp_cost_aware_v3(res, crp_fixc)

        new_served = p_new.served.get(bottleneck_pair, False)
        fixc_served = p_fixc.served.get(bottleneck_pair, False)

        results = {}
        for tag, placement in [("c0", p_c0), ("new", p_new), ("fixc", p_fixc)]:
            name = f"fixc_multiseed_s{seed}_{tag}"
            emit_ets(os.path.join(SYSDIR, name), name, placement, wcfg)
            write_system_json(name)
            cycles_max, _ = run_cycles_n(name, WL, NET)
            results[tag] = cycles_max

        c0_max, new_max, fixc_max = results["c0"], results["new"], results["fixc"]
        if c0_max and new_max and fixc_max:
            gain_new = (c0_max - new_max) / c0_max * 100
            gain_fixc = (c0_max - fixc_max) / c0_max * 100
            better = ("FIXC" if fixc_max < new_max else
                      ("NEW" if fixc_max > new_max else "tie"))
        else:
            gain_new, gain_fixc, better = None, None, "N/A (parse fail)"

        rows.append((seed, bottleneck_pair, raw_b, unique_b, new_served, fixc_served,
                     c0_max, new_max, fixc_max, gain_new, gain_fixc, better))
        print(f"{seed:>4} {str(bottleneck_pair):>8} {raw_b:>5} {unique_b:>5} "
              f"{str(new_served):>8} {str(fixc_served):>9} {str(c0_max):>8} "
              f"{str(new_max):>8} {str(fixc_max):>8} "
              f"{(f'{gain_new:+.2f}' if gain_new is not None else 'N/A'):>9} "
              f"{(f'{gain_fixc:+.2f}' if gain_fixc is not None else 'N/A'):>9} {better:>8}")

    better_counts = {}
    for r in rows:
        better_counts[r[-1]] = better_counts.get(r[-1], 0) + 1
    print(f"\nFIXC vs NEW tally across {len(rows)} seeds: {better_counts}")
    return rows


if __name__ == "__main__":
    main()
