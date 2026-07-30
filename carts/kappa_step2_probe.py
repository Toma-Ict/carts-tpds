"""
kappa_step2_probe.py -- Step 2 (output-side): real ASTRA-sim run at the
headline config (N=32, gpd=8, 4 domains, skew=1.5, dpu=20500), bw in
{5,10,20,50} GB/s, seeds 0-5. At each (bw,seed), NEW and C0 are both run
first; j* is then picked as the gateway rank where NEW and C0 cycles
actually differ most (Option A2, confirmed this session -- avoids the
case found this session where the highest-incoming-load served domain
was tied/unaffected while the real dedup benefit landed at a different,
lower-ranked domain's gateway). Compares that rank's ACTUAL simulated
gain (NEW vs C0) against two predictions, both following M2's
gain = T0 - T1 form:
  gain_serial_pred  = T0 - (T1_link + Sum_T_dpu)    (additive: COMP fully exposed)
  gain_overlap_pred = T0 - max(T1_link, Sum_T_dpu)  (M2's roofline assumption)
Whichever actual gain is closer to settles the serial-vs-overlap question
flagged in M2 section 6, and gives the basis for re-deriving kappa honestly.

Reuses sweep_any_n.py's emit_ets()/write_system_json() (generic, no N=16
hardcode inside those two functions themselves) and scale_multiseed.py's /
sensitivity_gpd8.py's N=32 paths, BW_YAML mapping, link_bw_bytes_per_us
coupling rule, and run_cycles_n() pattern (all confirmed by user, this session).

Append-only: imports only, modifies nothing in any validated file.
Run from /workspace/astra-sim:
    CHAKRA_ROOT=/workspace/astra-sim/extern/graph_frontend python3 carts/kappa_step2_probe.py
"""
import os, sys, re, subprocess
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import demand_extractor as D
from greedy_crp import CRPConfig, CRPConfigCostAware, greedy_crp, greedy_crp_cost_aware
from carts_et_writer import WriterConfig, dpu_duration_micros
from sweep_any_n import emit_ets, write_system_json, pick_cap

NUM_GPUS, GPD, NUM_EXPERTS, TOP_K, TPG, SKEW = 32, 8, 32, 2, 64, 1.5
DPU_BYTES_PER_US = 20500
BW_GB_S_LIST = [5, 10, 20, 50]   # S25 headline-dpu row, confirmed scope (this session)

BASE = "/workspace/astra-sim/examples"
BIN = "./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
WL  = "examples/workload/microbenchmarks/all_to_all/scale_32npus/carts_wl"
MEM = "examples/remote_memory/analytical/no_memory_expansion.json"
SEEDS = range(6)
CLOCK_PERIOD_NS = 1   # astra-sim/common/Common.hh: CLOCK_PERIOD = 1 (1ns/cycle), confirmed this session

# bandwidth (GB/s) -> topology yaml, exact mapping from sensitivity_gpd8.py (confirmed this session)
BW_YAML = {5:  "examples/network/analytical/Switch_32npus_slow.yml",
           10: "examples/network/analytical/Switch_32npus_bw10.yml",
           20: "examples/network/analytical/Switch_32npus_bw20.yml",
           50: "examples/network/analytical/Switch_32npus_bw50.yml"}


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
    """sys[rank] N cycles -- parses the per-rank breakdown line (CARTS_STATE.md
    gotcha #11: always check per-rank breakdown before trusting an aggregate)."""
    out = {}
    for m in re.finditer(r"sys\[(\d+)\].*?(\d+)\s+cycles", stdout_text):
        out[int(m.group(1))] = int(m.group(2))
    return out


def headline_cfg(seed):
    return D.MoEConfig(num_gpus=NUM_GPUS, gpus_per_domain=GPD,
                        num_experts=NUM_EXPERTS, top_k=TOP_K,
                        tokens_per_gpu=TPG, skew=SKEW, seed=seed)


def main():
    cap_dpu = WriterConfig(dpu_bytes_per_us=DPU_BYTES_PER_US)
    print(f"{'bw':>4} {'seed':>4} {'jstar':>5} {'gw_rank':>7} {'n_served':>8} {'T0(us)':>9} "
          f"{'T1_ser':>9} {'T1_ovl':>9} {'gainSer':>9} {'gainOvl':>9} "
          f"{'NEW_gw':>9} {'C0_gw':>9} {'gainAct_us':>10} {'closer_to':>10}")
    rows = []
    skipped = []
    for bw in BW_GB_S_LIST:
        net = BW_YAML[bw]
        link_bw = bw * 1000.0  # GB/s -> bytes/us, exact sensitivity_gpd8.py pattern
        for seed in SEEDS:
            cfg = headline_cfg(seed)
            res = D.generate_demand(cfg)
            cap, _ = pick_cap(res)  # validated per-seed cap (CARTS_STATE.md gotcha #10)
            crp = CRPConfigCostAware(relay_capacity=cap, dpu_bytes_per_us=DPU_BYTES_PER_US,
                                      link_bw_bytes_per_us=link_bw)  # coupling rule, confirmed
            placement = greedy_crp_cost_aware(res, crp)
            c0_placement = greedy_crp(res, CRPConfig(relay_capacity=0))
            domains = cfg.num_domains  # = 4 at gpd=8, N=32

            # Run NEW and C0 FIRST (Option A2, confirmed this session): j* is
            # picked from where the cycles ACTUALLY differ, not guessed from
            # served-pair status beforehand. Avoids the case found this session
            # where the highest-incoming-load served domain was tied (no real
            # effect) while a different, lower-ranked domain's gateway actually
            # carried the dedup benefit.
            SYSDIR = os.path.join(BASE, "system/custom_collectives")
            wcfg = WriterConfig(dpu_bytes_per_us=DPU_BYTES_PER_US)
            name_new = f"kappa_probe_new_bw{bw}_s{seed}"
            emit_ets(os.path.join(SYSDIR, name_new), name_new, placement, wcfg)
            write_system_json(name_new)
            cycles_new, stdout_new = run_cycles_n(name_new, WL, net)
            ranks_new = per_rank_cycles(stdout_new)

            name_c0 = f"kappa_probe_c0_bw{bw}_s{seed}"
            emit_ets(os.path.join(SYSDIR, name_c0), name_c0, c0_placement, wcfg)
            write_system_json(name_c0)
            cycles_c0, stdout_c0 = run_cycles_n(name_c0, WL, net)
            ranks_c0 = per_rank_cycles(stdout_c0)

            if ranks_new is None or ranks_c0 is None or not ranks_new or not ranks_c0:
                skipped.append((bw, seed))
                continue

            # candidate gateway ranks (one per domain, per emit_ets()'s gw map)
            gw_ranks = [d * GPD for d in range(domains)]
            diffs = {gw: (ranks_c0.get(gw, 0) - ranks_new.get(gw, 0))
                     for gw in gw_ranks if gw in ranks_new and gw in ranks_c0}
            if not diffs:
                skipped.append((bw, seed))
                continue
            gw_rank = max(diffs, key=lambda gw: abs(diffs[gw]))
            jstar = gw_rank // GPD
            actual_new_gw = ranks_new[gw_rank]
            actual_c0_gw = ranks_c0[gw_rank]

            served_pairs = [(i, jstar) for i in range(domains)
                             if i != jstar and placement.served.get((i, jstar), False)]

            # NEW side: unique-byte transfer + DPU exposure at j* (served pairs only)
            sizes_unique = [placement.load[p] * cap_dpu.token_bytes for p in served_pairs]
            sum_unique_size = sum(sizes_unique)
            sum_t_dpu = sum(dpu_duration_micros(s, cap_dpu) for s in sizes_unique)
            t1_link = sum_unique_size / link_bw
            t1_serial = t1_link + sum_t_dpu
            t1_overlap = max(t1_link, sum_t_dpu)

            # C0 side: same pairs, RAW bytes, same gateway j* (the rank where NEW
            # vs C0 actually differs most), no DPU.
            sizes_raw = [c0_placement.load[p] * cap_dpu.token_bytes for p in served_pairs]
            sum_raw_size = sum(sizes_raw)
            t0 = sum_raw_size / link_bw

            gain_serial_pred = t0 - t1_serial    # M2's gain = T0 - T1, serial hypothesis
            gain_overlap_pred = t0 - t1_overlap  # M2's gain = T0 - T1, overlap hypothesis

            # cycles -> us: CLOCK_PERIOD_NS=1 (astra-sim/common/Common.hh), confirmed.
            gain_actual_cycles = actual_c0_gw - actual_new_gw  # positive if NEW faster
            gain_actual_us = gain_actual_cycles * CLOCK_PERIOD_NS / 1000.0
            closer = ("serial" if abs(gain_actual_us - gain_serial_pred) < abs(gain_actual_us - gain_overlap_pred)
                       else "overlap")

            rows.append((bw, seed, jstar, gw_rank, len(served_pairs), t0, t1_serial, t1_overlap,
                         gain_serial_pred, gain_overlap_pred, actual_new_gw, actual_c0_gw,
                         gain_actual_us, closer))
            print(f"{bw:>4} {seed:>4} {jstar:>5} {gw_rank:>7} {len(served_pairs):>8} {t0:>9.2f} "
                  f"{t1_serial:>9.2f} {t1_overlap:>9.2f} {gain_serial_pred:>9.2f} {gain_overlap_pred:>9.2f} "
                  f"{str(actual_new_gw):>9} {str(actual_c0_gw):>9} {str(gain_actual_us):>10} {closer:>10}")
    if skipped:
        sys.stderr.write(f"\nSkipped (bw,seed) pairs (NO domain has any served pair -- "
                          f"truly degenerate at this cell, not just at the busiest gateway): "
                          f"{skipped}\n")
    return rows


if __name__ == "__main__":
    main()
