#!/usr/bin/env python3
"""
verify_fig5_timeline.py -- READ-ONLY. Establishes the exact per-pair numbers
behind the proposed Fig 5 timeline before any figure is drawn.

Answers three questions the timeline sketch assumes:
  (1) is the C0 bottleneck pair the same pair as the C3c bottleneck pair?
  (2) is the C3c bottleneck pair actually SERVED (i.e. does it carry a COMP)?
  (3) does dpu_duration_micros equal size/dpu_rate, or is there rounding?

Writes nothing, simulates nothing.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from demand_extractor import MoEConfig, generate_demand
from greedy_crp import (CRPConfig, greedy_crp,
                        CRPConfigCostAware, greedy_crp_cost_aware)
from carts_et_writer import WriterConfig, dpu_duration_micros
from sweep_any_n import pick_cap

GPD, NUM_GPUS, SEED, SKEW = 8, 32, 0, 1.5
DPU, BW, TOKEN_BYTES = 20500, 5, 2048
LINK_BPUS = BW * 1000.0


def argmax_pair(pl):
    return max(pl.inter_pairs(), key=lambda p: pl.load[p])


def main():
    dem = generate_demand(MoEConfig(skew=SKEW, seed=SEED, num_gpus=NUM_GPUS,
                                    gpus_per_domain=GPD, num_experts=NUM_GPUS))
    cap, _ = pick_cap(dem)
    wcfg = WriterConfig(dpu_bytes_per_us=DPU)
    c0 = greedy_crp(dem, CRPConfig(relay_capacity=0))
    c3 = greedy_crp_cost_aware(dem, CRPConfigCostAware(
        relay_capacity=cap, dpu_bytes_per_us=DPU,
        link_bw_bytes_per_us=LINK_BPUS))

    p0, p3 = argmax_pair(c0), argmax_pair(c3)
    print("seed=%d gpd=%d cap=%d  bw=%d GB/s  dpu=%d B/us" % (SEED, GPD, cap, BW, DPU))
    print("\n(1) BOTTLENECK PAIR IDENTITY")
    print("    C0  argmax pair %s  load=%s" % (p0, c0.load[p0]))
    print("    C3c argmax pair %s  load=%s  served=%s"
          % (p3, c3.load[p3], c3.served[p3]))
    print("    same pair: %s" % (p0 == p3))

    print("\n(2) THAT PAIR IN DETAIL")
    for tag, p in (("C0 bottleneck", p0), ("C3c bottleneck", p3)):
        print("    %-16s %s  lam=%s unique=%s  c3c_served=%s c3c_load=%s"
              % (tag, p, dem.lam(*p), dem.unique(*p), c3.served[p], c3.load[p]))

    print("\n(3) TIMES FOR THE TIMELINE (bottleneck pair)")
    raw, new = float(c0.load[p0]), float(c3.load[p3])
    t_raw = raw * TOKEN_BYTES / LINK_BPUS
    t_new = new * TOKEN_BYTES / LINK_BPUS
    comp = dpu_duration_micros(int(new * TOKEN_BYTES), wcfg)
    naive = new * TOKEN_BYTES / DPU
    print("    C0  transfer : %8.1f us   (%d copies)" % (t_raw, raw))
    print("    C3c transfer : %8.1f us   (%d copies)" % (t_new, new))
    print("    C3c COMP     : %8s      (naive size/rate = %.1f us)" % (comp, naive))
    print("    link bytes   : %+.2f %%" % (100.0 * (raw - new) / raw))

    print("\n(4) TOP 5 C0 PAIRS AND THEIR C3c FATE")
    for p in sorted(c0.inter_pairs(), key=lambda q: -c0.load[q])[:5]:
        print("    %s  c0=%5s  lam=%5s uniq=%5s  ->  served=%-5s c3c_load=%5s"
              % (p, c0.load[p], dem.lam(*p), dem.unique(*p),
                 c3.served[p], c3.load[p]))

    ns = sum(1 for p in c3.inter_pairs() if c3.served[p])
    print("\n    served pairs: %d of %d" % (ns, len(c3.inter_pairs())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
