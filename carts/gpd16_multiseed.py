"""
gpd16_multiseed.py -- MC1 experiment: does the sweet-spot gain persist at
NVL72-class domain widths? gpd=16 at num_gpus=64 (4 domains) and 128
(8 domains), seeds 0-5, bw=5, dpu=20500 -- the exact S24 protocol with
GPUS_PER_DOMAIN=16.

Pre-registered M1 prediction (closed form, 200-seed): load-gamma ceiling
~17.6% at 64/4dom (same class as gpd=8/32's ~17%), ~8.7% at 128/8dom.
Expected cycles outcome: PERSISTENCE of the ~+10% class at 4 domains,
not growth (E=N scaling offsets larger-domain pooling).

Append-only driver: overrides scale_multiseed globals at call time (S24
pattern), edits no file. Run:
  CHAKRA_ROOT=/workspace/astra-sim/extern/graph_frontend python3 gpd16_multiseed.py
"""
import os, sys, csv

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.chdir(os.path.dirname(_HERE))

import scale_multiseed as sms

sms.GPUS_PER_DOMAIN = 16
sms.PER_SEED_CAP = True
sms.ANCHOR_S15C = {}          # gpd=4 anchor does not apply (S24 precedent)
sms.DPU_BYTES_PER_US = 20500
sms.LINK_BW_BYTES_PER_US = 5000.0
SCALES = [64, 128]


def main():
    all_rows = []
    for n in SCALES:
        cap, rows = sms.run_one_scale(n)
        all_rows.extend(rows)
    out = os.path.join(_HERE, "scale_multiseed_gpd16_results.csv")
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(all_rows)
        print(f"\n[saved] {out} ({len(all_rows)} rows)")
    print(f"\n{'n':>4} {'dom':>4} | {'NEWvsC0 mean':>12} | {'NEW>=C0':>7} "
          f"| {'OLDvsC0 mean':>12} | {'cap=0':>5}")
    for n in SCALES:
        rs = [r for r in all_rows if r["num_gpus"] == n]
        if not rs:
            continue
        nm = sum(r["new_vs_c0"] for r in rs) / len(rs)
        om = sum(r["old_vs_c0"] for r in rs) / len(rs)
        ge = sum(1 for r in rs if r["new_vs_c0"] >= 0)
        cz = sum(1 for r in rs if r["cap"] == 0)
        print(f"{n:>4} {rs[0]['num_domains']:>4} | {nm:>+11.2f}% | {ge}/{len(rs)}"
              f"   | {om:>+11.2f}% | {cz}/{len(rs)}")


if __name__ == "__main__":
    main()
