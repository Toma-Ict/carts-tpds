#!/usr/bin/env python3
# verify_table5_numbers.py -- READ-ONLY.
# Verifies Table V min/max and the Sec VII-C adversarial-seed figures
# against the locked CSVs. Writes nothing, modifies nothing.
import csv
from collections import defaultdict

GPD8 = "scale_multiseed_gpd8_results.csv"
SENS = "sensitivity_gpd8_results.csv"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def show_formats():
    rows = load(SENS)
    print("distinct bw_gbs:", sorted({r['bw_gbs'] for r in rows}))
    print("distinct dpu:   ", sorted({r['dpu_bytes_per_us'] for r in rows}))
    print("sample new_vs_c0:", [r['new_vs_c0'] for r in rows[:3]])


def summarize_headline():
    rows = load(GPD8)
    buckets = defaultdict(list)
    for r in rows:
        key = (int(r["num_gpus"]), int(r["num_domains"]))
        buckets[key].append((int(r["seed"]), float(r["new_vs_c0"])))
    print()
    print("=== Table V: headline gpd=8, column new_vs_c0 ===")
    print(f"{'GPUs':>5} {'dom':>4} {'mean':>8} {'min':>8} {'max':>8} "
          f"{'>=C0':>6}  worst")
    for key in sorted(buckets):
        vals = buckets[key]
        g = [v for _, v in vals]
        n_ok = sum(1 for v in g if v >= 0)
        worst = min(vals, key=lambda t: t[1])
        print(f"{key[0]:>5} {key[1]:>4} {sum(g)/len(g):>8.2f} "
              f"{min(g):>8.2f} {max(g):>8.2f} "
              f"{n_ok:>3}/{len(g)}  seed {worst[0]}")
    return buckets


def check_sensitivity():
    rows = load(SENS)
    print()
    print("=== Sec VII-C: CA per-seed at bw=5, dpu in {50000, 83900} ===")
    for dpu in ("50000", "83900"):
        sel = [r for r in rows
               if float(r["bw_gbs"]) == 5.0
               and r["dpu_bytes_per_us"].strip() == dpu]
        sel.sort(key=lambda r: int(r["seed"]))
        if not sel:
            print(f"  dpu={dpu}: NO ROWS (check formats printed above)")
            continue
        vals = [float(r["new_vs_c0"]) for r in sel]
        print("  dpu=%s: %s" % (dpu, ", ".join(
            "s%s=%+.2f" % (r['seed'], float(r['new_vs_c0'])) for r in sel)))
        print("    mean=%+.2f  min=%+.2f  n>=0: %d/%d"
              % (sum(vals)/len(vals), min(vals),
                 sum(1 for v in vals if v >= 0), len(vals)))


if __name__ == "__main__":
    show_formats()
    b = summarize_headline()
    check_sensitivity()
    m32 = [v for _, v in b[(32, 4)]]
    print()
    print("ANCHOR CHECK 32/4 mean = %.2f  (paper claims +10.3)"
          % (sum(m32)/len(m32)))
