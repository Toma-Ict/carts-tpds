"""
scale_smoke_test.py — Checks whether ASTRA-sim's congestion_aware analytical
backend can run a flat-Switch topology at N ranks, and how long it takes.
Pure smoke test: minimal 2-rank-style workload pattern extended to N ranks,
no CARTS demand/placement pipeline involved. Isolates "can ASTRA-sim itself
scale" from "can the CARTS pipeline scale" -- run this first.
"""
import os
import sys
import time
import json
import re
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from make_2rank_workload import write_2rank_workload
from carts_et_writer import ETBuilder
from carts_et_writer import ETBuilder

BASE = "/workspace/astra-sim/examples"
BIN = "./build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
MEM = "examples/remote_memory/analytical/no_memory_expansion.json"


def run_smoke(num_ranks):
    wl_dir = os.path.join(BASE, f"workload/microbenchmarks/all_to_all/scale_smoke_{num_ranks}")
    write_2rank_workload(wl_dir, 1048576, num_ranks=num_ranks)
    wl = f"examples/workload/microbenchmarks/all_to_all/scale_smoke_{num_ranks}/carts_wl"
    net = f"examples/network/analytical/Switch_{num_ranks}npus_slow.yml"
    if not os.path.exists(os.path.join(BASE, "..", net)) and not os.path.exists(net):
        print(f"[{num_ranks}] SKIP -- {net} does not exist, create it first")
        return None

    # Write minimal no-op .et per rank (gotcha #3: every rank named in the
    # custom_collectives system json needs a readable, non-empty .et file).
    et_dir = os.path.join(BASE, "system/custom_collectives/scale_smoke")
    os.makedirs(et_dir, exist_ok=True)
    for r in range(num_ranks):
        b = ETBuilder()
        b.comp(1)
        b.write(os.path.join(et_dir, f"scale_smoke.{r}.et"))

    t0 = time.time()
    p = subprocess.run([BIN, f"--workload-configuration={wl}",
                        "--system-configuration=examples/system/custom_collectives/scale_smoke.json",
                        f"--network-configuration={net}",
                        f"--remote-memory-configuration={MEM}"],
                       capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0
    return p, elapsed


def write_system_json():
    sysdir = os.path.join(BASE, "system/custom_collectives")
    prefix = "examples/system/custom_collectives/scale_smoke/scale_smoke"
    j = {"scheduling-policy": "FIFO", "preferred-dataset-splits": 1,
         "all-to-all-implementation-custom": [prefix], "local-mem-bw": 50}
    os.makedirs(sysdir, exist_ok=True)
    with open(os.path.join(sysdir, "scale_smoke.json"), "w") as f:
        json.dump(j, f, indent=2)
    # minimal no-op ET per rank isn't needed for this smoke test --
    # write_2rank_workload already emits a valid COMM_COLL_NODE workload,
    # and we are not using the custom_collectives ET path here at all;
    # the system json above is unused by this binary invocation but kept
    # for parity with the other scripts in case it's needed later.


def main():
    write_system_json()
    print("=" * 70)
    print("ASTRA-sim scale smoke test: flat Switch, congestion_aware backend")
    print("=" * 70)
    for n in [16, 32, 64, 128]:
        result = run_smoke(n)
        if result is None:
            continue
        p, elapsed = result
        if p.returncode != 0:
            print(f"  n={n:>4} | FAILED (returncode={p.returncode}) | {elapsed:.2f}s")
            print(f"    stderr tail: {p.stderr[-300:]}")
        else:
            nums = [int(m.split()[0]) for m in re.findall(r"\d+ cycles", p.stdout)]
            cyc = max(nums) if nums else None
            print(f"  n={n:>4} | OK | {elapsed:.2f}s | cycles={cyc}")


if __name__ == "__main__":
    main()
