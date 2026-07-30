"""
CARTS - minimal 2-rank custom collective (first sim-load smoke test)
====================================================================
Goal: prove the encoder -> ASTRA-sim pipeline end-to-end with the SMALLEST
possible VALID custom collective. Not about congestion yet - just "does the
simulator accept and run a CARTS-generated .et set without deadlock".

Matching rule learned from the ring example:
  every SEND dst=B on rank A must have a matching RECV src=A on rank B.

Layout (2 ranks, fully matched, no deadlock):
  rank 0:  SEND(dst=1) -> RECV(src=1)
  rank 1:  RECV(src=0) -> COMP(dpu) -> SEND(dst=0)

So rank0's first SEND matches rank1's RECV(src=0); rank1's SEND(dst=0) matches
rank0's RECV(src=1). COMP on rank1 carries the DPU reduce cost.

Writes:  <out_dir>/carts_a2a.0.et  and  carts_a2a.1.et
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from carts_et_writer import ETBuilder, WriterConfig, dpu_duration_micros


def write_minimal_2rank(out_dir, deduped_tokens=218, cfg=None, tag=0):
    cfg = cfg or WriterConfig()
    os.makedirs(out_dir, exist_ok=True)
    comm_size = deduped_tokens * cfg.token_bytes
    dur = dpu_duration_micros(comm_size, cfg)

    # ---- rank 0 : SEND to 1, then RECV from 1 ----
    b0 = ETBuilder()
    s0 = b0.send(comm_size, comm_dst=1, tag=tag)
    b0.recv(comm_size, comm_src=1, tag=tag, dep=s0)
    p0 = b0.write(os.path.join(out_dir, "carts_a2a.0.et"))

    # ---- rank 1 : RECV from 0 -> COMP(dpu) -> SEND to 0 ----
    b1 = ETBuilder()
    r1 = b1.recv(comm_size, comm_src=0, tag=tag)
    c1 = b1.comp(dur, dep=r1)
    b1.send(comm_size, comm_dst=0, tag=tag, dep=c1)
    p1 = b1.write(os.path.join(out_dir, "carts_a2a.1.et"))

    return p0, p1, comm_size, dur


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "/workspace/astra-sim/examples/system/custom_collectives/carts_a2a_2npus"
    p0, p1, sz, dur = write_minimal_2rank(out_dir)
    print(f"wrote: {p0}  ({os.path.getsize(p0)} bytes)")
    print(f"wrote: {p1}  ({os.path.getsize(p1)} bytes)")
    print(f"comm_size = {sz} bytes,  COMP duration = {dur} us")
    print(f"\nET prefix for system JSON:\n  {os.path.join(out_dir, 'carts_a2a')}")
