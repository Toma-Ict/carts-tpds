"""
CARTS - Component 3: ET writer (official Chakra protobuf API)
=============================================================
Phase 4 of the CARTS implementation roadmap (Component 3 of 3) - HIGHEST RISK.

Turns the CARTS placement (Component 2) into per-rank Chakra .et files that
ASTRA-sim's custom collective API consumes.

Verified ET structure (from decoding examples/system/custom_collectives/...):
  - File starts with GlobalMetadata(version="0.0.4").
  - Then a sequence of Nodes. Each Node has id / name / type / data_deps /
    duration_micros / attr.
  - A relay's deduplicating aggregation is a chain:  RECV -> COMP -> SEND
      RECV (type COMM_RECV_NODE=6): attr comm_size, comm_src, comm_tag
      COMP (type COMP_NODE=4):       duration_micros = DPU reduce cost
      SEND (type COMM_SEND_NODE=5):  attr comm_size, comm_dst, comm_tag
    data_deps wires RECV -> COMP -> SEND in order.

SCOPE NOTE (honest): this is the MINIMAL-but-correct writer. It emits a clean,
encoder-valid RECV->COMP->SEND chain per (src,dst) flow assigned to a rank, so
we can prove the official encoder produces a file ASTRA-sim accepts. Full
all-to-all mesh wiring across all ranks is the next layer once the encoder is
proven against the simulator.

DPU cost model (duration_micros) is still a PLACEHOLDER:
    duration_micros = ceil(reduce_bytes / DPU_BYTES_PER_US)
Real DPU_BYTES_PER_US comes from BlueField-3 throughput calibration.

This module imports Chakra from astra-sim's extern/graph_frontend. Run it from
a machine where that import works (host carts env, after `pip install protobuf`).
"""

import os
import sys
import math
from dataclasses import dataclass

# --- locate the Chakra package inside astra-sim -----------------------------
# Default assumes the standard layout; override with CHAKRA_ROOT if needed.
_DEFAULT_CHAKRA = os.path.expanduser(
    "~/carts_project/astra-sim/extern/graph_frontend"
)
_CHAKRA_ROOT = os.environ.get("CHAKRA_ROOT", _DEFAULT_CHAKRA)
if _CHAKRA_ROOT not in sys.path:
    sys.path.insert(0, _CHAKRA_ROOT)

from chakra.schema.protobuf.et_def_pb2 import (   # noqa: E402
    GlobalMetadata,
    Node as ChakraNode,
    AttributeProto as ChakraAttr,
    COMM_SEND_NODE,
    COMM_RECV_NODE,
    COMP_NODE,
)
from chakra.src.third_party.utils.protolib import encodeMessage as encode_message  # noqa: E402

# comm_type ids ASTRA-sim expects in the attr (matches decoded examples):
#   send-direction = 5, recv-direction = 6
COMM_SEND_TYPE = 5
COMM_RECV_TYPE = 6


# --------------------------------------------------------------------------
@dataclass
class WriterConfig:
    token_bytes: int = 2048          # bytes per token (comm_size = load * token_bytes)
    dpu_bytes_per_us: int = 100_000  # PLACEHOLDER DPU throughput (bytes / microsecond)
                                     # -> real value from BlueField-3 calibration


# --------------------------------------------------------------------------
class ETBuilder:
    """Accumulates nodes for ONE rank and writes a valid .et file."""

    def __init__(self):
        self._nodes = []
        self._next_id = 0

    def _new(self, name, ntype):
        n = ChakraNode()
        n.id = self._next_id
        n.name = name
        n.type = ntype
        self._next_id += 1
        self._nodes.append(n)
        return n

    def recv(self, comm_size, comm_src, tag=0, dep=None):
        n = self._new(f"COMM_RECV_NODE_{comm_src}_{tag}", COMM_RECV_NODE)
        n.attr.append(ChakraAttr(name="comm_type", int64_val=COMM_RECV_TYPE))
        n.attr.append(ChakraAttr(name="comm_size", int64_val=int(comm_size)))
        n.attr.append(ChakraAttr(name="comm_src", int32_val=int(comm_src)))
        n.attr.append(ChakraAttr(name="comm_tag", int32_val=int(tag)))
        if dep is not None:
            n.data_deps.append(dep)
        return n.id

    def comp(self, duration_micros, dep=None):
        n = self._new("COMP_NODE_dpu_reduce", COMP_NODE)
        n.duration_micros = int(duration_micros)
        if dep is not None:
            n.data_deps.append(dep)
        return n.id

    def send(self, comm_size, comm_dst, tag=0, dep=None):
        n = self._new(f"COMM_SEND_NODE_{comm_dst}_{tag}", COMM_SEND_NODE)
        n.attr.append(ChakraAttr(name="comm_type", int64_val=COMM_SEND_TYPE))
        n.attr.append(ChakraAttr(name="comm_size", int64_val=int(comm_size)))
        n.attr.append(ChakraAttr(name="comm_dst", int32_val=int(comm_dst)))
        n.attr.append(ChakraAttr(name="comm_tag", int32_val=int(tag)))
        if dep is not None:
            n.data_deps.append(dep)
        return n.id

    def write(self, path):
        with open(path, "wb") as f:
            encode_message(f, GlobalMetadata(version="0.0.4"))
            for n in self._nodes:
                encode_message(f, n)
        return path


# --------------------------------------------------------------------------
def dpu_duration_micros(reduce_bytes, cfg: WriterConfig):
    """PLACEHOLDER DPU reduce cost model."""
    return max(1, math.ceil(reduce_bytes / cfg.dpu_bytes_per_us))


def write_relay_chain_et(path, comm_size_bytes, comm_src, comm_dst,
                         cfg: WriterConfig, tag=0):
    """Emit one RECV -> COMP -> SEND deduplicating relay chain to `path`.

    comm_size_bytes : deduped payload size on the inter-domain link
    comm_src/dst    : the ranks this relay receives from / sends to
    """
    b = ETBuilder()
    r = b.recv(comm_size_bytes, comm_src, tag=tag)
    c = b.comp(dpu_duration_micros(comm_size_bytes, cfg), dep=r)
    b.send(comm_size_bytes, comm_dst, tag=tag, dep=c)
    return b.write(path)


# --------------------------------------------------------------------------
def _roundtrip_check(path):
    """Read the file back with the Chakra decoder and report the node chain."""
    from chakra.src.third_party.utils.protolib import decodeMessage as decode_message
    md = GlobalMetadata()
    nodes = []
    with open(path, "rb") as f:
        decode_message(f, md)
        while True:
            n = ChakraNode()
            if not decode_message(f, n):
                break
            nodes.append(n)
    return md, nodes


def selftest():
    print("=" * 64)
    print("SELF-TEST  (write a relay chain, read it back)")
    print("=" * 64)
    cfg = WriterConfig()

    # Pretend: relay aggregates a deduped pair of 218 distinct tokens.
    load_tokens = 218
    comm_size = load_tokens * cfg.token_bytes
    out = "/tmp/carts_relay_test.0.et"
    write_relay_chain_et(out, comm_size, comm_src=3, comm_dst=9, cfg=cfg)
    print(f"  wrote: {out}  ({os.path.getsize(out)} bytes)")
    print(f"  comm_size = {load_tokens} tokens x {cfg.token_bytes} B = {comm_size} B")
    print(f"  duration_micros (DPU) = {dpu_duration_micros(comm_size, cfg)} us")

    md, nodes = _roundtrip_check(out)
    print(f"\n  decoded version : {md.version}")
    print(f"  decoded nodes   : {len(nodes)}")
    type_name = {COMM_RECV_NODE: "RECV", COMP_NODE: "COMP", COMM_SEND_NODE: "SEND"}
    for n in nodes:
        deps = list(n.data_deps)
        extra = f" dur={n.duration_micros}us" if n.type == COMP_NODE else ""
        sz = next((a.int64_val for a in n.attr if a.name == "comm_size"), None)
        szs = f" size={sz}B" if sz is not None else ""
        print(f"    id={n.id} {type_name.get(n.type,'?'):4} deps={deps}{szs}{extra}")

    # Assertions: exactly RECV->COMP->SEND, wired in order.
    assert len(nodes) == 3, "expected 3 nodes"
    assert nodes[0].type == COMM_RECV_NODE
    assert nodes[1].type == COMP_NODE and list(nodes[1].data_deps) == [0]
    assert nodes[2].type == COMM_SEND_NODE and list(nodes[2].data_deps) == [1]
    print("\n  Round-trip PASS: RECV -> COMP -> SEND wired correctly.\n")


if __name__ == "__main__":
    selftest()
