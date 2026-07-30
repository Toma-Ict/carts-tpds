import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from carts_et_writer import (
    GlobalMetadata, ChakraNode, ChakraAttr, encode_message,
)
ALL_TO_ALL = 6
from chakra.schema.protobuf.et_def_pb2 import COMM_COLL_NODE


def write_2rank_workload(out_dir, comm_size_bytes, num_ranks=2):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for r in range(num_ranks):
        n = ChakraNode()
        n.id = 0
        n.name = f"carts_all_to_all_{num_ranks}npus"
        n.type = COMM_COLL_NODE
        n.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        n.attr.append(ChakraAttr(name="comm_type", int64_val=ALL_TO_ALL))
        n.attr.append(ChakraAttr(name="comm_size", int64_val=int(comm_size_bytes)))
        p = os.path.join(out_dir, f"carts_wl.{r}.et")
        with open(p, "wb") as f:
            encode_message(f, GlobalMetadata(version="0.0.4"))
            encode_message(f, n)
        paths.append(p)
    return paths


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "/workspace/astra-sim/examples/workload/microbenchmarks/all_to_all/carts_2npus"
    comm_size = int(sys.argv[2]) if len(sys.argv) > 2 else 446464
    paths = write_2rank_workload(out_dir, comm_size)
    for p in paths:
        print(f"wrote: {p}  ({os.path.getsize(p)} bytes)")
    print(f"comm_size = {comm_size} bytes")
    print(f"\nWORKLOAD prefix for run command:\n  {os.path.join(out_dir, 'carts_wl')}")
