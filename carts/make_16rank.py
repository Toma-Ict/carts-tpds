import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from demand_extractor import MoEConfig, generate_demand
from greedy_crp import CRPConfig, greedy_crp
from carts_et_writer import ETBuilder, WriterConfig, dpu_duration_micros
from make_2rank_workload import write_2rank_workload


def emit_ets(out_dir, prefix, placement, wcfg, mode):
    cfg = placement.demand.cfg
    D, G, N = cfg.num_domains, cfg.gpus_per_domain, cfg.num_gpus
    gw = {d: d * G for d in range(D)}
    builders = {r: ETBuilder() for r in range(N)}
    pairs = [(i, j) for i in range(D) for j in range(D) if i != j]
    tag = {p: k for k, p in enumerate(pairs)}
    for (di, dj) in pairs:
        load = placement.load[(di, dj)]
        size = load * wcfg.token_bytes
        t, gsrc, gdst = tag[(di, dj)], gw[di], gw[dj]
        bs, bd = builders[gsrc], builders[gdst]
        if mode == "C3" and placement.served[(di, dj)]:
            c = bs.comp(dpu_duration_micros(size, wcfg))
            bs.send(size, comm_dst=gdst, tag=t, dep=c)
        else:
            bs.send(size, comm_dst=gdst, tag=t)
        bd.recv(size, comm_src=gsrc, tag=t)
    os.makedirs(out_dir, exist_ok=True)
    for r in range(N):
        b = builders[r]
        if len(b._nodes) == 0:
            b.comp(1)
        b.write(os.path.join(out_dir, f"{prefix}.{r}.et"))
    return N


def gateway_incoming(placement):
    D = placement.demand.cfg.num_domains
    return {dj: sum(placement.load[(di, dj)] for di in range(D) if di != dj)
            for dj in range(D)}


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser("~/carts_project/astra-sim/examples")
    wcfg = WriterConfig()
    dem = generate_demand(MoEConfig(skew=1.5, seed=0))
    c0 = greedy_crp(dem, CRPConfig(relay_capacity=0))
    c3 = greedy_crp(dem, CRPConfig(relay_capacity=10**9))
    sys_dir = os.path.join(base, "system/custom_collectives")
    emit_ets(os.path.join(sys_dir, "carts16_c0"), "carts16_c0", c0, wcfg, "C0")
    emit_ets(os.path.join(sys_dir, "carts16_c3"), "carts16_c3", c3, wcfg, "C3")
    wl_dir = os.path.join(base, "workload/microbenchmarks/all_to_all/carts_16npus")
    write_2rank_workload(wl_dir, 1048576, num_ranks=16)
    inc0, inc3 = gateway_incoming(c0), gateway_incoming(c3)
    print("=" * 60)
    print("Predicted incast (incoming tokens per dest-domain gateway)")
    print("=" * 60)
    print(f"  C0 baseline : {inc0}  -> bottleneck {max(inc0.values())}")
    print(f"  C3 CARTS    : {inc3}  -> bottleneck {max(inc3.values())}")
    drop = max(inc0.values()) - max(inc3.values())
    print(f"  bottleneck reduction: {drop} tokens ({100*drop/max(inc0.values()):.1f}%)")
    print()
    print("ET prefixes for system JSON:")
    print(f"  C0: {sys_dir}/carts16_c0/carts16_c0")
    print(f"  C3: {sys_dir}/carts16_c3/carts16_c3")
    print(f"WORKLOAD prefix: {wl_dir}/carts_wl")
