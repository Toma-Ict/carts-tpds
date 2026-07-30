import random
from demand_extractor import MoEConfig, generate_demand
def max_link_load(demand, cap, order, dedup, seed=0):
    n = demand.cfg.num_domains
    cap_left = {d: cap for d in range(n)}
    load = {}
    inter = [(i, j) for i in range(n) for j in range(n) if i != j]
    if order == "greedy":
        inter.sort(key=lambda p: demand.lam(*p), reverse=True)
    elif order == "random":
        random.Random(seed).shuffle(inter)
    for (di, dj) in inter:
        raw = demand.lam(di, dj)
        if raw == 0:
            load[(di, dj)] = 0
            continue
        u = demand.unique(di, dj)
        if dedup and cap_left[di] >= u:
            cap_left[di] -= u
            load[(di, dj)] = u
        else:
            load[(di, dj)] = raw
    return max(load[p] for p in inter)
def c1_random(demand, cap, trials=50):
    vals = [max_link_load(demand, cap, "random", True, seed=s) for s in range(trials)]
    return sum(vals) / len(vals), min(vals), max(vals)
def main():
    demand = generate_demand(MoEConfig(skew=1.5, seed=0))
    n = demand.cfg.num_domains
    inter = [(i, j) for i in range(n) for j in range(n) if i != j]
    raw_max = max(demand.lam(*p) for p in inter)
    dedup_max = max_link_load(demand, 10**9, "greedy", True)
    out_unique = {d: sum(demand.unique(d, j) for j in range(n) if j != d)
                  for d in range(n)}
    cap_hi = max(out_unique.values())
    print(f"demand: skew=1.5 seed=0, domains={n}")
    print(f"raw max link load (C0)   = {raw_max}   (anchor: expect 252)")
    print(f"full-dedup max link load = {dedup_max}   (anchor: expect 218)")
    print(f"per-domain outgoing Unique = {out_unique}")
    print(f"cap range probed: 0 .. {cap_hi}\n")
    caps = [0] + [int(cap_hi * f) for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    caps += [cap_hi, 10**9]
    caps = sorted(set(caps))
    print(f"{'cap':>10} | {'C0':>5} {'C1avg':>6} {'C1rng':>9} {'C2':>5} {'C3':>5} | "
          f"{'C1-C3':>6} | bites?")
    print("-" * 72)
    for cap in caps:
        c0 = max_link_load(demand, cap, "greedy", False)
        c2 = max_link_load(demand, cap, "greedy", False)
        c3 = max_link_load(demand, cap, "greedy", True)
        c1a, c1lo, c1hi = c1_random(demand, cap)
        gap = c1a - c3
        bites = "YES" if c3 < c1a - 1e-9 else ""
        tag = "inf" if cap >= 10**9 else str(cap)
        print(f"{tag:>10} | {c0:>5} {c1a:>6.1f} {f'{c1lo}-{c1hi}':>9} "
              f"{c2:>5} {c3:>5} | {gap:>6.1f} | {bites}")
if __name__ == "__main__":
    main()
