"""
CARTS - Component 2: Greedy-CRP relay placement
===============================================
Phase 4 of the CARTS implementation roadmap (Component 2 of 3).

Consumes the demand from Component 1 (demand_extractor.py) and decides WHERE to
place deduplicating relays, using a bottleneck-first greedy heuristic.

Model (v1, kept physically clean):
  - One relay candidate per domain (the source-side aggregator r_d in domain d).
  - Dispatch dedup saves inter-domain bandwidth ONLY if done BEFORE the crossing,
    so the relay for pair (di, dj) is the SOURCE-domain relay r_di: it dedups
    di's outgoing distinct tokens, sends each across once, dest domain fans out
    locally.
  - Each relay r_d has a finite capacity cap (distinct tokens it can dedup per
    round) -> a domain cannot always dedup ALL its outgoing pairs. Bottleneck-
    first decides which pairs get the scarce dedup capacity.

Objective proxy: minimize the maximum inter-domain link load (congestion proxy).
  - Before CARTS (direct):  link(di->dj) carries lambda(di,dj)  (raw pairs).
  - After CARTS (served):   link(di->dj) carries Unique(di,dj)  (deduped, lossless).
  - After CARTS (unserved): link(di->dj) still carries lambda(di,dj).

Honest simplification (vs v0.8 `min(Unique, cap)`):
  Capacity is consumed at WHOLE-PAIR granularity (a pair is fully deduped or
  not deduped). The `min()` form could forward fewer than Unique distinct tokens,
  which would DROP tokens -> lossy. Whole-pair keeps it lossless. Finer-grained
  partial dedup is a possible refinement (needs per-token copy counts).

NOTE: relay_capacity default is a PLACEHOLDER. The real value comes from the
BlueField-3 throughput calibration (DPU cost model, resolved in Component 3).
"""

from dataclasses import dataclass
from demand_extractor import MoEConfig, DemandResult, generate_demand, from_routes


# --------------------------------------------------------------------------
@dataclass
class CRPConfig:
    relay_capacity: int = 10**9   # distinct tokens a domain relay can dedup / round
                                  # PLACEHOLDER pending BlueField-3 calibration


@dataclass
class Placement:
    demand: DemandResult
    crp: CRPConfig
    served: dict     # (di,dj) -> bool  (was this pair deduped by a relay?)
    load: dict       # (di,dj) -> inter-domain link load AFTER placement
    x: dict          # domain d -> 0/1  (is relay r_d used?)
    cap_left: dict   # domain d -> remaining relay capacity

    def inter_pairs(self):
        n = self.demand.cfg.num_domains
        return [(i, j) for i in range(n) for j in range(n) if i != j]

    def max_load_before(self):
        return max(self.demand.lam(i, j) for (i, j) in self.inter_pairs())

    def max_load_after(self):
        return max(self.load[(i, j)] for (i, j) in self.inter_pairs())

    def relays_used(self):
        return sorted(d for d, on in self.x.items() if on)


# --------------------------------------------------------------------------
def greedy_crp(demand: DemandResult, crp: CRPConfig) -> Placement:
    n = demand.cfg.num_domains
    cap_left = {d: crp.relay_capacity for d in range(n)}
    x = {d: 0 for d in range(n)}
    served, load = {}, {}

    inter = [(i, j) for i in range(n) for j in range(n) if i != j]
    # bottleneck-first: largest raw lambda handled first
    inter.sort(key=lambda p: demand.lam(*p), reverse=True)

    for (di, dj) in inter:
        raw = demand.lam(di, dj)
        u = demand.unique(di, dj)
        if raw == 0:
            served[(di, dj)] = False
            load[(di, dj)] = 0
            continue
        r = di                      # source-side aggregator
        if cap_left[r] >= u:        # enough capacity to dedup this whole pair
            cap_left[r] -= u
            x[r] = 1
            served[(di, dj)] = True
            load[(di, dj)] = u      # deduped, lossless
        else:                       # no capacity -> traffic crosses un-deduped
            served[(di, dj)] = False
            load[(di, dj)] = raw    # direct

    return Placement(demand, crp, served, load, x, cap_left)


# --------------------------------------------------------------------------
def print_load(title, placement):
    n = placement.demand.cfg.num_domains
    print(f"\n{title}  (rows=src, cols=dst; '*' = deduped by relay)")
    print("        " + "".join(f"d{j:<7}" for j in range(n)))
    for i in range(n):
        cells = []
        for j in range(n):
            if i == j:
                cells.append(f"{'-':<8}")
            else:
                mark = "*" if placement.served[(i, j)] else " "
                cells.append(f"{placement.load[(i, j)]}{mark}".ljust(8))
        print(f"  d{i:<4}" + "".join(cells))


# --------------------------------------------------------------------------
def validate():
    print("=" * 64)
    print("VALIDATION")
    print("=" * 64)
    cfg = MoEConfig(num_gpus=12, gpus_per_domain=4)   # 3 domains

    pair_tokens = {
        (0, 1): [1, 2, 3, 4, 5, 6, 1, 2, 3, 4],
        (0, 2): [11, 12, 13, 14, 15, 16, 17, 18],
    }
    d = DemandResult(cfg, pair_tokens, [])
    assert d.lam(0, 1) == 10 and d.unique(0, 1) == 6
    assert d.lam(0, 2) == 8 and d.unique(0, 2) == 8

    p1 = greedy_crp(d, CRPConfig(relay_capacity=10**9))
    assert p1.served[(0, 1)] and p1.load[(0, 1)] == 6
    assert p1.served[(0, 2)] and p1.load[(0, 2)] == 8
    assert p1.max_load_before() == 10 and p1.max_load_after() == 8
    print("  Case 1 (unlimited cap):   PASS  -> all deduped, max 10 -> 8")

    p2 = greedy_crp(d, CRPConfig(relay_capacity=0))
    assert not p2.served[(0, 1)] and p2.load[(0, 1)] == 10
    assert p2.max_load_after() == 10 and p2.relays_used() == []
    print("  Case 2 (zero cap):        PASS  -> nothing deduped, max stays 10")

    p3 = greedy_crp(d, CRPConfig(relay_capacity=6))
    assert p3.served[(0, 1)] and p3.load[(0, 1)] == 6
    assert (not p3.served[(0, 2)]) and p3.load[(0, 2)] == 8
    assert p3.max_load_before() == 10 and p3.max_load_after() == 8
    assert p3.cap_left[0] == 0 and p3.relays_used() == [0]
    print("  Case 3 (tight, contended): PASS  -> bottleneck pair wins cap, max 10 -> 8")

    print("\n  All validation cases passed.\n")


def demo():
    print("=" * 64)
    print("DEMO - 16 GPU, 4x4 domains, top-2, skew=1.5")
    print("=" * 64)
    demand = generate_demand(MoEConfig(skew=1.5, seed=0))
    before = max(demand.lam(i, j) for i in range(4) for j in range(4) if i != j)
    print(f"\n  max inter-domain link load BEFORE CARTS (direct): {before}")

    for cap in (10**9, 220, 150):
        p = greedy_crp(demand, CRPConfig(relay_capacity=cap))
        tag = "unlimited" if cap == 10**9 else str(cap)
        print_load(f"--- relay_capacity = {tag} ---", p)
        nserved = sum(1 for v in p.served.values() if v)
        print(f"    relays used         : {p.relays_used()}")
        print(f"    pairs deduped       : {nserved} / 12")
        print(f"    max load AFTER CARTS : {p.max_load_after()}  "
              f"(was {p.max_load_before()})")


if __name__ == "__main__":
    validate()
    demo()


# --------------------------------------------------------------------------
# Cost-aware variant (Fix B) -- orders each gateway's candidate pairs by NET
# TIME BENEFIT (link-time saved by dedup minus DPU comp-time cost) instead of
# raw token count. Does not modify greedy_crp() above; kept separate for A/B
# comparison. Routing is unchanged (pair (di,dj) only relay-able by domain
# di's own gateway) -- this only changes WHICH pairs that gateway dedups
# within its own capacity budget.
@dataclass
class CRPConfigCostAware:
    relay_capacity: int = 10**9
    dpu_bytes_per_us: int = 100_000     # must match WriterConfig for a fair comparison
    token_bytes: int = 2048             # must match WriterConfig
    link_bw_bytes_per_us: float = 5000.0  # must match the network config's link bandwidth


def greedy_crp_cost_aware(demand: DemandResult, crp: CRPConfigCostAware) -> Placement:
    import math
    n = demand.cfg.num_domains
    cap_left = {d: crp.relay_capacity for d in range(n)}
    x = {d: 0 for d in range(n)}
    served, load = {}, {}

    def comp_cost_micros(unique_tokens):
        size_bytes = unique_tokens * crp.token_bytes
        return max(1, math.ceil(size_bytes / crp.dpu_bytes_per_us))

    def link_time_micros(tokens):
        return (tokens * crp.token_bytes) / crp.link_bw_bytes_per_us

    for gw in range(n):
        my_pairs = [(gw, dj) for dj in range(n) if dj != gw]
        scored = []
        for (di, dj) in my_pairs:
            raw = demand.lam(di, dj)
            u = demand.unique(di, dj)
            if raw == 0:
                served[(di, dj)] = False
                load[(di, dj)] = 0
                continue
            link_savings = link_time_micros(raw - u)
            comp_cost = comp_cost_micros(u)
            scored.append((di, dj, raw, u, link_savings - comp_cost))
        scored.sort(key=lambda t: t[4], reverse=True)
        for (di, dj, raw, u, net_benefit) in scored:
            if cap_left[gw] >= u:
                cap_left[gw] -= u
                x[gw] = 1
                served[(di, dj)] = True
                load[(di, dj)] = u
            else:
                served[(di, dj)] = False
                load[(di, dj)] = raw

    return Placement(demand, crp, served, load, x, cap_left)


def greedy_crp_cost_aware_v2(demand, crp):
    """Fix C (session follow-up to Fix B / Bug 3, S15e). Same net-benefit
    ordering as greedy_crp_cost_aware(), but the per-gateway global
    bottleneck pair (max raw load among that gateway's own outgoing pairs)
    is force-served FIRST, before the capacity budget is spent on
    net-benefit-ranked pairs. This corrects Bug 3: in the original Fix B,
    a high-net-benefit but low-raw-load pair could exhaust capacity before
    the gateway's own worst-congestion pair got served, even though serving
    the bottleneck pair is what actually reduces the system's max link load.
    Appended per code-discipline: does not modify greedy_crp_cost_aware()."""
    import math
    n = demand.cfg.num_domains
    cap_left = {d: crp.relay_capacity for d in range(n)}
    x = {d: 0 for d in range(n)}
    served, load = {}, {}

    def comp_cost_micros(unique_tokens):
        size_bytes = unique_tokens * crp.token_bytes
        return max(1, math.ceil(size_bytes / crp.dpu_bytes_per_us))

    def link_time_micros(tokens):
        return (tokens * crp.token_bytes) / crp.link_bw_bytes_per_us

    for gw in range(n):
        my_pairs = [(gw, dj) for dj in range(n) if dj != gw]
        candidates = []
        for (di, dj) in my_pairs:
            raw = demand.lam(di, dj)
            u = demand.unique(di, dj)
            if raw == 0:
                served[(di, dj)] = False
                load[(di, dj)] = 0
                continue
            link_savings = link_time_micros(raw - u)
            comp_cost = comp_cost_micros(u)
            candidates.append((di, dj, raw, u, link_savings - comp_cost))

        if not candidates:
            continue

        bottleneck = max(candidates, key=lambda t: t[2])  # max raw load
        rest = [c for c in candidates if c is not bottleneck]
        rest.sort(key=lambda t: t[4], reverse=True)
        ordered = [bottleneck] + rest

        for (di, dj, raw, u, net_benefit) in ordered:
            if cap_left[gw] >= u:
                cap_left[gw] -= u
                x[gw] = 1
                served[(di, dj)] = True
                load[(di, dj)] = u
            else:
                served[(di, dj)] = False
                load[(di, dj)] = raw

    return Placement(demand, crp, served, load, x, cap_left)


def greedy_crp_cost_aware_v3(demand, crp):
    """Fix C, corrected (session follow-up to v2's over-correction bug).
    v2 incorrectly forced each GATEWAY's own local max-raw-load pair to be
    served first, which could override a better net-benefit decision for
    pairs that were never the actual system bottleneck (observed: gateway 2
    forced (2,0) over a better-net-benefit (2,1), with zero effect on the
    true bottleneck, which was (1,2) on gateway 1 -- a pure regression).

    Correct version: identify the SINGLE global bottleneck pair (max raw
    load across the entire demand, not per-gateway), force-serve only that
    one pair first (within its own gateway's capacity budget), and process
    every other gateway/pair with pure net-benefit ordering, identical to
    greedy_crp_cost_aware(). Appended per code-discipline: does not modify
    greedy_crp_cost_aware() or greedy_crp_cost_aware_v2()."""
    import math
    n = demand.cfg.num_domains
    cap_left = {d: crp.relay_capacity for d in range(n)}
    x = {d: 0 for d in range(n)}
    served, load = {}, {}

    def comp_cost_micros(unique_tokens):
        size_bytes = unique_tokens * crp.token_bytes
        return max(1, math.ceil(size_bytes / crp.dpu_bytes_per_us))

    def link_time_micros(tokens):
        return (tokens * crp.token_bytes) / crp.link_bw_bytes_per_us

    all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    raw_loads = {p: demand.lam(*p) for p in all_pairs}
    global_bottleneck = max(all_pairs, key=lambda p: raw_loads[p])

    bi, bj = global_bottleneck
    bu = demand.unique(bi, bj)
    if raw_loads[global_bottleneck] > 0 and cap_left[bi] >= bu:
        cap_left[bi] -= bu
        x[bi] = 1
        served[global_bottleneck] = True
        load[global_bottleneck] = bu
    else:
        served[global_bottleneck] = False
        load[global_bottleneck] = raw_loads[global_bottleneck]

    for gw in range(n):
        my_pairs = [(gw, dj) for dj in range(n) if dj != gw and (gw, dj) != global_bottleneck]
        candidates = []
        for (di, dj) in my_pairs:
            raw = demand.lam(di, dj)
            u = demand.unique(di, dj)
            if raw == 0:
                served[(di, dj)] = False
                load[(di, dj)] = 0
                continue
            link_savings = link_time_micros(raw - u)
            comp_cost = comp_cost_micros(u)
            candidates.append((di, dj, raw, u, link_savings - comp_cost))
        candidates.sort(key=lambda t: t[4], reverse=True)
        for (di, dj, raw, u, net_benefit) in candidates:
            if cap_left[gw] >= u:
                cap_left[gw] -= u
                x[gw] = 1
                served[(di, dj)] = True
                load[(di, dj)] = u
            else:
                served[(di, dj)] = False
                load[(di, dj)] = raw

    return Placement(demand, crp, served, load, x, cap_left)


def greedy_crp_cost_aware_gated(demand: DemandResult, crp: CRPConfigCostAware) -> Placement:
    """DEPRECATED -- reference only, NOT used in the active pipeline (S20/S21).

    Originally written as a fix for Bug 2 (S14b): greedy_crp_cost_aware()
    ranks pairs by net benefit but never gates on its sign, so a pair with
    net_benefit <= 0 gets served anyway whenever capacity allows. The first
    gate attempt (raw condition: net_benefit<=0) broke headline numbers
    outright (dropped 6-8/8 served pairs at gpd=4, see S20). A narrower gate
    (skip only when raw==unique, i.e. zero dedup savings) was substituted
    and looked safe at the LOAD level: gpd=4 maxload exactly unchanged
    (252->252) across all 6 seeds, gpd=2 maxload IMPROVED (110->106).

    CYCLES-LEVEL CONFIRMATION (S21) REVERSED THIS CONCLUSION. Real
    ASTRA-sim runs at both validated headline configs show the gate makes
    things WORSE, not better or neutral:
      - gpd=4, cap=237, seed=0: OLD=+7.59 pct vs C0, gated=+5.16 pct vs C0
        (gate delta -2.63 pct, despite load-level showing zero change)
      - gpd=2, cap=120, seed=0: OLD=+0.25 pct vs C0, gated=-5.76 pct vs C0
        (gate delta -6.02 pct, despite load-level showing improvement)

    Root cause (per-rank breakdown, S21): skipping raw==unique pairs at one
    gateway frees relay capacity that gets reassigned elsewhere, but this
    shifts WHICH gateway becomes the system bottleneck via shared-link
    contention/timing that the max_link_load proxy (a byte-count-only
    metric) cannot see. At gpd=4 specifically, gateway 4's cycles improved
    (362250->243612) but an unrelated gateway (12) became the new, HIGHER
    bottleneck (266120->371787) -- a net regression invisible to the
    load-level check. This is the clearest concrete instance yet of the
    S8 limitation (Layer-1 load evidence is solver-side, not
    simulator-independent): load-level-unchanged or load-level-improved
    does NOT imply cycles-level-unchanged-or-improved.

    DECISION: kept for reference only (append-only discipline, same
    treatment as v2/v3, S17). NOT called anywhere in the active pipeline.
    The validated Fix B for production use remains the original, UNGATED
    greedy_crp_cost_aware() above -- its headline numbers (S5/S6/S18) are
    the ones backed by real cycles-level runs and should be cited in the
    paper. Bug 2 itself remains an open, disclosed limitation (like Bug 3),
    not silently fixed."""
    import math
    n = demand.cfg.num_domains
    cap_left = {d: crp.relay_capacity for d in range(n)}
    x = {d: 0 for d in range(n)}
    served, load = {}, {}
    def comp_cost_micros(unique_tokens):
        size_bytes = unique_tokens * crp.token_bytes
        return max(1, math.ceil(size_bytes / crp.dpu_bytes_per_us))
    def link_time_micros(tokens):
        return (tokens * crp.token_bytes) / crp.link_bw_bytes_per_us
    for gw in range(n):
        my_pairs = [(gw, dj) for dj in range(n) if dj != gw]
        scored = []
        for (di, dj) in my_pairs:
            raw = demand.lam(di, dj)
            u = demand.unique(di, dj)
            if raw == 0:
                served[(di, dj)] = False
                load[(di, dj)] = 0
                continue
            link_savings = link_time_micros(raw - u)
            comp_cost = comp_cost_micros(u)
            scored.append((di, dj, raw, u, link_savings - comp_cost))
        scored.sort(key=lambda t: t[4], reverse=True)
        for (di, dj, raw, u, net_benefit) in scored:
            if raw == u:
                served[(di, dj)] = False
                load[(di, dj)] = raw
                continue
            if cap_left[gw] >= u:
                cap_left[gw] -= u
                x[gw] = 1
                served[(di, dj)] = True
                load[(di, dj)] = u
            else:
                served[(di, dj)] = False
                load[(di, dj)] = raw
    return Placement(demand, crp, served, load, x, cap_left)
