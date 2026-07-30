with open('greedy_crp.py') as f:
    src = f.read()

old = '''def greedy_crp_cost_aware_gated(demand: DemandResult, crp: CRPConfigCostAware) -> Placement:
    """Fix for Bug 2 (CARTS_STATE.md S14b): greedy_crp_cost_aware() ranks
    pairs by net benefit but never gates on its SIGN -- it still fills
    capacity greedily, so a pair with net_benefit <= 0 (e.g. raw==unique,
    pure DPU cost, no dedup saving) gets served anyway whenever capacity
    allows. Confirmed re-triggering at gpus_per_domain=1 (S14b) AND at
    num_experts=64 block-mapping (this session). This function is an exact
    copy of greedy_crp_cost_aware() with ONE added guard: skip serving a
    pair when its net benefit is non-positive, even within capacity budget.

    Local per-pair decision only -- does NOT touch routing or global
    min-max coupling, so it carries none of the Bug-3 cascade risk that
    forced the v2/v3 reverts (S17). Does NOT fix Bug 3 (S15e: a marginally
    negative pair that is itself the global bottleneck) -- that remains
    open by design. Original greedy_crp_cost_aware() is left untouched
    (append-only discipline)."""'''

new = '''def greedy_crp_cost_aware_gated(demand: DemandResult, crp: CRPConfigCostAware) -> Placement:
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
    not silently fixed."""'''

n = src.count(old)
if n != 1:
    print("MISMATCH: expected 1 occurrence of old docstring, found " + str(n))
    print("Aborting -- file NOT modified.")
else:
    src = src.replace(old, new, 1)
    with open('greedy_crp.py', 'w') as f:
        f.write(src)
    print("PATCHED: gated function docstring marked DEPRECATED")
