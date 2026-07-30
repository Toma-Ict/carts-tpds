"""
general_k_validate.py -- M1 general-k validation (CARTS_Part9 9.7 item 1,
M1_load_level_dedup_model.md section 6 caveat: "the general-k form is
validated only at k=2... if a top_k sweep is added, the P(|picks∩Sj|=m)
distribution under Plackett-Luce must be checked numerically for k>2 before
this form is cited").

Computes the EXACT distribution P(|picks ∩ Sj| = m) for m = 0..k under
sequential weighted-without-replacement sampling (matching demand_extractor's
_topk exactly: each pick removes its weight from the pool, renormalizing the
remaining distribution).

IMPORTANT design correction made while writing this: an earlier draft of
this docstring proposed collapsing every "pick lands outside Sj" branch into
a single aggregate-weight step. That is NOT exact here, because out-of-Sj
experts can have different individual weights -- which specific out-of-Sj
expert gets picked changes how much weight is removed from the pool, which
changes the remaining-weight state for every subsequent pick. The k=2
closed form in M1 section 2 gets away with an aggregate sum-over-a precisely
because k=2 only has ONE pick after the first, so there is no THIRD step
whose probabilities could depend on which exact out-of-Sj expert the second
pick consumed. For k>=3 that shortcut breaks down.

The exact method used here is therefore full recursion over INDIVIDUAL
experts at each of the k sequential picks (branching over every remaining
expert, weighted by its probability, removing it from the pool for the
recursive call) -- i.e. directly mirroring _topk's own pop-from-pool
mechanism, evaluated exactly (summing over every order of draws) rather than
sampled. This is exponential in the number of experts in the worst case, so
it is run only at the small E=16 isolation config (matching m1_validate.py's
existing config), not the N=32 headline config.

From the resulting m-distribution, M1 section 6's general-k gain formula is
evaluated:
  gain = sum_{m>=2} (m-1)*P(m) / sum_{m>=1} m*P(m)

Validated against direct Monte Carlo of demand_extractor's real _topk
sampler (no reimplementation of the sampler itself -- this script imports
and calls it directly).

Append-only: imports demand_extractor, modifies nothing.
Run from the carts dir:  python3 general_k_validate.py
"""
import random
import statistics as st
import demand_extractor as D

TPG, E, N, SKEW = 64, 16, 16, 1.5
K_VALUES = (2, 3, 4)   # extends m1_validate.py's k=2-only scope; flagged for review
GPD_VALUES = (8, 4, 2, 1)
MC_TRIALS = 200_000


def exact_m_distribution(weights, in_sj, k):
    """P(|picks ∩ Sj| = m) for m = 0..k, exact, via full recursion over
    individual experts at each sequential pick (mirrors _topk's pop-from-pool
    exactly, evaluated over every draw order instead of sampled).
    weights: dict expert_id -> weight. in_sj: set of expert ids in Sj."""
    dist = [0.0] * (k + 1)
    experts = list(weights.keys())

    def recurse(remaining, hits, picks_left, prob_so_far):
        if picks_left == 0:
            dist[hits] += prob_so_far
            return
        total_w = sum(weights[e] for e in remaining)
        if total_w <= 0:
            dist[hits] += prob_so_far  # degenerate: no weight left to pick from
            return
        for e in remaining:
            p_e = weights[e] / total_w
            new_hits = hits + (1 if e in in_sj else 0)
            new_remaining = [x for x in remaining if x != e]
            recurse(new_remaining, new_hits, picks_left - 1, prob_so_far * p_e)

    recurse(experts, 0, k, 1.0)
    return dist


def gain_from_distribution(dist):
    """M1 section 6: gain = sum_{m>=2}(m-1)*P(m) / sum_{m>=1} m*P(m)."""
    numer = sum((m - 1) * dist[m] for m in range(2, len(dist)))
    denom = sum(m * dist[m] for m in range(1, len(dist)))
    return 0.0 if denom == 0 else numer / denom


def mc_m_distribution(weights, in_sj, k, trials, rng):
    """Monte Carlo cross-check, calling demand_extractor._topk directly
    (no reimplementation of the sampler). _topk(n, w, k, rng) expects w as a
    DICT keyed 0..n-1 (confirmed against demand_extractor.py: pool=list(
    range(n)); ww=[w[e] for e in pool]) and returns picks as those same
    integer keys -- so weights' keys (0..E-1 per _weights()) are used
    directly as expert ids, with no list-conversion indirection."""
    n = len(weights)
    counts = [0] * (k + 1)
    for _ in range(trials):
        picks = D._topk(n, weights, k, rng)
        hits = sum(1 for e in picks if e in in_sj)
        counts[hits] += 1
    return [c / trials for c in counts]


def in_domain_set(gpd, j):
    """Sj = experts assigned to domain j, matching demand_extractor's
    domain_of_gpu/gpu_of_expert convention (gpu_of_expert(e)=e, domain(e) =
    e // gpd), confirmed in M1_load_level_dedup_model.md section 1."""
    return set(e for e in range(E) if e // gpd == j)


def main():
    print("=" * 78)
    print(f"M1 general-k validation  (E={E}, N={N}, skew={SKEW}, MC_TRIALS={MC_TRIALS})")
    print("=" * 78)

    for k in K_VALUES:
        print(f"\n--- top_k = {k} ---")
        print(f"{'gpd':>4} {'exact_gain%':>12} {'mc_gain%':>10} {'diff(pp)':>9} "
              f"{'exact_P(0..k)':>30}")
        for gpd in GPD_VALUES:
            rng_w = random.Random(0)
            weights = D._weights(D.MoEConfig(num_gpus=N, gpus_per_domain=gpd,
                                              num_experts=E, top_k=k,
                                              tokens_per_gpu=TPG, skew=SKEW, seed=0),
                                  rng_w)
            n_domains = N // gpd
            # bottleneck domain j* = argmax in-domain weight share (matches
            # M1 doc's j* = argmax_j r_j; for a fixed weights draw, higher
            # in-domain weight share drives higher r_j directly)
            doms = {j: in_domain_set(gpd, j) for j in range(n_domains)}
            jstar = max(doms, key=lambda j: sum(weights[e] for e in doms[j]))
            in_sj = doms[jstar]

            exact_dist = exact_m_distribution(weights, in_sj, k)
            exact_gain = gain_from_distribution(exact_dist) * 100

            rng_mc = random.Random(1)
            mc_dist = mc_m_distribution(weights, in_sj, k, MC_TRIALS, rng_mc)
            mc_gain = gain_from_distribution(mc_dist) * 100

            dist_str = "[" + ", ".join(f"{p:.4f}" for p in exact_dist) + "]"
            print(f"{gpd:>4} {exact_gain:>12.4f} {mc_gain:>10.4f} "
                  f"{exact_gain - mc_gain:>9.4f} {dist_str:>30}")

        # boundary theorem at general k: min(k,gpd)=1 -> gain=0 exactly
        # (M1 doc section 4 proves this for k=2; check it still holds for k>2)
        gpd1_weights = D._weights(D.MoEConfig(num_gpus=N, gpus_per_domain=1,
                                               num_experts=E, top_k=k,
                                               tokens_per_gpu=TPG, skew=SKEW, seed=0),
                                   random.Random(0))
        in_sj_gpd1 = in_domain_set(1, 0)
        dist_gpd1 = exact_m_distribution(gpd1_weights, in_sj_gpd1, k)
        gain_gpd1 = gain_from_distribution(dist_gpd1)
        print(f"boundary check (gpd=1, k={k}): exact gain = {gain_gpd1:.10f} "
              f"({'OK, exactly 0' if gain_gpd1 == 0.0 else 'VIOLATION'})")


if __name__ == "__main__":
    main()
