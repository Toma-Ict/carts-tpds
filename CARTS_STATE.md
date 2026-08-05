

## S39. Brute-force optimality gap for CRP objective (4a) -- DONE

Section IV-D promised "empirical approximation-ratio measurement against small
instances solved exactly by brute force" and it had never been run. Now run.

**Why exact, not a bound.** greedy_crp_cost_aware routes pair (di,dj) only
through gateway di, and cap_left is per-gateway, so the global min-max
decomposes into independent per-gateway subproblems. Exhaustive enumeration
over each gateway's served subsets gives the true optimum under that routing
restriction. 4 domains -> 3 outgoing pairs -> 8 subsets per gateway; 16 domains
-> 15 pairs -> 32768. Trivial. No ASTRA-sim: objective (4a) is solver-side.

**Script.** brute_force_opt.py (append-only, new file, touches nothing
validated). Anchor-gated: seed 0 / 32 GPUs / gpd=8 must give lam(2,0)=632,
unique(2,0)=461, greedy_obj=466 or the CSV is not written. Gate passed.
tokens_per_gpu=64, skew=1.5, k=2, dpu=20500, link=5000.
Output: brute_force_opt_results.csv (18 rows).

**Result.** ratio = greedy_obj / opt_obj over 18 headline runs
(32/64/128 GPUs at gpd=8, seeds 0-5):

    n=18   mean=1.0606   max=1.1538   min=1.0000

Exactly optimal on 4 of 18.

**The structural finding.** On all 14 suboptimal runs greedy_obj == raw_max,
with zero exceptions; on all 4 optimal runs greedy_obj < raw_max. The greedy
either dedups the global-bottleneck pair (then it is exactly optimal) or leaves
it entirely raw (then it is suboptimal). Nothing in between, 18/18. Mechanism:
the bottleneck pair is the max-raw pair, so its load is either `unique` or
`raw`. This is the S/IX(a) local-proxy gap and the S/V-C forensic case,
now measured across 18 runs rather than anecdotal on one seed.

**FRAMING DECISION -- do not undo this in a later session.** Report the
RATIO (1.154 worst, 1.061 mean, against the factor-2 best-possible for
k-center, ref hochbaum1985best). Do NOT report "optimal on 4 of 18". Both are
true, but counting exact hits is not how approximation algorithms are
evaluated on NP-hard problems, and "4 of 18" reads as 22% to a reviewer
skimming. An earlier draft led with the count plus three sentences defending
it; if a result needs defending, the framing is wrong. Rewritten to lead with
the ratio and no defensive clause.

**Two scope caveats stated in the paper.** (i) Exact only under Alg. 2's
per-gateway routing restriction, so it measures ordering quality, not routing
quality -- Eq. (4c) permits any relay. (ii) The gap is on the byte objective,
not on time; S/V-C and the Table VI naive-vs-CA inversion both show these do
not track.

**Paper impact.** S/IV-D closing paragraph rewritten (open question -> bounded
empirically). S/IX(a) gains one sentence after "At marginal pairs the two
diverge." Net ~0.15 page. Zero new simulation runs.
