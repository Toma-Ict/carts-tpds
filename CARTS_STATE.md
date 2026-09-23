

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

## S40. Repo migration -- CARTS (archived) -> carts-tpds

The old artifact repo github.com/Toma-Ict/CARTS was archived on GitHub and is
read-only; three S39 commits could not be pushed and were stranded locally in
/workspace/carts_repo/.

Active artifact repo is now **github.com/Toma-Ict/carts-tpds**, an ASTRA-sim
fork with the project scripts under **carts/** and default branch **master**
(not main). Local clone: /workspace/carts_tpds_repo/.

Layout differs from the old repo: old CARTS had scripts at root, carts-tpds has
them under carts/. Copy targets accordingly.

CARTS_STATE.md now lives at the repo root of carts-tpds. It currently
contains S39 and S40 only. The archived CARTS repo turned out not to hold the
S1-S38 history either, so the earlier commit message claiming a full restore
was wrong. The authoritative S1-S38 record is the copy uploaded to the Claude
project; reconstructing it here is an open task.

Guardrail: /workspace/carts_repo/ push URL set to DISABLED_archived_repo, so an
accidental push there fails loudly rather than silently targeting a dead repo.
This mirrors the existing guardrail on the /workspace/astra-sim/ clone.

**From here on: push to carts_tpds_repo, branch master.**

## S41. Figure layout pass for the 18-page ceiling

Fig. 5 (M3 timeline) moved from figure* to figure. The bar chart is wide and
short, so half width costs no legibility; saves about 0.25 page.

Fig. 10 split in two, because panel (a) is the only place three policies appear
side by side and is worth keeping, but stacking it above two heatmaps cost most
of a page:

- fig_necessity_a_1col.py -> panel (a) alone, single column, figsize 3.4x2.05,
  bar width 0.14 with 0.17 offset, legend above the axes without a frame, short
  labels, no in-plot title. Placed beside Section VII-D where the -32.7 percent
  collapse is discussed, so argument and evidence share a page.
- fig_necessity_bc.py -> the two heatmaps side by side at 7.1x2.35, roughly
  half the old height. Panel letters renumbered (b),(c) -> (a),(b).

Both are append-only new files. fig_necessity_composite_v2.py is untouched and
still runs; it is simply no longer included by the paper. Both new scripts carry
the same anchor gates as v2 and both passed.

Reference impact: \label{fig:necessity} was kept on the heatmap figure so
existing refs did not break, but every panel letter shifted. Audited across all
sections: 06_methodology line 36 (Fig. 7 caption) and line 128 (DPU sweep) b->a;
07_evaluation Table IV caption b--c -> no letter, VII-C b->a, VII-D c->b. The
bar chart gets its own label, fig:necessity-bw.

## S42. num_experts sensitivity re-run at the gpd=8 headline config -- DONE

S19's sweep ran at N=16/gpd=4 (the retired config); citing it in the paper's
Sec. V-A would have stated a headline-config claim from off-config data. Re-run
via new append-only driver num_experts_sweep_gpd8.py (reuses MoEConfigBlock
from num_experts_sweep.py, capacity_sweep.max_link_load, demand_extractor --
none modified). N=32, gpd=8, 4 domains, skew=1.5, E in {32,64,128,256,512}
(experts/GPU = 1,2,4,8,16), seeds 0-5, load-level cap=inf.

Two anchor gates, both PASS: A1 N=16/gpd=4/E=16 seed0 -> raw=252 dedup=218
(S7/S19 regression); A2 N=32/gpd=8/E=32 seed0 -> raw_max=632 (S38 locked C0
L1). A2 additionally reproduced cap=inf dedup_max=466 / 26.27%, matching S39's
locked 632->466 / -26.27% anchor exactly -- a third independent confirmation.

Per-E means: 17.40 / 17.02 / 19.47 / 18.30 / 18.13 %. BAND 17.0-19.5%, flatter
than S19's 15.3-20.1% and at the correct config. Seed spread wide (6.95-32.54%)
and overlapping across all five E values -- flat trend is not seed selection.

Paper: one paragraph appended to Sec. V-A (M1), NOT a Sec. VII subsection.
Reason: this is load-level (cap=inf); placing a 17-19% load number in the
cycles section would invite "why is the headline 10.3%" and contradict M3's
own finding that byte proxies do not predict cycles. Scoped strictly as
load-level.

Standing caveat NOT closed: the only cycles-level E!=N data point remains S20
(E=64, +0.00%, Bug 2). Do not state expert-count invariance as a time claim.

Files: num_experts_sweep_gpd8.py, num_experts_sweep_gpd8_results.csv (30 rows).
Both pushed to carts-tpds (e852c18). num_experts_gpd8_run.log kept on the
server only; matched by **/*.log in .gitignore.

## S43. Fixed-capacity sanity check -- DONE

Motivation: pick_cap() selects, per seed, the capacity maximising the
C1(random) - C3(greedy) gap on that seed's own demand. The headline
(scale_multiseed), the ablation (ablation_gpd8_multiseed) and the robustness
grid (sensitivity_gpd8) all use it, and the paper never discloses the rule.
That makes capacity an outcome-dependent experimental parameter, so the
C1-vs-C3b comparison in Sec. VII-E is partly circular as written.

New append-only driver ablation_fixedcap.py repeats the gpd=8 ablation at
capacities FIXED IN ADVANCE and shared across all six seeds: 400 (below every
per-seed pick), 600 (above every per-seed pick), and 10**9 (unbounded).
Imports ablation_gpd8_multiseed unmodified; only the capacity is overridden.
Its cap-specific anchor gate (cap=541) is deliberately not applied; C0 is
recomputed per seed and all gains are relative to it.

Setup confirmed: pick_cap's six values reproduce as 541/408/498/486/418/424.

Mean cycles gain vs C0 over seeds 0-5:

  cap        C1       C3n       C3c     C3c min   C3c >= C0
  400     3.72%     4.10%     3.41%       0.00%        6/6
  600     7.77%     7.12%    11.65%      +2.79%        6/6
  1e9     6.30%     6.30%     6.30%     -10.16%        4/6

Three findings, all to be stated in the paper without cherry-picking:
1. The benefit is not manufactured by pick_cap. Both finite fixed budgets
   give a positive mean with no seed below baseline.
2. Ordering only pays where the budget binds but does not strangle. At
   cap=600 cost-aware is +11.65% against naive's +7.12%, and naive is
   negative on seeds 4 and 5 (-8.71%, -1.86%) where cost-aware is not.
   At cap=400 the two are within 0.69 pp and identical on four of six
   seeds; the budget is too tight for any policy to decide much. Do NOT
   write "naive wins at cap=400" -- the whole gap is one seed.
3. Protection needs budget AND ordering together. At cap=1e9 all three
   policies are bit-identical per seed (ordering is moot once everything
   is served) and the regressions return, worst seed -10.16%.

Do not promote 11.65% to a new headline. Report all three rows.
Scope: same six seeds, headline bw/DPU point only. Not hardware-calibrated
capacity, not the full bandwidth grid.

Files: ablation_fixedcap.py, ablation_fixedcap_results.csv (90 rows).
ablation_fixedcap_run.log kept on the server only (**/*.log in .gitignore).
