# Which file backs which paper claim

Read this before browsing this directory. Several scripts are earlier
iterations kept for provenance; only the files listed under "Current" feed a
number, table, or figure in the submitted paper. If a number you find here
does not match the paper, check this map before assuming either is wrong --
you are very likely looking at a superseded file.

## Current (cited by the paper)

| Paper claim | Script | Result file |
|---|---|---|
| Table V headline, gpd=8 | scale_multiseed.py | scale_multiseed_gpd8_results.csv |
| Table V headline, gpd=16 | scale_multiseed.py | scale_multiseed_gpd16_results.csv |
| Table IV / Fig. 11 robustness grid | sensitivity_gpd8.py | sensitivity_gpd8_results.csv |
| Table IV necessity, C4 always-dedup | c4_always_dedup.py | c4_always_dedup_results.csv |
| Table VI ablation, C0-C3b | ablation_gpd8_multiseed.py | ablation_gpd8_multiseed_results.csv |
| Sec. IV-D approximation ratio (1.154/1.061) | brute_force_opt.py | brute_force_opt_results.csv |
| Table III DPU calibration | hardware microbenchmark, no script here | bf3_bench_results_v2.csv |
| Sec. V-A num_experts invariance, headline config (S42) | num_experts_sweep_gpd8.py | num_experts_sweep_gpd8_results.csv |
| Table VII envelope | sweep_any_n.py | sweep_any_n_results.csv |
| Fig. 9 (persistence across domain counts) | fig_headline_seeds_v4.py | reads the scale_multiseed files above |
| Fig. 10(a) (necessity, three policies) | fig_necessity_a_1col.py | reads ablation_gpd8_multiseed_results.csv |
| Fig. 10(b,c) (necessity heatmaps) | fig_necessity_bc.py | reads sensitivity_gpd8_results.csv |
| Fig. 7 (M2 bandwidth bridge) | fig_bandwidth_bridge_v7.py | reads sensitivity_gpd8_results.csv |
| Fig. 5 (M3 timeline, seed 0) | fig_timeline_m3.py | reads dump_lambda_gpd8 output |
| Fig. 4 (M3 correlation scatter) | make_m3_scatter_fig.py | reads sensitivity_gpd8_results.csv |
| Fig. 8 (M1 general-k) | make_m1_generalk_fig.py | reads general_k_validate.py output |

## Retired or superseded, not cited by the current paper

- `ablation_gpd8_results.csv` -- single-seed, superseded by the multiseed file
  above.
- `num_experts_sweep_results.csv` -- ran at N=16, gpd=4 (the retired headline
  config). Superseded by `num_experts_sweep_gpd8_results.csv` (S42), which
  reruns the same sweep at the correct N=32, gpd=8 headline config. If these
  two disagree, the gpd8 file is correct; the paper cites only that one.
- `fig_necessity_composite.py`, `fig_necessity_composite_v2.py` -- superseded
  by the a_1col / bc split (S41). Still runnable, no longer included in the
  paper.
- `fig_headline_seeds.py`, `_v2.py`, `_v3.py` -- earlier draft versions;
  `_v4.py` is what the paper uses.
- `fig_bandwidth_bridge_v2.py` through `_v6.py` -- earlier draft versions;
  `_v7.py` is what the paper uses.
- `gpd2_fixb_test.py`, `gpd2_gated_check.py`, `gpd4_gated_check.py`,
  `fixc_seed1_retest.py`, `kappa_step2_probe.py`, `scale_smoke_test.py`,
  `patch_deprecate.py` -- diagnostic scripts written during the Bug 2 / Bug 3
  investigations. None of these feed a paper number directly; they exist to
  explain how the bugs were found and fixed, recorded in CARTS_STATE.md.

## If a script or result is not on either list above

It predates this map or was written for a check that never became a paper
claim. Do not assume it is current. Cross-check CARTS_STATE.md for the session
that introduced it before citing any number from it.
