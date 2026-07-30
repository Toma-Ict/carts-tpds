#!/usr/bin/env python3
"""
fig_timeline_m3.py -- M3 figure: where the byte-count saving goes.

The three busiest inter-domain pairs before and after deduplication at the
headline configuration, drawn to scale in link occupancy, so three facts are
visible at once: the bottleneck MIGRATES from pair (2,0) to pair (1,0); the
DPU deduplication cost is small and overlapped; and the max-link byte saving
(26.3%) overstates the measured completion-time saving (17.5%).

Data provenance (verified; do not edit a number without re-verifying):
  lambda / unique : dump_lambda_gpd8.py           (seed 0, gpd=8, 32 GPUs)
  cycles          : ablation_gpd8_results.csv     (C0 953146, C3c 786443)
  six-seed mean   : ablation_gpd8_multiseed_results.csv  (+10.29%)
  COMP duration   : verify_fig5_timeline.py       (dpu_duration_micros -> 47)

Palette: CARTS_BLUE is fig_style's locked series blue, unchanged. The coral
pair is the schematic figures' "inter-domain crossing" hue (Figs 1-3), used
here for raw pre-deduplication crossings. NAIVE_AMBER and NEG_RED are
deliberately NOT used: amber denotes naive placement and red denotes a
regression elsewhere in the paper.
"""
import os
import matplotlib.pyplot as plt
import fig_style
from fig_style import CARTS_BLUE

CORAL_HI = "#C4442C"
CORAL_LO = "#E57F66"
BLUE_LO = "#5EA0CA"
INK = "#3A3A3A"
GUIDE = "#B0B0B0"

TOKEN_BYTES = 2048
LINK_BPUS = 5000.0
COMP_US = 47.0
CYC_C0, CYC_C3 = 953146.0, 786443.0
MEAN_6SEED = 10.29
PAIRS = [("(2,0)", 632, 461), ("(1,0)", 628, 466), ("(3,0)", 618, 458)]
EXP_MAXLINK_PCT = 26.27
EXP_CYCLE_PCT = 17.49
FIGW, FIGH = 7.0, 2.75


def t_us(copies):
    return copies * TOKEN_BYTES / LINK_BPUS


def main():
    raw_max = max(p[1] for p in PAIRS)
    ded_max = max(p[2] for p in PAIRS)
    maxlink_pct = 100.0 * (raw_max - ded_max) / raw_max
    cyc_pct = 100.0 * (CYC_C0 - CYC_C3) / CYC_C0
    assert abs(maxlink_pct - EXP_MAXLINK_PCT) < 0.01, maxlink_pct
    assert abs(cyc_pct - EXP_CYCLE_PCT) < 0.01, cyc_pct
    assert PAIRS[0][1] == raw_max, "pair 0 must be the C0 bottleneck"
    assert PAIRS[1][2] == ded_max, "pair 1 must be the CARTS bottleneck"
    assert PAIRS[0][2] < ded_max, "bottleneck must migrate"

    fig_style.apply()
    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    ax.grid(False)
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)
    h = 0.60
    ys, labels = [], []

    for i, (name, raw, ded) in enumerate(PAIRS):
        y = 5.7 - i * 0.78
        ys.append(y)
        labels.append(name)
        ax.barh(y, t_us(raw), height=h, edgecolor="none", zorder=2,
                color=CORAL_HI if i == 0 else CORAL_LO)
        ax.text(t_us(raw) + 4, y, "%d" % raw, va="center", fontsize=7.5,
                color=INK)
    ax.text(t_us(PAIRS[0][1]) - 6, 5.7, "bottleneck", va="center", ha="right",
            fontsize=7.5, color="white", zorder=4)

    ax.barh(3.45, COMP_US, height=0.34, color=CARTS_BLUE, alpha=0.40,
            hatch="///", edgecolor="white", linewidth=0.6, zorder=2)
    ax.text(COMP_US + 5, 3.45,
            "DPU dedup %.0f \u00b5s \u00b7 overlaps transfer" % COMP_US,
            va="center", ha="left", fontsize=7.5, color=INK)

    for i, (name, raw, ded) in enumerate(PAIRS):
        y = 2.45 - i * 0.78
        ys.append(y)
        labels.append(name)
        ax.barh(y, t_us(ded), height=h, edgecolor="none", zorder=2,
                color=CARTS_BLUE if i == 1 else BLUE_LO)
        ax.text(t_us(ded) + 4, y, "%d" % ded, va="center", fontsize=7.5,
                color=INK)
    ax.text(t_us(PAIRS[1][2]) - 6, 1.67, "new bottleneck", va="center",
            ha="right", fontsize=7.5, color="white", zorder=4)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.text(-0.115, 0.80, "baseline", transform=ax.transAxes, rotation=90,
            va="center", ha="center", fontsize=8.5, color=CORAL_HI)
    ax.text(-0.115, 0.30, "CARTS", transform=ax.transAxes, rotation=90,
            va="center", ha="center", fontsize=8.5, color=CARTS_BLUE)
    ax.set_xlabel("link occupancy of the pair (\u00b5s, at 5\u2009GB/s)")
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.75, 6.35)

    x_raw, x_ded = t_us(raw_max), t_us(ded_max)
    x_cyc = x_raw * (1.0 - cyc_pct / 100.0)
    for x in (x_raw, x_ded, x_cyc):
        ax.axvline(x, ymin=0.03, ymax=0.60, color=GUIDE, lw=0.5,
                   ls=(0, (2, 2)), zorder=1)
    ax.annotate("", xy=(x_raw, 0.10), xytext=(x_ded, 0.10),
                arrowprops=dict(arrowstyle="<->", color=CORAL_HI, lw=0.9))
    ax.text(x_ded - 5, 0.10,
            "max link load  \u2212%.1f\u2009%%" % maxlink_pct,
            va="center", ha="right", fontsize=7.5, color=CORAL_HI)
    ax.annotate("", xy=(x_raw, -0.48), xytext=(x_cyc, -0.48),
                arrowprops=dict(arrowstyle="<->", color=CARTS_BLUE, lw=1.0))
    ax.text(x_cyc - 5, -0.48,
            "measured makespan  \u2212%.1f\u2009%%   "
            "(\u2212%.1f\u2009%% mean, 6 seeds)" % (cyc_pct, MEAN_6SEED),
            va="center", ha="right", fontsize=7.5, color=CARTS_BLUE)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fig_timeline_m3.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    print("max link  %d -> %d   (-%.2f%%)" % (raw_max, ded_max, maxlink_pct))
    print("cycles    %d -> %d   (-%.2f%%)" % (CYC_C0, CYC_C3, cyc_pct))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
