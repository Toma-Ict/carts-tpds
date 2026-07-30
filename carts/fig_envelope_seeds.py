#!/usr/bin/env python3
# fig_envelope_seeds.py -- Fig 9 redesign: same raincloud-lite
# encoding as Fig 7 (visual twin: dots + mean diamond + min-max
# whisker) so the envelope boundary reads as "Fig 7's diamonds
# pinned to zero". Cap-collapse counts (CSV-verified: 1/6, 3/6,
# 3/6, matching Table VIII) annotated under each x position.
# NOTE: Table VIII's CA>=C0 column found erroneous during this
# figure's verify pass; correct values 6/6, 5/6, 6/6 (Overleaf fix).
import csv, statistics as st
from collections import defaultdict
import fig_style
fig_style.apply()
import matplotlib.pyplot as plt

CSV = "scale_multiseed_gpd4_results.csv"
OUT = "fig_envelope_seeds_v2.pdf"
ANCH = {32: (0.142, 6, 1), 64: (0.034, 5, 3), 128: (0.018, 6, 3)}
TOL = 0.005
XPOS = {32: 0, 64: 1, 128: 2}
JIT = [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]

d, cap0 = defaultdict(dict), defaultdict(int)
for r in csv.DictReader(open(CSV)):
    dom = int(r["num_domains"])
    d[dom][int(r["seed"])] = float(r["new_vs_c0"])
    if int(r["cap"]) == 0:
        cap0[dom] += 1
for dom, (em, eg, ec) in ANCH.items():
    v = list(d[dom].values())
    assert len(v) == 6 and abs(st.mean(v) - em) <= TOL \
        and sum(1 for x in v if x >= 0) == eg and cap0[dom] == ec, \
        "anchor fail dom=%d" % dom
print("all anchors verified -- plotting")

fig, ax = plt.subplots(figsize=(3.5, 2.5))
BLUE = fig_style.CARTS_BLUE
for dom in sorted(d):
    x0 = XPOS[dom]
    v = [d[dom][s] for s in sorted(d[dom])]
    ax.plot([x0, x0], [min(v), max(v)], color=BLUE,
            linewidth=0.8, alpha=0.6, zorder=2)
    ax.scatter([x0 + j for j in JIT], v, s=13, color=BLUE,
               alpha=0.9, linewidths=0, zorder=3)
    ax.scatter([x0], [st.mean(v)], marker="D", s=32, color=BLUE,
               edgecolors="white", linewidths=0.6, zorder=4)
    ax.text(x0, -0.135, "cap collapses: %d/6" % cap0[dom],
            ha="center", fontsize=6, color="#777777")
ax.axhline(0, color="black", linewidth=0.6, zorder=1)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["32", "64", "128"])
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.18, 0.55)
ax.set_xlabel("number of domains (gpd = 4 fixed)")
ax.set_ylabel("completion-time gain vs C0 (%)")
ax.annotate("means within seed noise\n(M1 boundary, Thm 2)",
            xy=(0.97, 0.86), xycoords="axes fraction",
            ha="right", fontsize=6.5, color="#555555")
fig.savefig(OUT)
print("wrote", OUT)
