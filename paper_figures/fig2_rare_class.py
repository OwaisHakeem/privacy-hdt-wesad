"""
Figure 2 — the fairness story: rare-class annihilation under sample-level DP.
Run:  python fig2_rare_class.py
"""
import matplotlib.pyplot as plt
import numpy as np
import _style as S

S.apply_style()
eps = S.EPS
x = np.log2(eps)

fig, ax = plt.subplots(figsize=(S.COL_1, 2.9))

yU = [S.AMUSE_USER[e] for e in eps]
yS = [S.AMUSE_SAMPLE[e] for e in eps]

ax.set_ylim(-0.03, 0.34)
ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)

# zero band where sample-level collapses
ax.axhspan(-0.014, 0.007, color=S.C["sample"], alpha=0.08, zorder=0)

ax.plot(x, yU, marker="s", color=S.C["user"], label="User-level DP", zorder=4)
ax.plot(x, yS, marker="o", color=S.C["sample"], label="Sample-level DP", zorder=5)

# annotation high in the open upper-middle, arrow down to the zero run
ax.annotate("rare class eliminated\n($F_1=0$ at $\\varepsilon=2,4,8$)",
            xy=(np.log2(4), 0.006), xytext=(np.log2(0.5), 0.235),
            fontsize=6.4, color=S.C["sample"], ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=S.C["sample"], lw=0.8))

ax.set_xticks(x); ax.set_xticklabels([f"{e:g}" for e in eps])
ax.set_xlabel(r"Privacy budget $\varepsilon$")
ax.set_ylabel(r"Amusement $F_1$")
# legend top-left corner, clear of both curves (which are mid and bottom)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), borderpad=0.4,
          handletextpad=0.5)

fig.tight_layout()
S.save(fig, "fig2_rare_class")
