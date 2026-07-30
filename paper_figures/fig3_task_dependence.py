"""
Figure 3 — privacy budgets are task-dependent.
Binary vs four-class utility under user-level DP, each against its own chance line.
Run:  python fig3_task_dependence.py
"""
import matplotlib.pyplot as plt
import numpy as np
import _style as S

S.apply_style()
eps = S.EPS
x = np.log2(eps)

fig, ax = plt.subplots(figsize=(S.COL_1, 2.9))

yB = [S.UTIL_BINARY[e] for e in eps]
yM = [S.UTIL_USER[e] for e in eps]

ax.set_ylim(0.13, 1.0)
ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)

ax.plot(x, yB, marker="D", color=S.C["accent"], label="Binary (stress vs.\\ rest)", zorder=5)
ax.plot(x, yM, marker="s", color=S.C["user"], label="Four-class", zorder=4)

ax.axhline(S.CHANCE_BINARY, ls=":", lw=0.9, color=S.C["accent"], alpha=0.85)
ax.axhline(S.CHANCE_MULTICLASS, ls=":", lw=0.9, color=S.C["user"], alpha=0.85)

# both chance labels at the LEFT edge, each above its own line, in clear space
ax.text(x[-1] + 0.30, S.CHANCE_BINARY - 0.010, "binary chance", ha="right", va="top",
        fontsize=5.8, color=S.C["accent"],
        bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.9))
ax.text(x[-1] + 0.30, S.CHANCE_MULTICLASS - 0.010, "4-class chance", ha="right", va="top",
        fontsize=5.8, color=S.C["user"],
        bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.9))

ax.set_xticks(x); ax.set_xticklabels([f"{e:g}" for e in eps])
ax.set_xlabel(r"Privacy budget $\varepsilon$")
ax.set_ylabel("Macro-F1")
# legend lower-right: open area below the binary curve, right of the four-class rise
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), borderpad=0.5,
          handletextpad=0.5)

fig.tight_layout()
S.save(fig, "fig3_task_dependence")
