"""
Figure 1 (the money plot) — utility diverges, empirical privacy does not.
Single shared legend above both panels; panel interiors kept clean.
Run:  python fig1_money_plot.py
"""
import matplotlib.pyplot as plt
import numpy as np
import _style as S

S.apply_style()
eps = S.EPS
x = np.log2(eps)

fig, (axU, axA) = plt.subplots(1, 2, figsize=(S.COL_2, 3.3))

# ---------- panel (a): utility ----------
yU = [S.UTIL_USER[e] for e in eps];   eU = [S.UTIL_USER_SD[e] for e in eps]
yS = [S.UTIL_SAMPLE[e] for e in eps];  eS = [S.UTIL_SAMPLE_SD[e] for e in eps]

axU.set_ylim(0.12, 0.85)
axU.set_xlim(x[0] - 0.6, x[-1] + 0.6)
axU.axhline(S.NONPRIVATE_MACRO, ls=(0, (5, 2)), lw=0.9, color=S.C["neutral"], alpha=0.75)
axU.axhline(S.CHANCE_MULTICLASS, ls=":", lw=0.9, color=S.C["chance"])

lS, = axU.plot([], [], marker="o", color=S.C["sample"], label="Sample-level DP")
lU, = axU.plot([], [], marker="s", color=S.C["user"], label="User-level DP")
axU.errorbar(x, yS, yerr=eS, marker="o", color=S.C["sample"], capsize=2, zorder=5)
axU.errorbar(x, yU, yerr=eU, marker="s", color=S.C["user"], capsize=2, zorder=4)

# reference labels sit just inside the right margin, in whitespace
axU.text(x[0] - 0.30, S.NONPRIVATE_MACRO + 0.035, "non-private (0.765)", ha="left",
         va="bottom", fontsize=5.8, color=S.C["neutral"],
         bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.9))
axU.text(x[-1] + 0.30, S.CHANCE_MULTICLASS + 0.030, "chance (0.240)", ha="right",
         va="bottom", fontsize=5.8, color=S.C["chance"],
         bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.9))

axU.set_xticks(x); axU.set_xticklabels([f"{e:g}" for e in eps])
axU.set_xlabel(r"Privacy budget $\varepsilon$")
axU.set_ylabel("Macro-F1")
axU.set_title("(a)  Utility", loc="left", fontweight="bold")

# ---------- panel (b): attack AUC ----------
aU = [S.AUC_USER[e] for e in eps];   aUe = [S.AUC_USER_SD[e] for e in eps]
aS = [S.AUC_SAMPLE[e] for e in eps];  aSe = [S.AUC_SAMPLE_SD[e] for e in eps]

axA.set_ylim(0.45, 0.59)
xinf = x[0] - 1.7
axA.set_xlim(xinf - 0.55, x[-1] + 0.6)
axA.axhline(0.5, ls=":", lw=0.9, color=S.C["chance"])

# non-private anchors: draw markers, clip the tall error bars at the frame
axA.errorbar([xinf], [S.AUC_SAMPLE["inf"]], yerr=[S.AUC_SAMPLE_SD["inf"]],
             marker="o", mfc="white", color=S.C["sample"], capsize=2, zorder=5,
             clip_on=True)
axA.errorbar([xinf], [S.AUC_USER["inf"]], yerr=[S.AUC_USER_SD["inf"]],
             marker="s", mfc="white", color=S.C["user"], capsize=2, zorder=4,
             clip_on=True)
axA.axvline((xinf + x[0]) / 2, ls="-", lw=0.5, color="#E0E0E0", zorder=0)
axA.errorbar(x, aS, yerr=aSe, marker="o", color=S.C["sample"], capsize=2, zorder=5)
axA.errorbar(x, aU, yerr=aUe, marker="s", color=S.C["user"], capsize=2, zorder=4)

axA.text(x[-1] + 0.5, 0.5, "chance\n0.500", ha="right", va="center",
         fontsize=5.8, color=S.C["chance"],
         bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.9))
axA.annotate("non-private", xy=(xinf, S.AUC_SAMPLE["inf"]), xytext=(xinf + 0.15, 0.573),
             ha="center", va="bottom", fontsize=5.8, color=S.C["neutral"])

axA.set_xticks([xinf] + list(x))
axA.set_xticklabels([r"$\infty$"] + [f"{e:g}" for e in eps])
axA.set_xlabel(r"Privacy budget $\varepsilon$")
axA.set_ylabel("Attack AUC")
axA.set_title("(b)  Empirical privacy", loc="left", fontweight="bold")

# ---------- one shared legend, above both panels ----------
fig.legend(handles=[lS, lU], loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.02), handletextpad=0.5, columnspacing=1.5)

fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=2.5)
S.save(fig, "fig1_money_plot")
