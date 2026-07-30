"""
Shared publication style and the single source of truth for all figure data.

Every figure script imports from here so that (a) the visual style is consistent
across the paper and (b) no number is ever retyped in two places. If a result
changes, edit it ONCE here and re-run the figure scripts.

All values are copied from the paper_notes/*.md result files. Provenance for
each block is given in the comment above it. Nothing here is inferred; these are
the recorded experimental outputs.

Output is vector PDF (editable in Illustrator/Inkscape) plus a high-DPI PNG for
quick viewing. Fonts are embedded (Type 42) so the PDF is portable and the text
stays selectable/editable.
"""

from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Publication style
# ----------------------------------------------------------------------
def apply_style() -> None:
    mpl.rcParams.update({
        # vector-friendly, editable text
        "pdf.fonttype": 42,           # TrueType, keeps text editable in the PDF
        "ps.fonttype": 42,
        "svg.fonttype": "none",       # text stays as text in SVG
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.color": "#B0B0B0",
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# Colour-blind-safe palette (Wong 2011, Nature Methods) — distinguishable in
# greyscale print and to the ~8% of readers with colour-vision deficiency.
C = {
    "user":    "#0072B2",   # blue   — user-level DP
    "sample":  "#D55E00",   # vermilion — sample-level DP
    "chance":  "#999999",   # grey   — chance / reference lines
    "accent":  "#009E73",   # green  — highlights
    "amuse":   "#CC79A7",   # pink   — amusement class
    "neutral": "#333333",
}

# Single-column and double-column widths for a typical IEEE/Elsevier journal.
COL_1 = 3.5    # inches, single column
COL_2 = 7.16   # inches, double column


import os
_HERE = os.path.dirname(os.path.abspath(__file__))


def save(fig, stem: str) -> None:
    """Write both an editable vector PDF and a high-DPI PNG, next to this file."""
    pdf = os.path.join(_HERE, f"{stem}.pdf")
    png = os.path.join(_HERE, f"{stem}.png")
    fig.savefig(pdf)
    fig.savefig(png)
    print(f"wrote {stem}.pdf and {stem}.png")


# ----------------------------------------------------------------------
# DATA — single source of truth. Provenance in comments.
# ----------------------------------------------------------------------

# ε grid used throughout (∞ handled separately per figure).
EPS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

CHANCE_MULTICLASS = 0.2402      # C_dp_fedavg_results.md
NONPRIVATE_MACRO = 0.7649       # FedAvg LOSO, B_fedavg_results.md (utility ceiling)

# --- Utility vs epsilon, macro-F1 ---
# user-level: C_dp_fedavg_results.md ; sample-level: C_prime_dp_sample_results.md
UTIL_USER   = {0.5: 0.2493, 1.0: 0.3038, 2.0: 0.3523, 4.0: 0.3658, 8.0: 0.5439, 16.0: 0.6054}
UTIL_USER_SD= {0.5: 0.0779, 1.0: 0.1103, 2.0: 0.1177, 4.0: 0.1215, 8.0: 0.1024, 16.0: 0.1108}
UTIL_SAMPLE = {0.5: 0.3812, 1.0: 0.5431, 2.0: 0.6190, 4.0: 0.6627, 8.0: 0.6708, 16.0: 0.6841}
UTIL_SAMPLE_SD={0.5:0.1159, 1.0: 0.0732, 2.0: 0.0573, 4.0: 0.0286, 8.0: 0.0281, 16.0: 0.0318}

# --- Amusement (rare class) F1 vs epsilon ---
# user-level: C_dp_fedavg_results.md ; sample-level: C_prime_dp_sample_results.md
AMUSE_USER  = {0.5: 0.1280, 1.0: 0.1367, 2.0: 0.1584, 4.0: 0.1790, 8.0: 0.2320, 16.0: 0.2480}
AMUSE_SAMPLE= {0.5: 0.0076, 1.0: 0.0007, 2.0: 0.0000, 4.0: 0.0000, 8.0: 0.0000, 16.0: 0.0384}

# --- Membership-inference attack AUC vs epsilon (cross-entropy signal) ---
# D_mia_results.md (user) and dp-mechanism-comparison (sample). inf = non-private.
AUC_USER   = {"inf": 0.5481, 0.5: 0.5021, 1.0: 0.5019, 2.0: 0.5134, 4.0: 0.5159, 8.0: 0.4997, 16.0: 0.5144}
AUC_USER_SD= {"inf": 0.0768, 0.5: 0.0430, 1.0: 0.0429, 2.0: 0.0577, 4.0: 0.0504, 8.0: 0.0463, 16.0: 0.0582}
AUC_SAMPLE = {"inf": 0.5488, 0.5: 0.4976, 1.0: 0.5005, 2.0: 0.4999, 4.0: 0.5018, 8.0: 0.5090, 16.0: 0.5118}
AUC_SAMPLE_SD={"inf":0.0771, 0.5: 0.0605, 1.0: 0.0559, 2.0: 0.0578, 4.0: 0.0621, 8.0: 0.0680, 16.0: 0.0719}

# --- Binary vs 4-class utility (task dependence) ---
# binary_results.md and C_dp_fedavg_results.md
UTIL_BINARY = {0.5: 0.6200, 1.0: 0.6700, 2.0: 0.7070, 4.0: 0.7840, 8.0: 0.8350, 16.0: 0.8930}
CHANCE_BINARY = 0.4400   # macro-F1 chance for the 78/22 split, binary_results.md

# --- Per-class F1 at a representative budget, both mechanisms (for the
#     decomposition figure). Classes in prevalence order. ---
CLASSES = ["baseline", "stress", "amusement", "meditation"]
PREVALENCE = {"baseline": 0.3951, "stress": 0.2220, "amusement": 0.1220, "meditation": 0.2610}


# ----------------------------------------------------------------------
# LIVE DATA OVERRIDE — read from result JSONs when available.
# The hardcoded values above are the fallback (used in environments without
# the results/ tree, e.g. a clean checkout or a sandbox). When the result
# files ARE present, every figure is drawn from source data, so re-running an
# experiment and re-running the figure script keeps them in exact agreement.
# To FORCE the fallback values, set env var FIG_USE_HARDCODED=1.
# ----------------------------------------------------------------------
import os as _os

def _try_load_from_results():
    if _os.environ.get("FIG_USE_HARDCODED") == "1":
        print("[figures] FIG_USE_HARDCODED=1 -> using embedded values")
        return
    try:
        import _load_results as R
    except Exception as e:
        print(f"[figures] loader import failed ({e}); using embedded values")
        return
    g = globals()
    try:
        um, usd = R.utility("user");   sm, ssd = R.utility("sample")
        au, ausd = R.attack_auc("user"); as_, assd = R.attack_auc("sample")
        amu = R.per_class("user", "amusement"); ams = R.per_class("sample", "amusement")
        binu = R.binary_utility()
    except FileNotFoundError as e:
        print(f"[figures] result file missing ({e.args[0] if e.args else e}); using embedded values")
        return
    except Exception as e:
        print(f"[figures] result read failed ({e}); using embedded values")
        return
    g["UTIL_USER"], g["UTIL_USER_SD"] = um, usd
    g["UTIL_SAMPLE"], g["UTIL_SAMPLE_SD"] = sm, ssd
    g["AUC_USER"], g["AUC_USER_SD"] = au, ausd
    g["AUC_SAMPLE"], g["AUC_SAMPLE_SD"] = as_, assd
    g["AMUSE_USER"], g["AMUSE_SAMPLE"] = amu, ams
    g["UTIL_BINARY"] = binu
    print("[figures] loaded values from results/ JSON files")

_try_load_from_results()
