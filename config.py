"""Shared settings. Import this at the top of every script or notebook so that
all results and figures use the same seed, paths and figure style."""
from pathlib import Path

SEED = 42                       # fixed random seed (state it in the paper)
N_BOOTSTRAP = 5000              # bootstrap resamples for confidence intervals
ALPHA = 0.05                    # significance level
DPI = 400                       # figure resolution required by the brief

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"

# Figure style: text must stay readable (>= ~8 pt) at IEEE column width.
FIG_STYLE = {
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}
