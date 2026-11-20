"""Step 5: figures (matplotlib; PNG at 400 dpi + vector PDF). Run after 02, 03, 04, 08, 09.
Palette: olive / ochre / deep teal (dark, colour-blind-safe by luminance; lines also differ in style)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
import config

OLIVE, OCHRE, TEAL, INK, GRID = "#56622B", "#B0701F", "#2C5A5E", "#26282A", "#D9D6CC"
COL = {"base": OLIVE, "stylometric": OCHRE, "extended": OCHRE, "stylometric+emotion": TEAL}
LAB = {"base": "Base (TF-IDF)", "extended": "Extended", "stylometric": "+ stylometric", "stylometric+emotion": "+ stylometric + emotion"}
plt.rcParams.update(config.FIG_STYLE)
plt.rcParams.update({"axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True})
config.FIG_DIR.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(config.FIG_DIR / f"{name}.png", dpi=config.DPI); fig.savefig(config.FIG_DIR / f"{name}.pdf"); plt.close(fig)
    print("saved", name)


def table(name):
    p = config.TAB_DIR / name
    if not p.exists():
        print(f"SKIP: {name} missing (run the earlier step)"); return None
    return pd.read_csv(p)


# 1. ROC curves (fixed split, own test and HC3)
pb, pe = table("preds_base.csv"), table("preds_extended.csv")
if pb is not None and pe is not None:
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.1), sharey=True)
    for ax, d, ttl in zip(axs, ["own_test", "external"], ["Own emails (held-out test)", "HC3 (zero-shot transfer)"]):
        for tag, df, ls in [("base", pb, "-"), ("extended", pe, "--")]:
            x = df[df.dataset == d]; fpr, tpr, _ = roc_curve(x.y, x.score)
            ax.plot(fpr, tpr, ls, color=COL[tag], lw=1.8, label=LAB[tag])
        ax.plot([0, 1], [0, 1], ":", color=INK, lw=.9); ax.set_title(ttl); ax.set_xlabel("False positive rate")
    axs[0].set_ylabel("True positive rate"); axs[0].legend(loc="lower right", frameon=False); fig.tight_layout(); save(fig, "fig_roc")


# 3. Repeated grouped splits: F1 mean +/- SD with every split as a dot
rs = table("repeated_splits.csv")
if rs is not None:
    models = [m for m in ["base", "stylometric", "stylometric+emotion"] if m in set(rs.model)]
    fig, ax = plt.subplots(figsize=(3.5, 3.1)); w = .24
    for i, m in enumerate(models):
        for c, corp in enumerate(["own", "hc3"]):
            v = rs[(rs.model == m) & (rs.corpus == corp)].f1; x = c + (i - (len(models) - 1) / 2) * w
            ax.scatter(np.full(len(v), x), v, s=9, color=COL[m], alpha=.35, zorder=2)
            ax.errorbar(x, v.mean(), yerr=v.std(), fmt="o", color=COL[m], ms=6, capsize=3, lw=1.6, zorder=3, label=LAB[m] if c == 0 else None)
    ax.set_xticks([0, 1], ["Own emails", "HC3"]); ax.set_xlim(-.55, 1.55); ax.set_ylabel("F1 (mean ± SD, 5 grouped splits)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.12), fontsize=7.5, ncol=1); fig.tight_layout(); save(fig, "fig_repeated_splits")

# 4. Robustness (fixed split): F1 under perturbation / unseen generator
rb = table("robustness.csv")
if rb is not None:
    p = rb.pivot(index="test", columns="model", values="f1").sort_values("base")
    fig, ax = plt.subplots(figsize=(3.5, 3.3)); y = np.arange(len(p)); h = .38
    for k, m in enumerate([c for c in ["base", "extended"] if c in p.columns]):
        ax.barh(y + (k - .5) * h, p[m], h * .95, color=COL[m], label=LAB[m])
    ax.set_yticks(y, [t.replace("perturb:", "").replace("unseen_generator:", "unseen: ").replace("_", " ") for t in p.index])
    ax.set_xlabel("F1"); ax.set_xlim(0, 1.05); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.4, -.16), fontsize=7.5, ncol=2); fig.tight_layout(); save(fig, "fig_robustness")

# 5. Cross-corpus transfer (ROC-AUC vs chance)
tr = table("transfer.csv")
if tr is not None:
    models = [m for m in ["base", "stylometric", "stylometric+emotion"] if m in set(tr.model)]
    fig, ax = plt.subplots(figsize=(3.5, 3.1)); w = .25
    for i, m in enumerate(models):
        v = tr[tr.model == m].set_index("direction").roc_auc.reindex(["own->hc3", "hc3->own"])
        ax.bar(np.arange(2) + (i - (len(models) - 1) / 2) * w, v.values, w * .9, color=COL[m], label=LAB[m])
    ax.axhline(.5, color=INK, ls=":", lw=1); ax.text(1.5, .515, "chance", ha="right", fontsize=7.5)
    ax.set_xticks([0, 1], ["own → HC3", "HC3 → own"]); ax.set_ylabel("ROC-AUC"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.12), fontsize=7.5, ncol=1); fig.tight_layout(); save(fig, "fig_transfer")

# 6. What the base detector relies on
tf = table("top_features.csv")
if tf is not None:
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.6))
    for ax, dirn, c, ttl in zip(axs, ["human", "AI"], [OCHRE, OLIVE], ["Words pointing to human", "Words pointing to AI"]):
        d = tf[tf.direction == dirn].copy(); d["a"] = d.weight.abs(); d = d.sort_values("a").tail(15)
        ax.barh(d.feature, d.a, color=c); ax.set_title(ttl); ax.set_xlabel("|logistic-regression weight|")
    fig.tight_layout(); save(fig, "fig_top_features")
