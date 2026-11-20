"""Step 5: figures (matplotlib, saved at 400 dpi as PNG + vector PDF)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import config

plt.rcParams.update(config.FIG_STYLE)
config.FIG_DIR.mkdir(exist_ok=True)
def save(fig, name):
    fig.savefig(config.FIG_DIR / f"{name}.png", dpi=config.DPI); fig.savefig(config.FIG_DIR / f"{name}.pdf"); plt.close(fig)

# Fig: ROC base vs extended
fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
for ax, d in zip(axs, ["own_test", "external"]):
    for tag in ["base", "extended"]:
        x = pd.read_csv(config.TAB_DIR / f"preds_{tag}.csv").query("dataset == @d")
        fpr, tpr, _ = roc_curve(x.y, x.score); ax.plot(fpr, tpr, label=tag)
    ax.plot([0, 1], [0, 1], "k:", lw=.8); ax.set_title(d); ax.set_xlabel("False positive rate")
axs[0].set_ylabel("True positive rate"); axs[0].legend(); fig.tight_layout(); save(fig, "fig_roc")

# Fig: F1 across seeds, own vs external
r = pd.concat([pd.read_csv(config.TAB_DIR / f"results_{t}.csv") for t in ["base", "extended"]])
m = r.groupby(["dataset", "model"]).f1.agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for i, tag in enumerate(["base", "extended"]):
    s = m[m.model == tag]; ax.bar([j + i * .38 for j in range(len(s))], s["mean"], .38, yerr=s["std"], label=tag, capsize=2)
ax.set_xticks([.19, 1.19]); ax.set_xticklabels(sorted(m.dataset.unique())); ax.set_ylabel("F1 (mean ± SD, 5 seeds)"); ax.legend()
fig.tight_layout(); save(fig, "fig_f1_seeds")

# Fig: robustness
rb = pd.read_csv(config.TAB_DIR / "robustness.csv")
p = rb.pivot(index="test", columns="model", values="f1")
fig, ax = plt.subplots(figsize=(3.4, 3)); p.plot.barh(ax=ax); ax.set_xlabel("F1"); ax.set_ylabel(""); fig.tight_layout(); save(fig, "fig_robustness")
print("figures saved to", config.FIG_DIR)
