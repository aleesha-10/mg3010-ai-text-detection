"""Step 3: EXTENDED model = base + stylometric (+ optional emotion-lexicon) features,
then base-vs-extended comparison with bootstrap CIs and McNemar's test (nested comparison)."""
import sys, pathlib, importlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import config
from stats_utils import bootstrap_ci, mcnemar

base = importlib.import_module("02_base_model")
lex = config.DATA_RAW / "emotion_lexicon.csv"        # optional: word,emotion
lex = lex if lex.exists() else None
base.run(True, "extended", lex)

rows = []
for d in ["own_test", "external"]:
    a = pd.read_csv(config.TAB_DIR / "preds_base.csv").query("dataset == @d")
    b = pd.read_csv(config.TAB_DIR / "preds_extended.csv").query("dataset == @d")
    stat, p, n01, n10 = mcnemar(a.y, a.pred, b.pred)
    for tag, x in [("base", a), ("extended", b)]:
        lo, hi = bootstrap_ci(x.y, x.pred, x.score, "f1")
        alo, ahi = bootstrap_ci(x.y, x.pred, x.score, "roc_auc")
        rows.append({"dataset": d, "model": tag, "f1_ci_lo": lo, "f1_ci_hi": hi,
                     "auc_ci_lo": alo, "auc_ci_hi": ahi,
                     "mcnemar_p_base_vs_ext": p, "base_only_correct": n01, "ext_only_correct": n10})
out = pd.DataFrame(rows)
out.to_csv(config.TAB_DIR / "comparison_ci_mcnemar.csv", index=False)
print(out.round(3).to_string(index=False))
