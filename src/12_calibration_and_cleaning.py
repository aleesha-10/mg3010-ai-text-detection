"""Step 12: exploratory robustness fixes (added after the main analysis; report as exploratory).

Motivation: the base model ranks AI emails from an unseen generator almost perfectly (AUC >= 0.998) but misses
about a third of them at the default 0.5 threshold. Two pre-specified, label-free fixes are tested on BOTH models:
(1) Validation-calibrated decision threshold: maximise F1 on a grouped validation set, never the test set.
(2) Artefact cleaning: remove corpus artefacts from ALL texts (both classes, train and test) and retrain, to check
    whether detection survives without the shortcuts found in 11_artefact_checks.py.
Evaluated on the base and extended models with:
  E1 fixed grouped split, in-domain (own corpus)
  E2 fixed grouped split, unseen generator (own corpus) + McNemar and bootstrap CIs (base, original corpus)
  E3 5 repeated grouped splits (seeds 42-46), in-domain (own, HC3) and unseen generator (own)
  E4 cross-corpus transfer (own -> HC3, HC3 -> own)
Writes tables/calibration_cleaning.csv, tables/calibration_summary.csv, tables/calibration_tests.csv.

Note on E3: the models are fitted on 87.5% of the 70% training part (about 61% of the data) because a validation
part is held out for calibration, unlike 09_repeated_splits.py (70%). In-domain numbers at 0.5 therefore differ
slightly from 09. This is expected.

Deviation from the pre-specified design (decided before any result was seen): the greeting rule in clean_email only
removes a greeting that ends with , : ! or a line break. The spec's rule deleted up to 30 characters of message
content when a greeting had no comma (8 human emails, 0 AI emails), which would have cut human text only."""
import sys, pathlib, re, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
import config
from features import Featurizer
from stats_utils import metrics, mcnemar, bootstrap_ci

assert (config.DATA_RAW / "emotion_lexicon.csv").exists(), "Run src/10_make_emotion_lexicon.py first (extended model needs it)."

GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)        # 91 candidate thresholds, fixed in advance
SEEDS = [config.SEED + i for i in range(5)]             # 42..46, same as 09_repeated_splits.py
MODELS = {"base": False, "extended": True}               # model name -> Featurizer(extended=...)
GENERATORS = ["gpt_oss_120b", "qwen3_27b"]

# ---------------------------------------------------------------- artefact cleaning (applied to every text)
# Typographic characters that only the LLM emails contain: curly quotes, narrow/no-break spaces, special dashes.
QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', " ": " ", "\xa0": " ",
                        "‑": "-", "–": "-", "—": "-"})


def clean_email(t):
    """Remove email-corpus artefacts: typographic characters, [placeholders], greeting line, short sign-off lines."""
    t = t.translate(QUOTES)
    t = re.sub(r"\[[^\]]{1,20}\]", "", t)                                       # e.g. [Your Name], [EMAIL], [PHONE]
    # greeting: only when it ends in , : ! or a line break within 30 characters (so no message content is cut)
    t = re.sub(r"^\s*(hi|hello|dear|hey)\b[^\n,:!]{0,30}(?:[,:!]|(?=\n))\s*", "", t, flags=re.I)
    lines = t.rstrip().split("\n")
    while len(lines) > 1 and len(lines[-1].split()) <= 4:                         # sign-off lines (Thanks, / Best, / name)
        lines.pop()
    return re.sub(r"[ \t]+", " ", "\n".join(lines)).strip()


def clean_hc3(t):
    """Remove the HC3 tokenisation artefact: spaces before punctuation (only human answers have them)."""
    t = t.translate(QUOTES)
    t = re.sub(r"\s+([,.;:?!)])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def cleaned(d, fn):
    """Return a copy of d with cleaned text; every other column (split, label, generator, text_id) is unchanged.
    If cleaning empties a text, the original text is kept and counted."""
    new = d.text.map(fn)
    empty = new.str.len() == 0
    print(f"  cleaned texts that became empty (original kept): {int(empty.sum())}")
    return d.assign(text=new.where(~empty, d.text))


# ---------------------------------------------------------------- modelling helpers
def groups_of(d):
    """Group id = trailing digits of text_id (human email and its matched AI email share it)."""
    return d.text_id.astype(str).str.extract(r"(\d+)$")[0]


def fit(train, extended):
    """Fit the featurizer + logistic regression exactly as in 02/03 (models are not changed)."""
    f = Featurizer(extended)
    clf = LogisticRegression(max_iter=2000, random_state=config.SEED).fit(f.fit_transform(train.text), train.label)
    return f, clf


def scores(f, clf, d):
    """Predicted probability of the AI class."""
    return clf.predict_proba(f.transform(d.text))[:, 1]


def calibrate(y_val, s_val):
    """Threshold with the highest VALIDATION F1 on the fixed grid; ties -> lowest threshold (argmax = first).
    Only validation labels and scores are ever passed here."""
    assert len(set(y_val)) == 2, "validation set must contain both classes"
    f1s = [f1_score(y_val, (s_val >= t).astype(int), zero_division=0) for t in GRID]
    return float(GRID[int(np.argmax(f1s))])


def evaluate(rows, meta, y, s, thr):
    """Append one row per threshold type (default 0.5 and validation-calibrated)."""
    for ttype, t in [("default_0.5", 0.5), ("val_calibrated", thr)]:
        pred = (s >= t).astype(int)
        rows.append({**meta, "threshold_type": ttype, "threshold": t, "n_test": len(y), **metrics(y, pred, s)})


def run(rows, meta, fitset, valset, testset, extended):
    """Fit on fitset, calibrate on valset, evaluate testset at both thresholds. Returns (y_test, scores, thr)."""
    fitset, valset, testset = (x.reset_index(drop=True) for x in (fitset, valset, testset))
    f, clf = fit(fitset, extended)
    thr = calibrate(valset.label.values, scores(f, clf, valset))
    y, s = testset.label.values, scores(f, clf, testset)
    evaluate(rows, meta, y, s, thr)
    return y, s, thr


def not_gen(d, g):
    """Rows not written by generator g (human rows have NaN generator, so they are kept)."""
    return d[d.generator != g]


def gen_or_human(d, g):
    """Test rows for the unseen-generator setting: AI rows of g plus all human rows."""
    return d[(d.generator == g) | (d.label == 0)]


# ---------------------------------------------------------------- data
own = pd.read_csv(config.DATA_CLEAN / "own_corpus.csv")
hc3 = pd.read_csv(config.DATA_CLEAN / "external_corpus.csv")
print("cleaning own corpus:"); own_c = cleaned(own, clean_email)
print("cleaning HC3:"); hc3_c = cleaned(hc3, clean_hc3)
VERSIONS = {"original": (own, hc3), "cleaned": (own_c, hc3_c)}

rows, tests = [], []
for version, (o, h) in VERSIONS.items():
    tr, va, te = (o[o.split == k] for k in ["train", "val", "test"])
    for model, ext_flag in MODELS.items():
        base_meta = {"corpus_version": version, "model": model, "seed": np.nan}

        # E1: fixed grouped split, in-domain
        run(rows, {"experiment": "E1_fixed_indomain", "corpus": "own", **base_meta, "heldout_generator": ""},
            tr, va, te, ext_flag)

        # E2: fixed grouped split, unseen generator g (g removed from train and val)
        for g in GENERATORS:
            y, s, thr = run(rows, {"experiment": "E2_fixed_unseen_generator", "corpus": "own", **base_meta,
                                   "heldout_generator": g},
                            not_gen(tr, g), not_gen(va, g), gen_or_human(te, g), ext_flag)
            if version == "original" and model == "base":
                p05, pcal = (s >= 0.5).astype(int), (s >= thr).astype(int)
                stat, p, b, c = mcnemar(y, p05, pcal)        # b: only 0.5 correct, c: only calibrated correct
                lo05, hi05 = bootstrap_ci(y, p05, s, "f1")
                locl, hicl = bootstrap_ci(y, pcal, s, "f1")
                tests.append({"experiment": "E2_fixed_unseen_generator", "corpus_version": version, "model": model,
                              "heldout_generator": g, "threshold_cal": thr, "n_test": len(y),
                              "f1_0.5": f1_score(y, p05), "f1_0.5_ci_lo": lo05, "f1_0.5_ci_hi": hi05,
                              "f1_cal": f1_score(y, pcal), "f1_cal_ci_lo": locl, "f1_cal_ci_hi": hicl,
                              "mcnemar_stat": stat, "mcnemar_p": p,
                              "only_0.5_correct": b, "only_cal_correct": c})

        # E3: repeated grouped splits (same 70/30 test sets as 09), validation = 12.5% of the training groups
        for corpus, d in [("own", o), ("hc3", h)]:
            d = d.reset_index(drop=True); grp = groups_of(d)
            for s_ in SEEDS:
                tr_i, te_i = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s_).split(d, groups=grp))
                trp, tep = d.iloc[tr_i].reset_index(drop=True), d.iloc[te_i]
                fi_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=s_)
                                  .split(trp, groups=groups_of(trp)))
                fitp, valp = trp.iloc[fi_i], trp.iloc[va_i]
                meta = {"corpus": corpus, "corpus_version": version, "model": model, "seed": s_}
                run(rows, {"experiment": "E3_repeated_indomain", **meta, "heldout_generator": ""},
                    fitp, valp, tep, ext_flag)
                if corpus == "own":
                    for g in GENERATORS:
                        run(rows, {"experiment": "E3_repeated_unseen_generator", **meta, "heldout_generator": g},
                            not_gen(fitp, g), not_gen(valp, g), gen_or_human(tep, g), ext_flag)

        # E4: cross-corpus transfer; validation = grouped 12.5% of the SOURCE corpus (seed 42)
        for src_name, src, tgt_name, tgt in [("own", o, "hc3", h), ("hc3", h, "own", o)]:
            src = src.reset_index(drop=True)
            fi_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=config.SEED)
                              .split(src, groups=groups_of(src)))
            run(rows, {"experiment": "E4_transfer", "corpus": f"{src_name}->{tgt_name}", **base_meta,
                       "heldout_generator": ""}, src.iloc[fi_i], src.iloc[va_i], tgt, ext_flag)

# ---------------------------------------------------------------- outputs
COLS = ["experiment", "corpus", "corpus_version", "model", "seed", "heldout_generator", "threshold_type", "threshold",
        "n_test", "accuracy", "precision", "recall", "f1", "roc_auc", "tpr_at_5fpr"]
res = pd.DataFrame(rows)[COLS]
res.to_csv(config.TAB_DIR / "calibration_cleaning.csv", index=False)

e3 = res[res.experiment.str.startswith("E3")]
summ = (e3.groupby(["experiment", "corpus", "corpus_version", "model", "threshold_type"], sort=False)
        .agg(f1_mean=("f1", "mean"), f1_sd=("f1", "std"), roc_auc_mean=("roc_auc", "mean"),
             roc_auc_sd=("roc_auc", "std"), recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
             threshold_mean=("threshold", "mean"), n_runs=("f1", "size"))
        .round(3).reset_index())
summ.to_csv(config.TAB_DIR / "calibration_summary.csv", index=False)
tst = pd.DataFrame(tests); tst.to_csv(config.TAB_DIR / "calibration_tests.csv", index=False)

pd.set_option("display.width", 200)
print("\nE1/E2 fixed split (F1 / AUC / threshold):")
fx = res[res.experiment.str.startswith(("E1", "E2"))]
print(fx[["experiment", "corpus_version", "model", "heldout_generator", "threshold_type", "threshold", "f1", "roc_auc"]]
      .round(3).to_string(index=False))
print("\nE3 summary:"); print(summ.to_string(index=False))
print("\nE2 tests (base, original):"); print(tst.round(3).to_string(index=False))
print("\nE4 transfer:")
print(res[res.experiment == "E4_transfer"][["corpus", "corpus_version", "model", "threshold_type", "threshold", "f1",
                                             "roc_auc"]].round(3).to_string(index=False))
