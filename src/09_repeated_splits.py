"""Honest variance + second-dataset evaluation.
(a) 5 different random grouped train/test splits per corpus (own, HC3) -> real mean +/- SD (the fixed split in
    01_clean_data makes 'seeds' in 02/03 identical).
(b) Grouping: a human email and the AI emails written for the same subject share a group, so a topic never
    appears in both train and test (prevents topic leakage).
(c) Cross-corpus transfer both ways (own -> HC3 and HC3 -> own).
Writes tables/repeated_splits.csv and tables/transfer.csv."""
import sys, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
import config
from features import Featurizer
from stats_utils import metrics

SEEDS = [config.SEED + i for i in range(5)]
own = pd.read_csv(config.DATA_CLEAN / "own_corpus.csv")
ext = pd.read_csv(config.DATA_CLEAN / "external_corpus.csv")
for d in (own, ext):
    d["group"] = d.text_id.astype(str).str.extract(r"(\d+)$")[0]


CONFIGS = [("base", False, False), ("stylometric", True, False), ("stylometric+emotion", True, True)]


def fit_eval(tr, te, extended, emotion):
    f = Featurizer(extended, emotion=emotion); clf = LogisticRegression(max_iter=2000, random_state=config.SEED).fit(f.fit_transform(tr.text), tr.label)
    X = f.transform(te.text)
    return metrics(te.label, clf.predict(X), clf.predict_proba(X)[:, 1])


rows = []
for name, d in [("own", own), ("hc3", ext)]:
    for s in SEEDS:
        tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=s).split(d, groups=d.group))
        for tag, ext_flag, emo in CONFIGS:
            rows.append({"corpus": name, "seed": s, "model": tag, **fit_eval(d.iloc[tr_i], d.iloc[te_i], ext_flag, emo)})
res = pd.DataFrame(rows); res.to_csv(config.TAB_DIR / "repeated_splits.csv", index=False)
print(res.groupby(["corpus", "model"])[["accuracy", "f1", "roc_auc"]].agg(["mean", "std"]).round(3))

tr_rows = []
for a, b, tag in [(own, ext, "own->hc3"), (ext, own, "hc3->own")]:
    for m, ext_flag, emo in CONFIGS:
        tr_rows.append({"direction": tag, "model": m, **fit_eval(a, b, ext_flag, emo)})
tr = pd.DataFrame(tr_rows); tr.to_csv(config.TAB_DIR / "transfer.csv", index=False)
print(tr.round(3).to_string(index=False))
