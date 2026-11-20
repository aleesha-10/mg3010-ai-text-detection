"""Step 2: BASE model. Replace/extend with the base paper's exact model once chosen.
Default: TF-IDF + logistic regression, 5 seeds, evaluated on own test + external data."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.linear_model import LogisticRegression
import config
from features import Featurizer
from stats_utils import metrics

SEEDS = [config.SEED + i for i in range(5)]
EXTENDED = False   # 03_extended_model.py flips this

def run(extended=False, tag="base", lexicon=None):
    own = pd.read_csv(config.DATA_CLEAN / "own_corpus.csv")
    ext = pd.read_csv(config.DATA_CLEAN / "external_corpus.csv")
    tr, te = own[own.split == "train"], own[own.split == "test"]
    rows, preds = [], []
    for s in SEEDS:
        f = Featurizer(extended, lexicon)
        Xtr = f.fit_transform(tr.text)
        clf = LogisticRegression(max_iter=2000, random_state=s).fit(Xtr, tr.label)
        for dname, d in [("own_test", te), ("external", ext)]:
            X = f.transform(d.text)
            p, sc = clf.predict(X), clf.predict_proba(X)[:, 1]
            rows.append({"model": tag, "seed": s, "dataset": dname, **metrics(d.label, p, sc)})
            if s == SEEDS[0]:
                preds.append(pd.DataFrame({"model": tag, "dataset": dname, "text_id": d.text_id.values,
                                           "y": d.label.values, "pred": p, "score": sc}))
    config.TAB_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(config.TAB_DIR / f"results_{tag}.csv", index=False)
    pd.concat(preds).to_csv(config.TAB_DIR / f"preds_{tag}.csv", index=False)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    r = run(False, "base")
    print(r.groupby("dataset")[["accuracy", "f1", "roc_auc"]].agg(["mean", "std"]).round(3))
