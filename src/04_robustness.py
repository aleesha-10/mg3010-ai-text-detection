"""Step 4: robustness. (a) text perturbations, (b) leave-one-generator-out (unseen LLM),
(c) pre-made paraphrases if a 'text_para' column exists in own_corpus (e.g. from DIPPER/an LLM)."""
import sys, pathlib, random, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.linear_model import LogisticRegression
import config
from features import Featurizer
from stats_utils import metrics

random.seed(config.SEED)

def typos(t, p=0.03):
    ch = list(t)
    for i in range(len(ch) - 1):
        if ch[i].isalpha() and random.random() < p:
            ch[i], ch[i + 1] = ch[i + 1], ch[i]
    return "".join(ch)
def strip_punct(t): return re.sub(r"[^\w\s]", "", t.lower())
def shuffle_sents(t):
    s = re.split(r"(?<=[.!?])\s+", t); random.shuffle(s); return " ".join(s)

def strip_boilerplate(t):
    """Remove greeting and sign-off, the most template-like parts of LLM emails."""
    t = re.sub(r"^\s*(hi|hello|dear|hey)\b[^,\n]{0,30},?\s*", "", t, flags=re.I)
    lines = t.rstrip().split("\n")
    while len(lines) > 1 and len(lines[-1].split()) <= 4:
        lines.pop()
    t = "\n".join(lines)
    t = re.sub(r"\s*(best|regards|best regards|sincerely|cheers|thanks|thank you)\b[^.\n]{0,40}\.?\s*(\w+\.?)?\s*$", "", t, flags=re.I)
    return t.strip()


PERTURB = {"strip_boilerplate": strip_boilerplate, "none": lambda t: t, "typos": typos, "no_punct_lower": strip_punct, "shuffle_sentences": shuffle_sents}
own = pd.read_csv(config.DATA_CLEAN / "own_corpus.csv")
tr, te = own[own.split == "train"], own[own.split == "test"]
rows = []
for ext_flag, tag in [(False, "base"), (True, "extended")]:
    f = Featurizer(ext_flag); clf = LogisticRegression(max_iter=2000, random_state=config.SEED).fit(f.fit_transform(tr.text), tr.label)
    for name, fn in PERTURB.items():
        X = f.transform(te.text.map(fn)); p, s = clf.predict(X), clf.predict_proba(X)[:, 1]
        rows.append({"model": tag, "test": f"perturb:{name}", **metrics(te.label, p, s)})
    if "text_para" in te.columns and te.text_para.notna().any():
        t2 = te.dropna(subset=["text_para"]); X = f.transform(t2.text_para)
        rows.append({"model": tag, "test": "paraphrase", **metrics(t2.label, clf.predict(X), clf.predict_proba(X)[:, 1])})
    # leave-one-generator-out: train without generator g (and its texts), test on g + all humans in test
    for g in sorted(set(own.generator.dropna()) - {""}):
        trg = tr[(tr.generator != g)]; teg = own[(own.split == "test") & ((own.generator == g) | (own.label == 0))]
        if teg.label.nunique() < 2: continue
        f2 = Featurizer(ext_flag); c2 = LogisticRegression(max_iter=2000, random_state=config.SEED).fit(f2.fit_transform(trg.text), trg.label)
        X = f2.transform(teg.text)
        rows.append({"model": tag, "test": f"unseen_generator:{g}", **metrics(teg.label, c2.predict(X), c2.predict_proba(X)[:, 1])})
out = pd.DataFrame(rows); out.to_csv(config.TAB_DIR / "robustness.csv", index=False)
print(out.round(3).to_string(index=False))
