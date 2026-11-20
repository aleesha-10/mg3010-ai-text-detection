"""Feature builders: base (TF-IDF) and extended (stylometric + optional emotion lexicon)."""
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import config

WORD = re.compile(r"\b\w+\b")


def stylometric(texts):
    rows = []
    for t in texts:
        w = WORD.findall(t.lower())
        n = max(len(w), 1)
        sents = max(len(re.findall(r"[.!?]+", t)), 1)
        rows.append([
            len(t), n, n / sents,
            len(set(w)) / n,                                   # type-token ratio
            sum(c in ",;:" for c in t) / max(len(t), 1),       # punctuation rate
            t.count("!") / max(len(t), 1),
            sum(c.isupper() for c in t) / max(len(t), 1),
            np.mean([len(x) for x in w]) if w else 0.0,        # mean word length
        ])
    return np.array(rows, dtype=float)


def emotion_features(texts, lexicon_csv=None):
    """Optional. lexicon_csv columns: word,emotion (e.g. NRC EmoLex, downloaded by you).
    Returns per-text share of words per emotion + range/variability across emotions."""
    if lexicon_csv is None or not lexicon_csv.exists():
        return np.zeros((len(texts), 0))
    lex = pd.read_csv(lexicon_csv)
    emos = sorted(lex["emotion"].unique())
    m = {e: set(lex.loc[lex.emotion == e, "word"]) for e in emos}
    out = []
    for t in texts:
        w = WORD.findall(t.lower()); n = max(len(w), 1)
        v = [sum(x in m[e] for x in w) / n for e in emos]
        out.append(v + [max(v) - min(v), float(np.std(v))])
    return np.array(out, dtype=float)


class Featurizer:
    def __init__(self, extended=False, lexicon_csv=None, emotion=True):
        """extended=True adds stylometric features; emotion=True also adds emotion-lexicon features
        (auto-loads data/raw/emotion_lexicon.csv if it exists)."""
        self.extended = extended
        default = config.DATA_RAW / "emotion_lexicon.csv"
        self.lexicon_csv = (lexicon_csv or (default if default.exists() else None)) if emotion else None
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.scaler = StandardScaler()

    def _extra(self, texts):
        return np.hstack([stylometric(texts), emotion_features(texts, self.lexicon_csv)])

    def fit_transform(self, texts):
        X = self.tfidf.fit_transform(texts)
        if self.extended:
            X = hstack([X, csr_matrix(self.scaler.fit_transform(self._extra(texts)))]).tocsr()
        return X

    def transform(self, texts):
        X = self.tfidf.transform(texts)
        if self.extended:
            X = hstack([X, csr_matrix(self.scaler.transform(self._extra(texts)))]).tocsr()
        return X
