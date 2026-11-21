"""Step 1: data cleaning + train/val/test split (grouped by email subject, fixed seed)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import config

REQ = ["text_id", "text", "label", "generator", "domain"]
config.DATA_CLEAN.mkdir(parents=True, exist_ok=True)
log = []
for name in ["own_corpus", "external_corpus"]:
    df = pd.read_csv(config.DATA_RAW / f"{name}.csv")
    miss = set(REQ) - set(df.columns)
    assert not miss, f"{name}: missing columns {miss}"
    n0 = len(df)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df.text.str.len() >= 20].drop_duplicates(subset="text")
    df["label"] = df["label"].astype(int)
    if name == "own_corpus":
        # Grouped split: a human email and the AI email written for the same subject share a
        # group id (the hNNNN suffix), so a subject never appears in both train and test
        # (prevents topic leakage). Rows are sorted first so the split does not depend on file order.
        # Each group holds exactly one human and one AI email, so classes stay balanced.
        df = df.sort_values("text_id").reset_index(drop=True)
        groups = df.text_id.astype(str).str.extract(r"(\d+)$")[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.SEED)
        tr_i, te_i = next(gss.split(df, groups=groups))
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=config.SEED)
        tr_j, va_j = next(gss2.split(tr, groups=groups.iloc[tr_i]))
        tr, va = tr.iloc[tr_j], tr.iloc[va_j]
        df = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])
    else:
        df["split"] = "test"
    df.to_csv(config.DATA_CLEAN / f"{name}.csv", index=False)
    log.append((name, n0, len(df), df.label.mean()))
print(pd.DataFrame(log, columns=["file", "rows_in", "rows_out", "share_AI"]).to_string(index=False))
