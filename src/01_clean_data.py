"""Step 1: data cleaning + train/val/test split (stratified, fixed seed)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.model_selection import train_test_split
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
        tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=config.SEED)
        tr, va = train_test_split(tr, test_size=0.125, stratify=tr.label, random_state=config.SEED)
        df = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])
    else:
        df["split"] = "test"
    df.to_csv(config.DATA_CLEAN / f"{name}.csv", index=False)
    log.append((name, n0, len(df), df.label.mean()))
print(pd.DataFrame(log, columns=["file", "rows_in", "rows_out", "share_AI"]).to_string(index=False))
