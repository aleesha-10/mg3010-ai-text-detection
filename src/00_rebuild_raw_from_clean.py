"""Step 0 (fresh clone only): rebuilds data/raw/own_corpus.csv and external_corpus.csv from data/clean/.
data/raw/ is git-ignored, so a fresh clone has no raw files. Row order matches the original assemble step
(humans h0000.., then gpt_oss_120b, then qwen3_27b), so 01_clean_data.py reproduces the same split.
Never overwrites existing raw files."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import config

config.DATA_RAW.mkdir(parents=True, exist_ok=True)
own_raw, ext_raw = config.DATA_RAW / "own_corpus.csv", config.DATA_RAW / "external_corpus.csv"

if own_raw.exists():
    print("data/raw/own_corpus.csv exists; left unchanged")
else:
    own = pd.read_csv(config.DATA_CLEAN / "own_corpus.csv").drop(columns="split")
    order = {"": 0, "gpt_oss_120b": 1, "qwen3_27b": 2}                 # original assembly order
    own["_o"] = own.generator.fillna("").map(order)
    own["_n"] = own.text_id.str.extract(r"h(\d{4})$")[0].astype(int)   # subject number
    own.sort_values(["_o", "_n"]).drop(columns=["_o", "_n"]).to_csv(own_raw, index=False)
    print("rebuilt data/raw/own_corpus.csv")

if ext_raw.exists():
    print("data/raw/external_corpus.csv exists; left unchanged")
else:
    pd.read_csv(config.DATA_CLEAN / "external_corpus.csv").drop(columns="split").to_csv(ext_raw, index=False)
    print("rebuilt data/raw/external_corpus.csv")
