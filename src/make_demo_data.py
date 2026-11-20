"""Creates CLEARLY SYNTHETIC toy data so the pipeline can be smoke-tested.
NEVER report results from this data. Real data goes in data/raw/ (see data/codebook_TEMPLATE.csv)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import config

rng = np.random.default_rng(config.SEED)
H = "hi thanks for the update will check tomorrow can we move the call pls see attached numbers look off quick fix needed".split()
A = "we are delighted to share our exciting offer designed to elevate your experience please explore the benefits tailored for you".split()
SHARED = "team meeting project report client budget deadline review".split()

def make(n, domain, gens, shift=0.0):
    rows = []
    for i in range(n):
        lab = int(rng.random() < 0.5)
        base = A if lab else H
        pool = list(base) * 2 + SHARED + (H if rng.random() < 0.15 + shift else [])
        words = list(rng.choice(pool, size=int(rng.integers(15, 40))))
        txt = " ".join(words).capitalize() + ("." if lab else "")
        rows.append((f"{domain}_{i}", txt, lab, rng.choice(gens) if lab else "", domain))
    return pd.DataFrame(rows, columns=["text_id", "text", "label", "generator", "domain"])

config.DATA_RAW.mkdir(parents=True, exist_ok=True)
make(600, "own", ["llmA", "llmB", "llmC"]).to_csv(config.DATA_RAW / "own_corpus.csv", index=False)
make(400, "external", ["llmD"], shift=0.15).to_csv(config.DATA_RAW / "external_corpus.csv", index=False)
print("SYNTHETIC demo data written to data/raw/ (do not report).")
