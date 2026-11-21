"""Step 11: shortcut / artefact checks. Measures surface cues that differ between human and AI text
in each corpus and could let a detector separate the classes without learning authorship.
Writes tables/artefact_checks.csv."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.metrics import roc_auc_score
import config

own = pd.read_csv(config.DATA_CLEAN / "own_corpus.csv")
ext = pd.read_csv(config.DATA_CLEAN / "external_corpus.csv")
rows = []

def share(d, mask, cue, corpus):
    rows.append({"corpus": corpus, "cue": cue, "human_share": round(mask[d.label == 0].mean(), 3),
                 "ai_share": round(mask[d.label == 1].mean(), 3), "length_auc": None})

share(own, own.text.str.match(r"^\s*(hi|hello|dear|hey)\b", case=False), "starts with greeting", "own")
share(own, own.text.str.contains(r"\[Your Name\]|\[Name\]", regex=True), "name placeholder e.g. [Your Name]", "own")
share(own, own.text.str.contains("[‘’“” ‑]", regex=True), "typographic quotes / special spaces", "own")
share(ext, ext.text.str.contains(r" [,.?!] | \.$| ,", regex=True), "space before punctuation", "hc3")
for name, d in [("own", own), ("hc3", ext)]:
    rows.append({"corpus": name, "cue": "length only (word count) ROC-AUC", "human_share": None, "ai_share": None,
                 "length_auc": round(roc_auc_score(d.label, d.text.str.split().str.len()), 3)})
out = pd.DataFrame(rows); out.to_csv(config.TAB_DIR / "artefact_checks.csv", index=False)
print(out.to_string(index=False))
