"""Which words does the BASE detector rely on? Writes tables/top_features.csv (explainability + artefact check).
If top 'AI' words are things like a signature name or stock phrases, the detector is partly learning
prompt/model artefacts, not general AI style. Report this honestly in the paper."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import config
from features import Featurizer

own = pd.read_csv(config.DATA_CLEAN / "own_corpus.csv")
tr = own[own.split == "train"]
f = Featurizer(False); clf = LogisticRegression(max_iter=2000, random_state=config.SEED).fit(f.fit_transform(tr.text), tr.label)
names = np.array(f.tfidf.get_feature_names_out()); w = clf.coef_[0]; o = np.argsort(w)
out = pd.concat([pd.DataFrame({"direction": "human", "feature": names[o[:25]], "weight": w[o[:25]]}),
                 pd.DataFrame({"direction": "AI", "feature": names[o[::-1][:25]], "weight": w[o[::-1][:25]]})])
out.to_csv(config.TAB_DIR / "top_features.csv", index=False)
print(out.round(2).to_string(index=False))
