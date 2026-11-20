"""Creates data/raw/emotion_lexicon.csv (columns: word,emotion) from the NRC Emotion Lexicon (EmoLex)
bundled in the `nrclex` pip package. After this, 02/03/04 automatically use emotion features
(per-emotion word share + range + variability) in the EXTENDED model.

  pip install nrclex
  python src/10_make_emotion_lexicon.py

Cite: Mohammad & Turney (2013), "Crowdsourcing a word-emotion association lexicon", Computational Intelligence 29(3).
EmoLex is free for research/education use; do not redistribute it commercially."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import nrclex
import config

path = pathlib.Path(nrclex.__file__).parent / "data" / "nrc_en.json"
lex = json.load(open(path, encoding="utf-8"))          # {word: [emotion, ...]}
KEEP = {"anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "positive", "negative"}
rows = [(w, e) for w, es in lex.items() for e in es if e in KEEP]
config.DATA_RAW.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows, columns=["word", "emotion"])
df.to_csv(config.DATA_RAW / "emotion_lexicon.csv", index=False)
print(len(df), "word-emotion pairs;", df.word.nunique(), "words;", sorted(df.emotion.unique()))
