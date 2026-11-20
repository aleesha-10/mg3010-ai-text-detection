"""Build data/raw/own_corpus.csv: human emails (AESLC, Enron subset) + AI emails written for the SAME subjects.

Steps:
  python src/06_build_corpus.py prepare --n 300
  python src/06_build_corpus.py generate --provider groq --model llama-3.3-70b-versatile --name llama3_70b --start 0 --end 150
  python src/06_build_corpus.py generate --provider groq --model llama-3.1-8b-instant --name llama3_8b --start 150 --end 296
  python src/06_build_corpus.py assemble

API key: set env var LLM_API_KEY (never paste it in code or commit it).
  PowerShell:  $env:LLM_API_KEY="your-key"
No API? Use `ingest`: paste prompts into a chat LLM yourself, save replies as CSV (text_id,text), then
  python src/06_build_corpus.py ingest --file replies.csv --name llmA
"""
import sys, pathlib, re, os, json, time, argparse, urllib.request, urllib.error
from typing import cast
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import config

RAW = config.DATA_RAW
HUMAN = RAW / "human_sampled.csv"
PROMPT = ("Write a realistic workplace email about the following subject: \"{subject}\". "
          "Aim for roughly {words} words. Output only the email body text, with no subject line and no explanation.")

ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "xai": "https://api.x.ai/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


def scrub(t):
    t = re.sub(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", "[EMAIL]", t)
    t = re.sub(r"https?://\S+|www\.\S+", "[URL]", t)
    t = re.sub(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", "[PHONE]", t)
    return re.sub(r"\s+\n", "\n", t).strip()


PREAMBLE = re.compile(r"^(here('s| is)|sure|certainly|of course|below is)[^\n]*:\s*\n+", re.I)
SUBJECT = re.compile(r"^subject\s*:[^\n]*\n+", re.I)


def clean_ai(t):
    """Remove chatty wrappers the model adds ('Here is the email:', 'Subject: ...', code fences).
    If left in, the detector would learn these artefacts instead of real style differences."""
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = t.strip().strip("`").strip()
    for _ in range(3):
        t = PREAMBLE.sub("", t, count=1)
        t = SUBJECT.sub("", t, count=1)
    t = t.strip().strip('"').strip()
    return scrub(t)


def prepare(a):
    from datasets import load_dataset
    d = load_dataset("Yale-LILY/aeslc")
    frames = [cast(pd.DataFrame, d[s].to_pandas()) for s in d]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"email_body": "text", "subject_line": "subject"}).dropna()
    bad = re.compile(r"forwarded by|original message|----- ?forward", re.I)
    df = df[~df.text.str.contains(bad)]
    df["text"] = df.text.map(scrub)
    df["words"] = df["text"].map(lambda text: len(text.split()))
    df = df[(df.words.between(40, 200)) & (df.subject.str.len() > 5)].drop_duplicates("text")
    df = df.sample(n=a.n, random_state=config.SEED).reset_index(drop=True)
    df.insert(0, "text_id", [f"h{i:04d}" for i in range(len(df))])
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(HUMAN, index=False)
    print(f"{len(df)} human emails saved to {HUMAN}. Open it and skim for leftover names/personal details.")


def post(url, headers, body):
    req = urllib.request.Request(
        url,
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0", **headers},
    )
    for attempt in range(6):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 5:
                wait = e.headers.get("Retry-After")
                time.sleep(min(float(wait), 120) if wait and wait.replace(".", "").isdigit() else 10 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 5:
                time.sleep(10); continue
            raise RuntimeError(f"Network error: {e}")


def call(provider, model, key, prompt):
    if provider == "anthropic":
        r = post("https://api.anthropic.com/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"},
                 {"model": model, "max_tokens": 600, "messages": [{"role": "user", "content": prompt}]})
        return r["content"][0]["text"]
    if provider in ENDPOINTS:
        r = post(ENDPOINTS[provider], {"Authorization": f"Bearer {key}"},
                 {"model": model, "messages": [{"role": "user", "content": prompt}]})
        return r["choices"][0]["message"]["content"]
    if provider == "gemini":
        r = post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                 {"x-goog-api-key": key}, {"contents": [{"parts": [{"text": prompt}]}]})
        candidates = r.get("candidates") or []
        parts = (candidates[0].get("content") or {}).get("parts") or [] if candidates else []
        if not parts or not parts[0].get("text"):
            raise RuntimeError(f"Gemini returned no text: {r.get('promptFeedback', 'no candidate content')}")
        return parts[0]["text"]
    raise ValueError(provider)


def generate(a):
    key = os.environ.get("LLM_API_KEY")
    assert key, "Set LLM_API_KEY first (see top of file)."
    h = pd.read_csv(HUMAN).iloc[a.start:a.end]
    out = RAW / f"ai_{a.name}.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["text_id", "text"])
    rows = done.to_dict("records"); have = set(done.text_id)
    for _, r in h.iterrows():
        if r.text_id in have: continue
        try:
            txt = call(a.provider, a.model, key, PROMPT.format(subject=r.subject, words=int(r.words)))
        except RuntimeError as e:
            print("STOPPED:", e, "\nProgress is saved. Wait a bit (or check your key/limits) and rerun the same command to resume.")
            return
        rows.append({"text_id": r.text_id, "text": clean_ai(txt)})
        pd.DataFrame(rows).to_csv(out, index=False)      # resumable
        print(len(rows), r.text_id)
        time.sleep(2)
    print("saved", out)


def ingest(a):
    d = pd.read_csv(a.file)[["text_id", "text"]]
    d["text"] = d.text.map(clean_ai)
    d.to_csv(RAW / f"ai_{a.name}.csv", index=False); print(len(d), "rows ingested for", a.name)


def assemble(a):
    h = pd.read_csv(HUMAN)
    human = pd.DataFrame({"text_id": h.text_id, "text": h.text, "label": 0, "generator": "", "domain": "enron_email"})
    parts = [human]
    for f in sorted(RAW.glob("ai_*.csv")):
        g = f.stem[3:]; d = pd.read_csv(f)
        d = d.assign(text_id="ai_" + g + "_" + d.text_id.astype(str), label=1, generator=g, domain="enron_email")
        parts.append(d[["text_id", "text", "label", "generator", "domain"]])
    out = pd.concat(parts, ignore_index=True)
    out = out[out.text.str.len() >= 20]
    out.to_csv(RAW / "own_corpus.csv", index=False)
    print(out.groupby(["label", "generator"]).size())
    sus = out[(out.label == 1) & out.text.str.match(r"^\s*(here|sure|certainly|subject)", case=False)]
    if len(sus): print(f"WARNING: {len(sus)} AI rows still start with a chatty wrapper; inspect:", list(sus.text_id[:5]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("prepare"); x.add_argument("--n", type=int, default=300); x.set_defaults(f=prepare)
    x = s.add_parser("generate"); x.add_argument("--provider", required=True, choices=["anthropic", "openai", "gemini", "groq", "xai", "mistral", "openrouter"])
    x.add_argument("--model", required=True); x.add_argument("--name", required=True)
    x.add_argument("--start", type=int, default=0); x.add_argument("--end", type=int, default=300); x.set_defaults(f=generate)
    x = s.add_parser("ingest"); x.add_argument("--file", required=True); x.add_argument("--name", required=True); x.set_defaults(f=ingest)
    x = s.add_parser("assemble"); x.set_defaults(f=assemble)
    a = p.parse_args(); a.f(a)