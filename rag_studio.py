"""RAG Studio - one Streamlit app with two tabs:
  1. Results and Advisor - heatmap of the study results, explained step by step, plus a recommended setup
  2. Ask your documents  - ask questions about your own documents; answers come with numbered sources

Run:      streamlit run rag_studio.py
Documents: one folder per corpus under data/ (e.g. data/legal/), files .md .txt .html .pdf
Results:  outputs/results.csv from the notebook (needed for the first tab)
Keys:     typed into the sidebar, kept in this browser session only, never written to disk.
          Without a key the Ask tab runs in "Offline demo" mode (extractive answers, no LLM).
"""
import hashlib
import html as _html
import importlib.util
import bisect
import math
import re
import time
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RAG Studio", page_icon=None, layout="wide")

DATA_DIR = Path("data")
RESULTS_PATH = Path("outputs/results.csv")
OFFLINE = "Offline demo (no key)"
TOPK = 5
PROFILES = {"Accuracy first": (0.8, 0.1, 0.1), "Balanced": (0.5, 0.25, 0.25),
            "Speed first": (0.4, 0.5, 0.1), "Low storage": (0.4, 0.1, 0.5)}
HAS_ST = importlib.util.find_spec("sentence_transformers") is not None
GEN_PROMPT = ("Answer the question using ONLY the context below. Cite the sources you use as [1], [2]. "
              "If the context does not contain the answer, reply exactly: I don't know.\n\nContext:\n")
STOP = set("""the a an of to in and or is are was were what how why which who when for on with by as at be it this that does do
document text documents say says said about according please tell me explain describe section""".split())

# Metrics shown in the heatmap: column -> (label, higher_is_better, one-line meaning for the audience)
METRICS = {
    "f_ctx": ("f_ctx (main score)", True, "Balance of recall and precision. The single number used to rank setups."),
    "recall": ("Recall", True, "How much of the evidence text the retrieved passages cover."),
    "precision": ("Precision", True, "How many of the retrieved passages are actually relevant, rank-aware."),
    "hit": ("Hit@k", True, "Share of questions where at least one relevant passage was retrieved."),
    "mrr": ("MRR", True, "How early the first relevant passage appears (1 = always first)."),
    "lat_p50_ms": ("Latency p50 (ms)", False, "Median time to retrieve for one question. Lower is better."),
    "index_mb": ("Index size (MB)", False, "Disk or memory used by the vector index. Lower is better."),
}

# ----------------------------------------------------------------- visual design
PALETTE = {"accent": "#12a39a", "accent_dark": "#0b6e6e", "navy": "#14213d"}

# Colours are theme-neutral on purpose: text inherits the Streamlit theme and panels use translucent fills,
# so the page stays readable in both light and dark mode.
st.markdown(f"""
<style>
  .block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1250px; }}
  .hero {{ background: linear-gradient(120deg, {PALETTE['navy']} 0%, #1f3b63 55%, {PALETTE['accent_dark']} 100%);
           border-radius: 16px; padding: 1.6rem 1.9rem; margin-bottom: 1.4rem; }}
  .hero h1 {{ color: #ffffff !important; margin: 0 0 .25rem 0; font-size: 2rem; }}
  .hero p {{ color: #d5e3f2 !important; margin: 0; font-size: 1.02rem; }}
  .step {{ display: flex; gap: .8rem; align-items: flex-start; margin: .2rem 0 .6rem 0; }}
  .step .num {{ background: {PALETTE['accent_dark']}; color: #ffffff; width: 1.8rem; height: 1.8rem; min-width: 1.8rem;
                border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: .95rem; }}
  .step .txt {{ font-size: 1.02rem; line-height: 1.5; color: inherit; }}
  .step .txt b {{ color: {PALETTE['accent']}; }}
  .callout {{ background: rgba(18,163,154,.13); border-left: 4px solid {PALETTE['accent']}; border-radius: 8px;
              padding: .85rem 1.1rem; margin: .6rem 0 1rem 0; line-height: 1.55; color: inherit; }}
  .callout b {{ color: {PALETTE['accent']}; }}
  .rec {{ border: 1px solid rgba(128,128,128,.35); border-left: 5px solid {PALETTE['accent']}; border-radius: 12px;
          padding: 1.1rem 1.3rem; background: rgba(128,128,128,.07); margin-bottom: .9rem; }}
  .rec .label {{ text-transform: uppercase; letter-spacing: .08em; font-size: .75rem; opacity: .7; font-weight: 600; }}
  .rec .title {{ font-size: 1.25rem; font-weight: 700; margin-top: .25rem; color: inherit; }}
  div[data-testid="stMetric"] {{ background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.3);
                                 border-radius: 12px; padding: .75rem 1rem; }}
  button[data-baseweb="tab"] {{ font-weight: 600; font-size: 1.02rem; }}
  button[data-baseweb="tab"][aria-selected="true"] {{ color: {PALETTE['accent']}; }}
  div[data-baseweb="tab-highlight"] {{ background-color: {PALETTE['accent']} !important; }}
  .pill {{ display: inline-block; background: rgba(18,163,154,.16); border: 1px solid rgba(18,163,154,.5); color: inherit;
           border-radius: 999px; padding: .12rem .75rem; font-size: .82rem; font-weight: 600; margin: 0 .4rem .35rem 0; }}
  .pill.muted {{ background: rgba(128,128,128,.12); border-color: rgba(128,128,128,.4); }}
  .q {{ font-size: 1.12rem; font-weight: 700; margin-bottom: .5rem; }}
  .lbl {{ text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; opacity: .65; font-weight: 600; margin: .4rem 0 .15rem 0; }}
  .src {{ line-height: 1.65; font-size: .95rem; }}
  mark {{ background: rgba(255,183,3,.4); color: inherit; padding: 0 .12rem; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)


def step(n, text):
    st.markdown(f'<div class="step"><div class="num">{n}</div><div class="txt">{text}</div></div>', unsafe_allow_html=True)


def callout(text):
    st.markdown(f'<div class="callout">{text}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------- chunking (character spans; same methods as the notebook)
BLOCK = re.compile(r"\S.*?(?=\n\s*\n|\Z)", re.S)
HEAD = re.compile(r"^(#{1,6}\s|(Section|SECTION|Article|ARTICLE|Chapter|CHAPTER|Clause|Rule|Regulation)\s+\d+|\d+(\.\d+)*[.)]?\s+[A-Z])")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[`])")


def wc(text, a, b):
    return len(text[a:b].split())


def chunk_fixed(text, size=256, overlap=64, s=0, e=None):
    e = len(text) if e is None else e
    w = [(m.start() + s, m.end() + s) for m in re.finditer(r"\S+", text[s:e])]
    out, step_ = [], max(1, size - overlap)
    for i in range(0, len(w), step_):
        j = min(i + size, len(w))
        out.append((w[i][0], w[j - 1][1]))
        if j == len(w):
            break
    return out


def sentence_spans(text):
    sents = []
    for m in BLOCK.finditer(text):
        bs, be = m.span(); blk = text[bs:be]
        if blk.startswith(("#", "```")):
            sents.append((bs, be)); continue
        last = 0
        for x in SENT.finditer(blk):
            sents.append((bs + last, bs + x.start())); last = x.end()
        sents.append((bs + last, be))
    return [x for x in sents if x[1] > x[0]]


def chunk_sentences(text, n=4, overlap=1):
    sp = sentence_spans(text)
    out, step_ = [], max(1, n - overlap)
    for i in range(0, len(sp), step_):
        j = min(i + n, len(sp))
        out.append((sp[i][0], sp[j - 1][1]))
        if j == len(sp):
            break
    return out


def chunk_structure(text, max_words=300, min_words=40):
    cur, cw, groups = None, 0, []
    for m in BLOCK.finditer(text):
        a, b = m.span(); w = wc(text, a, b)
        head = bool(HEAD.match(text[a:b])) and (b - a) < 200
        if cur is None:
            cur, cw = [a, b], w
        elif (head and cw >= min_words) or (cw + w > max_words and cw >= min_words):
            groups.append(tuple(cur)); cur, cw = [a, b], w
        else:
            cur[1] = b; cw += w
    if cur:
        groups.append(tuple(cur))
    out = []
    for a, b in groups:                                   # sections longer than max_words are cut into windows
        out += chunk_fixed(text, max_words, 0, a, b) if wc(text, a, b) > max_words else [(a, b)]
    return out


def chunk_semantic(text, embed_fn, percentile=90, buffer=1, min_words=40, max_words=300):
    """Cut where the meaning changes most: compare neighbouring sentence windows and split at the biggest jumps."""
    sp = sentence_spans(text)
    if len(sp) < 3:
        return [(sp[0][0], sp[-1][1])] if sp else []
    sents = [text[a:b] for a, b in sp]
    combo = [" ".join(sents[max(0, i - buffer): i + buffer + 1]) for i in range(len(sents))]
    E = embed_fn(combo)
    dist = 1 - np.sum(E[:-1] * E[1:], axis=1)
    thr = np.percentile(dist, percentile)
    groups, cur = [], [0]
    for i, d in enumerate(dist):
        if d > thr:
            groups.append(cur); cur = [i + 1]
        else:
            cur.append(i + 1)
    groups.append(cur)
    merged = []
    for g in groups:
        a, b = sp[g[0]][0], sp[g[-1]][1]
        if merged and wc(text, *merged[-1]) < min_words:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    if len(merged) > 1 and wc(text, *merged[-1]) < min_words:
        last = merged.pop(); merged[-1] = (merged[-1][0], last[1])
    final = []
    for a, b in merged:
        final += chunk_fixed(text, max_words, 0, a, b) if wc(text, a, b) > max_words else [(a, b)]
    return final


def chunk_parent_child(text, child=100, parent=400):
    """Index small child pieces, but return the larger parent section each child belongs to."""
    parents = chunk_structure(text, parent); pst = [p[0] for p in parents]
    out = []
    for a, b in chunk_fixed(text, child, 0):
        pi = max(0, bisect.bisect_right(pst, a) - 1)
        out.append((a, b, parents[pi][0], parents[pi][1]))
    return out


# The nine chunking methods from the study (same names as the heatmap rows).
CHUNKERS = {
    "fixed_128_o0":  dict(kind="fixed", size=128, overlap=0, info="Fixed windows of 128 words, no overlap. Small, precise chunks."),
    "fixed_128_o32": dict(kind="fixed", size=128, overlap=32, info="128-word windows; each shares 32 words with the previous one."),
    "fixed_256_o0":  dict(kind="fixed", size=256, overlap=0, info="Fixed windows of 256 words, no overlap. Larger chunks, more context."),
    "fixed_256_o64": dict(kind="fixed", size=256, overlap=64, info="256-word windows; each shares 64 words with the previous one."),
    "sent_4_o1":     dict(kind="sent", n=4, overlap=1, info="Groups of 4 whole sentences, 1 sentence of overlap."),
    "sent_8_o2":     dict(kind="sent", n=8, overlap=2, info="Groups of 8 whole sentences, 2 sentences of overlap."),
    "semantic_p90":  dict(kind="semantic", percentile=90, max_words=300, info="Splits where the topic changes most, up to 300 words."),
    "structure_300": dict(kind="structure", max_words=300, info="Follows headings and paragraphs, up to 300 words per chunk."),
    "parent_child":  dict(kind="parent_child", child=100, parent=400, info="Searches small 100-word pieces, shows the 400-word section around them."),
}


def make_spans(text, spec, embed_fn=None):
    """Returns (start, end, parent_start, parent_end) per chunk; parent_* is None except for parent_child."""
    k = spec["kind"]
    if k == "fixed":
        return [(a, b, None, None) for a, b in chunk_fixed(text, spec["size"], spec["overlap"])]
    if k == "sent":
        return [(a, b, None, None) for a, b in chunk_sentences(text, spec["n"], spec["overlap"])]
    if k == "semantic":
        return [(a, b, None, None) for a, b in chunk_semantic(text, embed_fn, spec["percentile"], max_words=spec["max_words"])]
    if k == "structure":
        return [(a, b, None, None) for a, b in chunk_structure(text, spec["max_words"])]
    return chunk_parent_child(text, spec["child"], spec["parent"])


# ----------------------------------------------------------------- documents
def clean(t):
    t = t.replace("\r", "")
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def read_file(p):
    s = p.suffix.lower()
    try:
        if s in (".md", ".txt"):
            return p.read_text(errors="ignore")
        if s in (".html", ".htm"):
            t = re.sub(r"(?is)<(script|style).*?</\1>", " ", p.read_text(errors="ignore"))
            return _html.unescape(re.sub(r"<[^>]+>", " ", t))
        if s == ".pdf":
            from pypdf import PdfReader
            return "\n\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
    except Exception as e:
        st.warning(f"Could not read {p.name}: {e}")
    return None


def list_corpora():
    return sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir()) if DATA_DIR.exists() else []


def load_docs(corpus):
    docs = []
    for p in sorted((DATA_DIR / corpus).glob("*")):
        t = read_file(p)
        if t and len(clean(t).split()) >= 50:
            docs.append((p.stem, clean(t)))
    return docs

# ----------------------------------------------------------------- embeddings, BM25, index
# The embedding models from the study (same names as the heatmap columns) plus an offline baseline.
EMBEDDERS = {
    "bge-small":   dict(kind="hf", model_id="BAAI/bge-small-en-v1.5", q_prefix="Represent this sentence for searching relevant passages: ",
                        info="BAAI bge-small-en-v1.5, 384 dimensions. Needs sentence-transformers."),
    "e5-small":    dict(kind="hf", model_id="intfloat/e5-small-v2", q_prefix="query: ", d_prefix="passage: ",
                        info="intfloat e5-small-v2, 384 dimensions. Needs sentence-transformers."),
    "gte-small":   dict(kind="hf", model_id="thenlper/gte-small",
                        info="thenlper gte-small, 384 dimensions. Needs sentence-transformers."),
    "minilm":      dict(kind="hf", model_id="sentence-transformers/all-MiniLM-L6-v2",
                        info="all-MiniLM-L6-v2, 384 dimensions. Smallest and fastest."),
    "hashing-bow": dict(kind="hash", info="Offline keyword hashing, 512 dimensions. No download, lowest quality."),
}


def hash_embed(texts, dim=512):
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            X[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def get_st_model(model_id):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_id, device="cpu")
    try:
        m.max_seq_length = min(512, m.max_seq_length)
    except Exception:
        pass
    return m


def embed(texts, name, query=False):
    cfg = EMBEDDERS[name]
    if cfg["kind"] == "hash":
        return hash_embed(texts)
    prefix = cfg.get("q_prefix", "") if query else cfg.get("d_prefix", "")
    return get_st_model(cfg["model_id"]).encode([prefix + t for t in texts], normalize_embeddings=True,
                                                batch_size=32, show_progress_bar=False)


TOK = lambda s: re.findall(r"\w+", s.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.tf = [Counter(d) for d in docs]
        self.dl = np.array([len(d) for d in docs], float)
        self.avg = self.dl.mean() if len(docs) else 1.0
        self.k1, self.b = k1, b
        df = Counter(t for c in self.tf for t in c); N = len(docs)
        self.idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, q):
        out = np.zeros(len(self.tf))
        for i, c in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg)
            out[i] = sum(self.idf[t] * c[t] * (self.k1 + 1) / (c[t] + norm) for t in set(q) if t in c)
        return out


@st.cache_resource(show_spinner="Indexing documents (first time for this setting)...")
def build_index(corpus, chunker_name, embedder):
    spec = CHUNKERS[chunker_name]
    sem_name = "minilm" if HAS_ST else "hashing-bow"           # the semantic chunker uses MiniLM, as in the study
    spans, show = [], []
    for doc_id, text in load_docs(corpus):
        for a, b, pa, pb in make_spans(text, spec, lambda x: embed(x, sem_name)):
            spans.append((doc_id, a, b, text[a:b]))
            show.append((doc_id, pa, pb, text[pa:pb]) if pa is not None else (doc_id, a, b, text[a:b]))
    if not spans:
        return None
    return {"spans": spans, "show": show, "X": embed([x[3] for x in spans], embedder),
            "bm25": BM25([TOK(x[3]) for x in spans]), "embedder": embedder}


def retrieve(idx, question, mode, k=TOPK):
    q = embed([question], idx["embedder"], query=True)[0]
    ranked = [int(i) for i in np.argsort(-(idx["X"] @ q))[:30]]
    if mode == "hybrid":                                   # reciprocal rank fusion of dense and BM25 rankings
        bm_rank = [int(i) for i in np.argsort(-idx["bm25"].scores(TOK(question)))[:30]]
        score = {}
        for r, i in enumerate(ranked):
            score[i] = score.get(i, 0) + 1 / (60 + r + 1)
        for r, i in enumerate(bm_rank):
            score[i] = score.get(i, 0) + 1 / (60 + r + 1)
        ranked = sorted(score, key=score.get, reverse=True)
    out, seen = [], set()
    for i in ranked:                                       # parent_child: several children can share one parent
        u = idx["show"][i]
        if u[:3] in seen:
            continue
        seen.add(u[:3]); out.append(u)
        if len(out) == k:
            break
    return out


# ----------------------------------------------------------------- answers
def build_prompt(question, units):
    ctx = "\n\n".join(f"[{i}] {u[3]}" for i, u in enumerate(units, 1))
    return GEN_PROMPT + ctx + f"\n\nQuestion: {question}\nAnswer:"


def demo_answer(question, units):
    """Offline mode: returns the single sentence that shares the most question words, with its source number."""
    qw = set(TOK(question)) - STOP; best = None
    for n, (_, _, _, t) in enumerate(units, 1):
        for line in t.split("\n"):
            if line.lstrip().startswith("#"):
                continue                                   # skip headings
            for sent in re.split(r"(?<=[.!?])\s+", line):
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                overlap = len(qw & set(TOK(sent)))
                if overlap and (best is None or overlap > best[0]):
                    best = (overlap, sent, n)
    if not best:
        return "I don't know."
    return f"{best[1][:400]} [{best[2]}]"


# ----------------------------------------------------------------- language-model providers
# Most providers speak the OpenAI "chat completions" format, so one code path serves them; Gemini and Anthropic have their own.
PROVIDERS = {
    "Gemini":         dict(kind="gemini", base="https://generativelanguage.googleapis.com/v1beta", hint="AIza...",
                           url="aistudio.google.com/apikey", example="gemini-2.5-flash"),
    "OpenRouter":     dict(kind="openai", base="https://openrouter.ai/api/v1", hint="sk-or-...",
                           url="openrouter.ai/keys", example="meta-llama/llama-3.3-70b-instruct:free"),
    "OpenAI":         dict(kind="openai", base="https://api.openai.com/v1", hint="sk-...",
                           url="platform.openai.com/api-keys", example="gpt-4o-mini"),
    "Groq":           dict(kind="openai", base="https://api.groq.com/openai/v1", hint="gsk_...",
                           url="console.groq.com/keys", example="llama-3.3-70b-versatile"),
    "SiliconFlow":    dict(kind="openai", base="https://api.siliconflow.com/v1", hint="sk-...",
                           url="cloud.siliconflow.com", example="Qwen/Qwen2.5-7B-Instruct", list_params={"sub_type": "chat"}),
    "Together AI":    dict(kind="openai", base="https://api.together.xyz/v1", hint="your Together key",
                           url="api.together.ai/settings/api-keys", example="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "DeepSeek":       dict(kind="openai", base="https://api.deepseek.com/v1", hint="sk-...",
                           url="platform.deepseek.com/api_keys", example="deepseek-chat"),
    "Mistral":        dict(kind="openai", base="https://api.mistral.ai/v1", hint="your Mistral key",
                           url="console.mistral.ai/api-keys", example="mistral-small-latest"),
    "Z.ai":           dict(kind="openai", base="https://api.z.ai/api/paas/v4", hint="your Z.ai key",
                           url="z.ai/manage-apikey/apikey-list", example="glm-4.5-air",
                           fallback=["glm-5.1", "glm-4.6", "glm-4.5", "glm-4.5-air", "glm-4.5-flash"]),
    "Anthropic":      dict(kind="anthropic", base="https://api.anthropic.com/v1", hint="sk-ant-...",
                           url="console.anthropic.com/settings/keys", example="claude-haiku-5-5"),
    "Ollama (local)": dict(kind="ollama", base="http://localhost:11434", hint="", url="ollama.com/download",
                           example="llama3.2", needs_key=False),
}
NOT_CHAT = ("embed", "whisper", "tts", "dall-e", "image", "imagen", "moderation", "audio", "transcribe",
            "realtime", "rerank", "guard", "ocr", "sora", "veo", "live", "stable-diffusion", "flux")


def clean_key(k):
    """Removes spaces, quotes and a leading 'Bearer' that often come along when a key is pasted."""
    k = (k or "").strip().strip("\"'").strip()
    return k[7:].strip() if k.lower().startswith("bearer ") else k


def _api_error(r):
    try:
        j = r.json(); e = j.get("error", j) if isinstance(j, dict) else j
        msg = e.get("message") if isinstance(e, dict) else e
        return f"HTTP {r.status_code}: {msg or r.text[:200]}"
    except Exception:
        return f"HTTP {r.status_code}: {r.text[:200]}"


def _headers(prov, key):
    kind = PROVIDERS[prov]["kind"]
    if kind == "gemini":
        return {"x-goog-api-key": key}
    if kind == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    if prov == "OpenRouter":
        h["X-Title"] = "RAG Studio"
    return h


def _http(method, url, prov, **kw):
    """requests wrapper with a readable message when the server cannot be reached."""
    import requests
    try:
        return requests.request(method, url, timeout=kw.pop("timeout", 120), **kw)
    except requests.exceptions.ConnectionError:
        hint = " Is Ollama running? Start it with: ollama serve" if PROVIDERS[prov]["kind"] == "ollama" else " Check your internet connection."
        raise RuntimeError(f"Cannot reach {prov} at {url.split('/v1')[0].split('/api')[0]}.{hint}")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"{prov} took too long to respond. Try again or pick a faster model.")


def _post(url, prov, headers, body):
    r = _http("POST", url, prov, headers=headers, json=body)
    if r.status_code == 400 and "temperature" in body and "temperature" in r.text.lower():
        body = {k: v for k, v in body.items() if k != "temperature"}      # some newer models only allow the default temperature
        r = _http("POST", url, prov, headers=headers, json=body)
    return r


def _root(base):
    b = base.rstrip("/")
    return b[:-3] if b.endswith("/v1") else b


def call_llm(provider, key, model, prompt, units, question, base=None):
    if provider == OFFLINE:
        return demo_answer(question, units)
    cfg = PROVIDERS[provider]; kind = cfg["kind"]
    base = (base or cfg["base"]).strip().rstrip("/")
    if cfg.get("needs_key", True) and not key:
        raise RuntimeError(f"No {provider} API key entered in the sidebar.")
    if not model:
        raise RuntimeError("Choose a model in the sidebar, or type a model id.")
    h = _headers(provider, key)
    if kind == "gemini":
        r = _http("POST", f"{base}/models/{model}:generateContent", provider, headers=h,
                  json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}})
    elif kind == "anthropic":
        r = _post(f"{base}/messages", provider, h, {"model": model, "max_tokens": 1024, "temperature": 0,
                                                    "messages": [{"role": "user", "content": prompt}]})
    else:
        url = f"{_root(base)}/v1/chat/completions" if kind == "ollama" else f"{base}/chat/completions"
        r = _post(url, provider, h, {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
    if r.status_code in (401, 403) and cfg.get("needs_key", True):
        raise RuntimeError(f"{provider} rejected the API key ({r.status_code}). Paste the key again in the sidebar. ({_api_error(r)})")
    if r.status_code != 200:
        raise RuntimeError(_api_error(r))
    data = r.json()
    if isinstance(data, dict) and data.get("error") and not data.get("choices"):
        raise RuntimeError(_api_error(r) if not isinstance(data["error"], dict) else data["error"].get("message", str(data["error"])))
    if kind == "gemini":
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        return "".join(x.get("text", "") for x in parts)
    if kind == "anthropic":
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    c = data["choices"][0]["message"].get("content")
    if isinstance(c, list):
        c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return c or ""


def fetch_models(provider, key, base=None):
    """Returns ([(label, model_id)], note). Raises RuntimeError with a readable message if the key or server is wrong."""
    cfg = PROVIDERS[provider]; kind = cfg["kind"]
    base = (base or cfg["base"]).strip().rstrip("/")
    h = _headers(provider, key)

    def check(r):
        if r.status_code in (401, 403):
            raise RuntimeError(f"{provider} says this API key is not valid ({_api_error(r)}). Create a key at {cfg['url']} and paste it again.")
        if r.status_code != 200:
            raise RuntimeError(_api_error(r))
        return r.json()

    if kind == "gemini":
        out, token = [], None
        while True:
            data = check(_http("GET", f"{base}/models", provider, headers=h, timeout=30,
                               params={"pageSize": 100, **({"pageToken": token} if token else {})}))
            for m in data.get("models", []):
                mid = m["name"].replace("models/", "")
                if "generateContent" in m.get("supportedGenerationMethods", []) and not any(x in mid.lower() for x in NOT_CHAT):
                    out.append((mid, mid))
            token = data.get("nextPageToken")
            if not token:
                return sorted(out), ""
    if kind == "anthropic":
        data = check(_http("GET", f"{base}/models", provider, headers=h, timeout=30, params={"limit": 1000}))
        return [(m["id"] if not m.get("display_name") else f'{m["id"]}  ({m["display_name"]})', m["id"]) for m in data.get("data", [])], ""
    if kind == "ollama":
        data = check(_http("GET", f"{_root(base)}/api/tags", provider, headers=h, timeout=15))
        names = sorted(m["name"] for m in data.get("models", []))
        if not names:
            return [], "Ollama is running but has no models yet. Download one with: ollama pull llama3.2"
        return [(n, n) for n in names], ""
    if provider == "OpenRouter":                                          # the model list is public; /key checks the key itself
        check(_http("GET", f"{base}/key", provider, headers=h, timeout=30))
    r = _http("GET", f"{base}/models", provider, headers=h, timeout=30, params=cfg.get("list_params"))
    if r.status_code == 404 and cfg.get("fallback"):
        return [(m, m) for m in cfg["fallback"]], "This provider has no model-list endpoint, so a built-in list is shown. You can type any other model id."
    data = check(r)
    items = data if isinstance(data, list) else data.get("data", [])      # Together returns a bare list
    rows = []
    for m in items:
        mid = m.get("id") or m.get("name") or ""
        if not mid or any(x in mid.lower() for x in NOT_CHAT):
            continue
        if m.get("type") not in (None, "chat", "language", "code"):         # Together tags each model with a type
            continue
        pr = m.get("pricing") or {}
        free = provider == "OpenRouter" and (mid.endswith(":free") or (str(pr.get("prompt", "1")) == "0" and str(pr.get("completion", "1")) == "0"))
        rows.append((mid + ("  (free)" if free else ""), mid))
    if not rows and cfg.get("fallback"):
        return [(m, m) for m in cfg["fallback"]], "No models were listed, so a built-in list is shown. You can type any other model id."
    return sorted(rows, key=lambda x: ("(free)" not in x[0], x[1])), ""


_ENTITY_WORDS = {"amp", "lt", "gt", "quot", "39", "x27"}


def highlight(text, question):
    words = sorted({w for w in TOK(question) if w not in STOP and w not in _ENTITY_WORDS and len(w) > 2}, key=len, reverse=True)
    t = _html.escape(text)
    if words:
        t = re.sub(r"\b(" + "|".join(map(re.escape, words)) + r")\b", lambda m: f"<mark>{m.group(0)}</mark>", t, flags=re.I)
    return t.replace("\n", "<br>")


def show_sources(units, question):
    for n, (d, s_, e_, t) in enumerate(units, 1):
        with st.expander(f"Source [{n}]   {d}   |   {len(t.split())} words"):
            st.markdown(f'<div class="src">{highlight(t, question)}</div>', unsafe_allow_html=True)
            st.caption(f"Characters {s_} to {e_} of the document. Words from your question are highlighted.")


# ----------------------------------------------------------------- results helpers (heatmap + advisor)
def load_results():
    if not RESULTS_PATH.exists():
        return None
    return pd.read_csv(RESULTS_PATH, dtype={"stage": str, "quant": str})


def heatmap_frame(raw, corp, mode, metric):
    """chunker x embedder grid. Uses the notebook's Stage 2 runs (the full grid) and ignores dimension/quantisation variants,
    so every cell is a like-for-like comparison and nothing is averaged across different setups."""
    d = raw[(raw.corpus == corp) & (raw["mode"] == mode) & (raw.quant.fillna("") == "") & (raw.stage != "3b")]
    d = d[~d.key.str.contains("@", regex=False)].dropna(subset=[metric])
    s2 = d[d.stage == "2"]
    if s2.embedder.nunique() >= 2:
        d = s2
    pv = d.pivot_table(index="chunker", columns="embedder", values=metric, aggfunc="mean")
    full_rows = pv.dropna(how="any")                       # only chunkers tested with every embedder -> no empty cells
    if pv.shape[1] >= 2 and not full_rows.empty:
        pv = full_rows
    return pv.dropna(axis=1, how="all")


def heatmap_chart(pv, label, higher_better):
    """Altair heatmap: rows = chunker, columns = embedder, colour = metric. Best cell gets a gold outline."""
    long = pv.reset_index().melt(id_vars="chunker", var_name="embedder", value_name="value").dropna()
    best_val = long.value.max() if higher_better else long.value.min()
    long["is_best"] = long.value == best_val
    lo, hi = float(long.value.min()), float(long.value.max())
    cut = lo + (hi - lo) * 0.5
    scale = alt.Scale(scheme="tealblues" if higher_better else "orangered", reverse=not higher_better)
    band = dict(paddingInner=0.06, paddingOuter=0.02)
    base = alt.Chart(long).encode(
        x=alt.X("embedder:N", title="Embedding model", scale=alt.Scale(**band), axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=13)),
        y=alt.Y("chunker:N", title="Chunking method", scale=alt.Scale(**band), axis=alt.Axis(labelFontSize=13, titleFontSize=13, labelLimit=260)))
    rect = base.mark_rect(cornerRadius=6).encode(
        color=alt.Color("value:Q", scale=scale, legend=alt.Legend(title=label, orient="bottom", direction="horizontal", gradientLength=320)),
        tooltip=[alt.Tooltip("chunker:N", title="Chunking"), alt.Tooltip("embedder:N", title="Embedding"),
                 alt.Tooltip("value:Q", title=label, format=".3f")])
    light_cell = (alt.datum.value < cut) if higher_better else (alt.datum.value > cut)   # pale cells need dark text
    text = base.mark_text(fontSize=17, fontWeight="bold").encode(
        text=alt.Text("value:Q", format=".2f"), color=alt.condition(light_cell, alt.value("#14213d"), alt.value("#ffffff")))
    best = base.mark_rect(fill=None, stroke="#ffb703", strokeWidth=4, cornerRadius=6).transform_filter(alt.datum.is_best)
    return (rect + text + best).properties(height=max(200, 78 * pv.shape[0] + 40)).configure_view(strokeWidth=0)


def bar_chart(df, x, y, color_field, title, fmt=".3f"):
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:N", title=None, axis=alt.Axis(labelAngle=-20, labelLimit=200)),
        y=alt.Y(f"{y}:Q", title=title),
        color=alt.Color(f"{color_field}:N", legend=alt.Legend(title=None), scale=alt.Scale(scheme="tableau10")))
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    return bars.properties(height=300).configure_view(strokeWidth=0)


def plain_findings(pv, label, higher_better, corpus, mode):
    """Auto-generated plain-language reading of the heatmap, so the numbers can be narrated."""
    stacked = pv.stack().dropna()
    if stacked.empty:
        return []
    pick = stacked.idxmax() if higher_better else stacked.idxmin()
    worst = stacked.idxmin() if higher_better else stacked.idxmax()
    ties = int((stacked.round(6) == round(float(stacked[pick]), 6)).sum()) - 1
    tie_txt = f" ({ties} other cell{'s' if ties != 1 else ''} tied with it, so these setups are equally good on this metric)" if ties > 0 else ""
    out = [f"Best cell: <b>{pick[0]}</b> with <b>{pick[1]}</b> scores <b>{stacked[pick]:.3f}</b> on {label}{tie_txt}.",
           f"Weakest cell: <b>{worst[0]}</b> with <b>{worst[1]}</b> scores <b>{stacked[worst]:.3f}</b>."]
    spread = stacked.max() - stacked.min()
    out.append(f"The gap between best and worst is <b>{spread:.3f}</b>, so the choice of setup {'matters a lot' if spread > 0.1 else 'matters only a little'} for this corpus.")
    row_avg, col_avg = pv.mean(axis=1), pv.mean(axis=0)
    if pv.shape[0] > 1:
        rb = row_avg.idxmax() if higher_better else row_avg.idxmin()
        out.append(f"Across all embedding models, the best chunking method on average is <b>{rb}</b> (mean {row_avg[rb]:.3f}).")
    if pv.shape[1] > 1:
        cb = col_avg.idxmax() if higher_better else col_avg.idxmin()
        out.append(f"Across all chunking methods, the best embedding model on average is <b>{cb}</b> (mean {col_avg[cb]:.3f}).")
    if pv.shape[0] > 1 and pv.shape[1] > 1:
        r_gap, c_gap = row_avg.max() - row_avg.min(), col_avg.max() - col_avg.min()
        driver = "chunking method" if r_gap > c_gap else "embedding model"
        out.append(f"Comparing the row averages with the column averages, the <b>{driver}</b> changes the score more than the other choice does.")
    return out

# ----------------------------------------------------------------- ITU-T F.748.52 helpers
F748_URL = "https://handle.itu.int/11.1002/1000/16242"
STATUS_COLORS = {"Implemented": "#12a39a", "Partial": "#f2a900", "Out of scope": "#8a94a6", "Not implemented": "#d64545"}
F748_ALIGNMENT_DEFAULT = [
    ("7.1", "Data access", "Partial", "Text, HTML, PDF and markdown corpora; no images or video; not large-scale"),
    ("7.2", "Data segmentation", "Implemented", "Fixed-length, sentence/punctuation, semantic, structure-aware and parent-child; configurable size and overlap. Recursive-character splitting not included"),
    ("7.3", "Data indexing", "Partial", "Dense vector index (FAISS) and inverted-index scoring (BM25); no layer, list or graph indexes"),
    ("7.4", "Data updates", "Out of scope", "Static corpus; embeddings are cached per text set, not updated incrementally or in real time"),
    ("7.5", "Data storage", "Partial", "Index compression studied (dimension truncation, binary codes); no distributed storage, backup or partitioning"),
    ("8.1.1", "Retrieval modes", "Implemented", "Semantic (dense), keyword (BM25), hybrid and combined retrieval with reranking; no MMR, iterative or cross-modal retrieval"),
    ("8.1.2", "Retrieval quality", "Implemented", "P_A, R_A and MAP at n = 5 (eq. 1-4) plus span-based recall, precision and f_ctx"),
    ("8.2.1", "Answer relevance (Table A.1)", "Implemented", "5-level LLM grade; multi-reviewer averaging through human rating sheets"),
    ("8.2.2", "Answer reliability / faithfulness (Table A.2)", "Implemented", "5-level LLM grade against the retrieved context (clause text cites A.1; read as A.2)"),
    ("8.2.3", "Answer truthfulness (Table A.3)", "Implemented", "5-level LLM grade against the reference answer and context"),
    ("8.2.4", "Answer correctness (eq. 5)", "Implemented", "Exact-match rate plus a judged correctness score (exact match is very strict for free-text answers)"),
    ("8.2.5", "Context relevance (Table A.4)", "Not implemented", "Table A.4 grades coherence with preceding and following text; not meaningful for single-turn QA"),
    ("8.3.1-8.3.5", "Optimization capabilities", "Partial", "Reranking, hybrid fusion, parent-child retrieval, constrained prompt template and context numbering; no query rewrite, HyDE, embedding fine-tuning, hallucination correction or feedback loop"),
    ("9.1", "Application maturity", "Partial", "Modular components (loader, embedder, vector store, retriever, generator), configurable by script and by the Streamlit app; text only; no API or SDK"),
    ("9.2.1", "Robustness (noise, counterfactual)", "Not implemented", "No noisy-document or counterfactual test sets"),
    ("9.2.2", "Security", "Out of scope", "Research prototype; only public documents are sent to the LLM provider"),
    ("9.2.3", "Throughput (TPS)", "Implemented", "Retrieval-level TPS with 1 and several threads (notebook, section 14b)"),
    ("9.2.4", "Response time", "Implemented", "Retrieval response time and end-to-end non-streaming response time (notebook, section 14b)"),
    ("9.2.5", "Negative rejection (eq. 11-13)", "Implemented", "P_ERR, P_EAR, P_NRR on the unanswerable and answerable questions of Stage 3b"),
    ("9.2.6", "Information integration (eq. 14)", "Implemented", "P_MIR on the multi-hop questions of Stage 3b"),
]
F748_RET = {   # column in results.csv -> (label, plain-words meaning)
    "map5": ("MAP@5", "Average precision at n = 5 (eq. 4). Rewards finding the evidence early in the top 5; 1.0 means every evidence passage sits at the top."),
    "std_precision": ("P_A", "Accuracy (eq. 1): the share of the 5 retrieved passages that are relevant."),
    "std_recall": ("R_A", "Recall (eq. 2-3): the share of the gold evidence passages that appear among the 5 retrieved."),
}
F748_SCALES = {
    "Answer relevance (Annex A, clause 8.2.1)": [
        (1, "Unrelated to the question"), (2, "Only touches the fringes"), (3, "Partly relevant but misses the core points"),
        (4, "Closely related and useful"), (5, "Answers the core points directly, accurately and usefully")],
    "Faithfulness: match with the retrieved context (clause 8.2.2)": [
        (1, "Only keyword overlap, no real link to the context"), (2, "Refers to the context but much is fabricated"),
        (3, "Mostly consistent, with a small amount of fabricated content"), (4, "Consistent but incomplete (omissions)"),
        (5, "Consistent and complete")],
    "Truthfulness: consistency with the reference and context (clause 8.2.3)": [
        (1, "Contradicts them or is false"), (2, "Partly consistent, with significant errors or omissions"),
        (3, "Generally consistent, with minor inaccuracies"), (4, "Highly consistent and reliable"),
        (5, "Fully consistent, no errors or omissions")],
}


def load_alignment():
    f = RESULTS_PATH.parent / "f748_alignment.csv"
    try:
        if f.exists():
            d = pd.read_csv(f)
            if {"Clause", "Requirement", "Status"} <= set(d.columns) and len(d):
                d = d.rename(columns={"How this study covers it": "Coverage"}) if "Coverage" not in d.columns else d
                return d[["Clause", "Requirement", "Status", "Coverage"]]
    except Exception:
        pass
    return pd.DataFrame(F748_ALIGNMENT_DEFAULT, columns=["Clause", "Requirement", "Status", "Coverage"])


def read_out(name):
    f = RESULTS_PATH.parent / name
    try:
        return pd.read_csv(f) if f.exists() else None
    except Exception:
        return None


def style_status(df):
    def col(v):
        return f"background-color: {STATUS_COLORS.get(v, '#888888')}33; font-weight: 600;"
    return df.style.map(col, subset=["Status"]) if hasattr(df.style, "map") else df.style.applymap(col, subset=["Status"])


def grouped_bar(df, id_col, value_cols, title, domain=None):
    long = df.melt(id_vars=id_col, value_vars=value_cols, var_name="Measure", value_name="Value").dropna()
    if long.empty:
        return None
    ch = alt.Chart(long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X(f"{id_col}:N", title=None, axis=alt.Axis(labelAngle=-25, labelLimit=260)), xOffset="Measure:N",
        y=alt.Y("Value:Q", title=title, scale=alt.Scale(domain=domain) if domain else alt.Undefined),
        color=alt.Color("Measure:N", scale=alt.Scale(scheme="tableau10"), legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[id_col, "Measure", alt.Tooltip("Value:Q", format=".2f")])
    return ch.properties(height=330)


def fmt_pct(v):
    return "n/a" if pd.isna(v) else f"{100 * v:.0f}%"


def fmt_lvl(v):
    return "n/a" if pd.isna(v) else f"{v:.2f} / 5"


# ----------------------------------------------------------------- sidebar
st.session_state.setdefault("models", {})
st.session_state.setdefault("api_keys", {})
st.session_state.setdefault("bases", {})
with st.sidebar:
    st.markdown("### Language model")
    st.caption("Used by the Ask your documents tab. Without a key, the offline demo returns the best-matching sentence.")
    provider = st.selectbox("Provider", [OFFLINE] + list(PROVIDERS), key="prov")
    api_key, base_url, model_id = "", None, "offline-demo"
    if provider != OFFLINE:
        cfg = PROVIDERS[provider]; needs_key = cfg.get("needs_key", True)
        if needs_key:
            api_key = clean_key(st.text_input(f"{provider} API key", type="password", key=f"key_{provider}",
                                              value=st.session_state["api_keys"].get(provider, ""), placeholder=cfg["hint"],
                                              help=f"Get a key at {cfg['url']}. Kept in this browser session only; never saved to disk."))
            st.session_state["api_keys"][provider] = api_key
        with st.expander("Advanced: server address", expanded=(cfg["kind"] == "ollama")):
            base_url = st.text_input("Base URL", key=f"base_{provider}", value=st.session_state["bases"].get(provider, cfg["base"]),
                                     help="Change this for a regional endpoint or a remote Ollama server.").strip()
            st.session_state["bases"][provider] = base_url
        if st.button(f"Load {provider} models", key=f"load_{provider}", disabled=(needs_key and not api_key), width="stretch"):
            try:
                with st.spinner("Loading models..."):
                    found, note = fetch_models(provider, api_key, base_url)
                st.session_state["models"][provider] = found
                if found:
                    st.success(f"{len(found)} models available")
                if note:
                    st.info(note)
            except Exception as e:
                st.error(f"Could not load models: {e}")
        found = st.session_state["models"].get(provider) or []
        picked = dict(found)[st.selectbox("Model", [l for l, _ in found], key=f"model_{provider}")] if found else ""
        custom = st.text_input("Or type a model id", key=f"custom_{provider}", placeholder=f"e.g. {cfg['example']}")
        model_id = custom.strip() or picked
        if not model_id:
            st.caption("Load the models for this key, or type a model id, before asking a question.")

# ----------------------------------------------------------------- header
st.markdown("""
<div class="hero">
  <h1>RAG Studio</h1>
  <p>How chunking, embedding and retrieval choices change the quality of a document question-answering system.</p>
</div>
""", unsafe_allow_html=True)

tab_res, tab_itu, tab_ask = st.tabs(["Results and Advisor", "ITU-T F.748.52", "Ask your documents"])

# ================================================================= TAB 1: results + advisor
with tab_res:
    raw = load_results()
    if raw is None:
        st.info("Run the notebook first; it writes outputs/results.csv.")
    else:
        base = raw[raw.stage != "3b"].drop_duplicates("key").dropna(subset=["f_ctx"])
        if base.empty:
            st.warning("outputs/results.csv has no retrieval results yet.")
        else:
            # ---- controls
            c1, c2, c3 = st.columns([1.2, 1.2, 1])
            corp = c1.selectbox("Document type (corpus)", sorted(base.corpus.unique()), key="res_corpus")
            metric = c2.selectbox("Metric on the heatmap", list(METRICS), format_func=lambda m: METRICS[m][0], key="res_metric")
            modes_here = sorted(base[base.corpus == corp]["mode"].dropna().unique(), key=lambda m: ["dense", "hybrid", "hybrid+rerank"].index(m) if m in ["dense", "hybrid", "hybrid+rerank"] else 9)
            res_mode = c3.selectbox("Retrieval mode", modes_here, key="res_mode")
            label, higher_better, meaning = METRICS[metric]

            full = base[(base.corpus == corp) & (base.quant.fillna("") == "") & (~base.key.str.contains("@", regex=False))]

            # ---- Section 1: heatmap
            st.markdown("## 1. The heatmap: which combination works best?")
            step(1, "Each <b>row</b> is a way of cutting documents into chunks. Each <b>column</b> is an embedding model that turns chunks into vectors.")
            step(2, f"Each <b>cell</b> shows <b>{label}</b> for that pair, averaged over all test questions for the <b>{corp}</b> corpus. {meaning}")
            step(3, "<b>Darker colour means better</b>" + ("" if higher_better else " (colours are reversed here because lower is better)") + ". The cell with the gold outline is the best one.")

            pv = heatmap_frame(raw, corp, res_mode, metric)
            if pv.empty or pv.shape[1] < 2:
                st.info("The heatmap needs at least two embedding models tested on the same chunking methods. "
                        f"That comparison was only run for dense retrieval, so choose 'dense' as the retrieval mode (current: {res_mode}).")
            else:
                st.caption(f"Showing the {pv.shape[0]} chunking methods that were tested with all {pv.shape[1]} embedding models, "
                           "so every cell is a fair, like-for-like comparison.")
                st.altair_chart(heatmap_chart(pv, label, higher_better), width="stretch")
                st.markdown("#### What the heatmap says, in plain words")
                for line in plain_findings(pv, label, higher_better, corp, res_mode):
                    st.markdown(f"- {line}", unsafe_allow_html=True)
                with st.expander("Show the numbers behind the heatmap"):
                    st.dataframe(pv.round(3), width="stretch")

            # ---- Section 2: effect of retrieval mode
            st.markdown("## 2. Does a smarter retrieval mode pay off?")
            st.caption("Same chunking methods, three ways of searching: dense (meaning only), hybrid (meaning plus keywords) and hybrid with a reranker.")
            s3a = raw[(raw.corpus == corp) & (raw.stage == "3a") & (raw.quant.fillna("") == "")].dropna(subset=["f_ctx"])
            mode_src = s3a if not s3a.empty else full
            mode_df = mode_src.groupby(["chunker", "mode"], as_index=False)["f_ctx"].mean() if not mode_src.empty else pd.DataFrame()
            if mode_df.empty or mode_df["mode"].nunique() < 2:
                st.info("Only one retrieval mode was run for this corpus, so there is nothing to compare.")
            else:
                g1, g2 = st.columns(2)
                with g1:
                    st.markdown("**Quality (f_ctx) by chunking method**")
                    ch = alt.Chart(mode_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                        x=alt.X("chunker:N", title=None, axis=alt.Axis(labelAngle=-20, labelLimit=180)),
                        xOffset="mode:N", y=alt.Y("f_ctx:Q", title="f_ctx"),
                        color=alt.Color("mode:N", legend=alt.Legend(title="Retrieval mode"), scale=alt.Scale(scheme="tableau10")),
                        tooltip=["chunker", "mode", alt.Tooltip("f_ctx:Q", format=".3f")]).properties(height=320)
                    st.altair_chart(ch, width="stretch")
                with g2:
                    lat = mode_src.dropna(subset=["lat_p50_ms"]).groupby("mode", as_index=False)["lat_p50_ms"].median()
                    st.markdown("**Median time per question (ms)**")
                    if lat.empty:
                        st.caption("No latency measurements available.")
                    else:
                        cl = alt.Chart(lat).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                            x=alt.X("mode:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("lat_p50_ms:Q", title="Latency p50 (ms)"),
                            color=alt.Color("mode:N", legend=None, scale=alt.Scale(scheme="tableau10")),
                            tooltip=["mode", alt.Tooltip("lat_p50_ms:Q", format=".0f")]).properties(height=320)
                        st.altair_chart(cl, width="stretch")
                best_mode = mode_df.groupby("mode")["f_ctx"].mean().sort_values(ascending=False)
                callout(f"On average, <b>{best_mode.index[0]}</b> gives the best quality ({best_mode.iloc[0]:.3f} f_ctx). "
                        "Compare this with the right-hand chart to decide whether the extra time is worth the quality gain.")

            # ---- Section 3: advisor
            st.markdown("## 3. Advisor: which setup should I use?")
            st.caption("Pick what matters most. The advisor ranks every tested setup for this corpus using that priority.")
            prof = st.radio("What matters most?", list(PROFILES), index=1, horizontal=True, key="adv_prof")
            d = full.copy()
            if d.empty:
                st.warning("No configurations for this document type.")
            else:
                def mm(s):
                    s = s.astype(float); r = s.max() - s.min()
                    return (s - s.min()) / r if r > 0 else pd.Series(1.0, index=s.index)
                w = PROFILES[prof]; lat_ = d.lat_p50_ms.fillna(d.lat_p50_ms.median()).fillna(0)
                d["score"] = (w[0] * mm(d.f_ctx) + w[1] * (1 - mm(np.log1p(lat_))) + w[2] * (1 - mm(np.log1p(d.index_mb.fillna(0))))) / sum(w)
                d = d.sort_values(["score", "f_ctx"], ascending=False)
                b = d.iloc[0]
                st.markdown(f"""
                <div class="rec">
                  <div class="label">Recommended setup ({prof.lower()})</div>
                  <div class="title">{b.chunker} &nbsp;+&nbsp; {b.embedder} &nbsp;+&nbsp; {b['mode']}</div>
                </div>""", unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("f_ctx", f"{b.f_ctx:.3f}")
                m2.metric("Recall", f"{b.recall:.3f}")
                m3.metric("Latency p50", f"{b.lat_p50_ms:.0f} ms" if pd.notna(b.lat_p50_ms) else "n/a")
                m4.metric("Index size", f"{b.index_mb:.2f} MB" if pd.notna(b.index_mb) else "n/a")

                st.markdown("**Top 10 setups for this priority**")
                top = d.head(10).copy()
                top["setup"] = top.chunker + " | " + top.embedder + " | " + top["mode"]
                rank_chart = alt.Chart(top).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color=PALETTE["accent"]).encode(
                    y=alt.Y("setup:N", sort="-x", title=None, axis=alt.Axis(labelLimit=420)),
                    x=alt.X("score:Q", title="Advisor score (0 to 1)", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["setup", alt.Tooltip("score:Q", format=".3f"), alt.Tooltip("f_ctx:Q", format=".3f"),
                             alt.Tooltip("lat_p50_ms:Q", format=".0f"), alt.Tooltip("index_mb:Q", format=".2f")]).properties(height=34 * len(top) + 40)
                st.altair_chart(rank_chart, width="stretch")
                with st.expander("Show the full ranking table"):
                    st.dataframe(d.head(10)[["chunker", "embedder", "mode", "f_ctx", "recall", "precision", "lat_p50_ms", "index_mb", "score"]].round(3),
                                 width="stretch", hide_index=True)
                st.caption("Ranked within this document type, with full-size vectors only. Scores use the priority you picked. "
                           "Latency comes from a small CPU test, so treat small differences as noise.")

# ================================================================= TAB 2: ITU-T F.748.52
with tab_itu:
    ri = load_results()
    align = load_alignment()
    align["Coverage"] = align["Coverage"].astype(str).str.replace("(cell above)", "(notebook, section 14b)", regex=False)
    st.markdown("## ITU-T F.748.52 evaluation")
    st.caption("International standard for evaluating retrieval-augmented generation (RAG).")
    with st.container(border=True):
        st.markdown("**ITU-T Recommendation F.748.52 (03/2025)**: Requirements and evaluation methods for retrieval augmented generation "
                    "of large scale pre-trained models. Approved 2025-03-16, status in force, maintained by ITU-T Study Group 21. "
                    f"[Read the standard]({F748_URL}).")
        callout("This study is <b>aligned with</b> the standard, <b>not conformant</b>. Full conformance would also need image and video support, "
                "distributed storage and a security system. Section 6 lists every clause and how it is covered.")

    cnt = align.Status.value_counts()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Clauses reviewed", len(align))
    k2.metric("Implemented", int(cnt.get("Implemented", 0)))
    k3.metric("Partial", int(cnt.get("Partial", 0)))
    k4.metric("Out of scope", int(cnt.get("Out of scope", 0)))
    k5.metric("Not implemented", int(cnt.get("Not implemented", 0)))

    have_ret = ri is not None and {"map5", "std_precision", "std_recall"} <= set(ri.columns) and ri.std_precision.notna().any()
    q3_all = ri[ri.stage == "3b"] if ri is not None else pd.DataFrame()
    ans_cols = ["neg_nrr", "neg_ear", "integration_rate", "exact_match", "rel_level", "faith_level", "truth_level", "abstain_rate"]
    have_ans = (not q3_all.empty) and any(c in q3_all.columns and q3_all[c].notna().any() for c in ans_cols)
    perf_df, hv_df = read_out("f748_performance.csv"), read_out("f748_human_vs_llm.csv")
    yn = lambda b: "Yes" if b else "Not yet"

    st.markdown("### What is measured and where")
    at_glance = pd.DataFrame([
        ("8.1.2 (eq. 1-4)", "Accuracy P_A, recall R_A, MAP at n = 5", "Every run", yn(have_ret)),
        ("8.2.1-8.2.3, Annex A", "Answer relevance, faithfulness, truthfulness (5-level scales)", "Stage 3b LLM judge", yn(have_ans)),
        ("8.2.4 (eq. 5)", "Exact-match rate", "Stage 3b", yn(have_ans)),
        ("9.2.5 (eq. 11-13)", "Negative rejection: P_ERR, P_EAR, P_NRR", "Stage 3b", yn(have_ans)),
        ("9.2.6 (eq. 14)", "Information integration P_MIR (multi-document questions)", "Stage 3b", yn(have_ans)),
        ("9.2.3-9.2.4 (eq. 9-10)", "Throughput (TPS) and response time", "Performance test", yn(perf_df is not None)),
        ("Annex A reviewers", "Human vs LLM grades: agreement", "Human rating sheets", yn(hv_df is not None)),
    ], columns=["Clause", "Measure", "Where it is computed", "Results found"])
    st.dataframe(at_glance, width="stretch", hide_index=True)

    # ---- global corpus selector
    corp_choices = sorted(ri.corpus.dropna().unique()) if ri is not None and "corpus" in ri.columns else []
    corp_i = st.selectbox("Document type (corpus) for sections 2 to 4", corp_choices, key="itu_corpus") if corp_choices else None

    # ---------------- 2. retrieval quality
    st.markdown("## 2. Retrieval quality (clause 8.1.2)")
    if not have_ret or corp_i is None:
        st.info("No F.748.52 retrieval measures yet. Re-run the notebook stages so outputs/results.csv contains std_precision, std_recall and map5.")
    else:
        rb = ri[ri.stage != "3b"].drop_duplicates("key").dropna(subset=["std_precision"])
        rbc = rb[rb.corpus == corp_i]
        if rbc.empty:
            st.info("No retrieval results for this corpus.")
        else:
            for m_, (lab_, mean_) in F748_RET.items():
                step({"map5": 1, "std_precision": 2, "std_recall": 3}[m_], f"<b>{lab_}</b>: {mean_}")
            c1, c2 = st.columns(2)
            ret_metric = c1.selectbox("Measure on the heatmap", list(F748_RET), format_func=lambda m: F748_RET[m][0], key="itu_ret_metric")
            ret_modes = sorted(rbc["mode"].dropna().unique(), key=lambda m: ["dense", "hybrid", "hybrid+rerank"].index(m) if m in ["dense", "hybrid", "hybrid+rerank"] else 9)
            ret_mode = c2.selectbox("Retrieval mode", ret_modes, key="itu_ret_mode")
            lab = F748_RET[ret_metric][0]
            pv = heatmap_frame(ri, corp_i, ret_mode, ret_metric)
            if pv.empty or pv.shape[1] < 2:
                st.info("The heatmap needs at least two embedding models tested on the same chunking methods. Choose 'dense' as the retrieval mode.")
            else:
                st.caption(f"Showing the {pv.shape[0]} chunking methods tested with all {pv.shape[1]} embedding models. Darker is better; the gold outline marks the best cell.")
                st.altair_chart(heatmap_chart(pv, lab, True), width="stretch")
                st.markdown("#### What the heatmap says, in plain words")
                for line in plain_findings(pv, lab, True, corp_i, ret_mode):
                    st.markdown(f"- {line}", unsafe_allow_html=True)

            s3 = ri[(ri.corpus == corp_i) & (ri.stage == "3a")].dropna(subset=["map5"])
            if not s3.empty and s3["mode"].nunique() > 1:
                st.markdown("#### Does a smarter retrieval mode help? (MAP@5 by chunking method)")
                md = s3.groupby(["chunker", "mode"], as_index=False)["map5"].mean()
                st.altair_chart(alt.Chart(md).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("chunker:N", title=None, axis=alt.Axis(labelAngle=-20, labelLimit=180)), xOffset="mode:N",
                    y=alt.Y("map5:Q", title="MAP@5", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("mode:N", legend=alt.Legend(title="Retrieval mode"), scale=alt.Scale(scheme="tableau10")),
                    tooltip=["chunker", "mode", alt.Tooltip("map5:Q", format=".3f")]).properties(height=320), width="stretch")

            st.markdown("#### Best configurations by MAP@5")
            t1c1, t1c2 = st.columns([1, 1])
            topn = t1c1.slider("How many to show", 5, 25, 10, key="itu_topn")
            incl = t1c2.checkbox("Include reduced-size and compressed variants", value=True, key="itu_incl")
            t1 = rbc.copy()
            if not incl:
                t1 = t1[(t1["quant"].fillna("") == "") & ~t1.key.str.contains("@", regex=False)]
            t1 = t1.assign(**{"P_A (%)": t1.std_precision * 100, "R_A (%)": t1.std_recall * 100, "MAP@5": t1.map5})
            t1 = t1.sort_values(["MAP@5", "P_A (%)", "R_A (%)"], ascending=False)
            show = t1[["chunker", "embedder", "mode", "dim", "quant", "P_A (%)", "R_A (%)", "MAP@5"]].head(topn).reset_index(drop=True)
            st.dataframe(show, width="stretch", hide_index=True, column_config={
                "P_A (%)": st.column_config.ProgressColumn("P_A (%)", min_value=0, max_value=100, format="%.1f"),
                "R_A (%)": st.column_config.ProgressColumn("R_A (%)", min_value=0, max_value=100, format="%.1f"),
                "MAP@5": st.column_config.ProgressColumn("MAP@5", min_value=0, max_value=1, format="%.3f")})
            full_t1 = rb.assign(**{"P_A (%)": rb.std_precision * 100, "R_A (%)": rb.std_recall * 100, "MAP@5": rb.map5})[
                ["corpus", "chunker", "embedder", "mode", "dim", "quant", "P_A (%)", "R_A (%)", "MAP@5"]]
            st.download_button("Download retrieval table, all corpora (CSV)", full_t1.to_csv(index=False), "f748_retrieval_quality.csv", "text/csv")

    # ---------------- 3. answer quality
    st.markdown("## 3. Answer quality (clauses 8.2, 9.2.5 and 9.2.6)")
    if not have_ans or corp_i is None:
        st.info("Stage 3b has not produced the answer measures yet. Run Stage 3b in the notebook with an LLM configured, then restart this app.")
    else:
        q3 = q3_all
        provs = sorted(q3.provider.dropna().astype(str).unique()) if "provider" in q3.columns else []
        if len(provs) > 1:
            prov_i = st.selectbox("LLM provider", provs, key="itu_prov")
            q3 = q3[q3.provider.astype(str) == prov_i]
        q3c = q3[q3.corpus == corp_i]
        if q3c.empty:
            st.info("No Stage 3b results for this corpus.")
        else:
            step(1, "<b>Relevance, faithfulness, truthfulness</b> are graded 1 to 5 by an LLM judge using the Annex A scales (open the scales below).")
            step(2, "<b>Negative rejection</b>: <b>P_ERR</b> is the share of unanswerable questions the system correctly refused. <b>P_EAR</b> is the share of answerable questions answered correctly. <b>P_NRR</b> is the average of the two.")
            step(3, "<b>P_MIR</b> is the share of multi-document questions answered fully correctly. <b>Exact match</b> counts only answers identical to the reference, so it is very strict.")
            best = q3c.sort_values("f_ctx", ascending=False).iloc[0]
            st.markdown(f"#### Scorecard: best retrieval setup in this corpus")
            st.markdown(f'<span class="pill">{best.chunker}</span><span class="pill">{best.embedder}</span><span class="pill">{best["mode"]}</span>'
                        f'<span class="pill muted">{_html.escape(str(best.get("llm_model", "")))}</span>', unsafe_allow_html=True)
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("P_NRR (negative rejection)", fmt_pct(best.get("neg_nrr")))
            a2.metric("P_ERR (refuses unanswerable)", fmt_pct(best.get("abstain_rate")))
            a3.metric("P_EAR (answers answerable)", fmt_pct(best.get("neg_ear")))
            a4.metric("P_MIR (multi-document)", fmt_pct(best.get("integration_rate")))
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Relevance", fmt_lvl(best.get("rel_level")))
            b2.metric("Faithfulness", fmt_lvl(best.get("faith_level")))
            b3.metric("Truthfulness", fmt_lvl(best.get("truth_level")))
            b4.metric("Exact match", fmt_pct(best.get("exact_match")))

            t2 = q3c.assign(Setup=q3c.chunker + " / " + q3c.embedder + " / " + q3c["mode"],
                            **{"P_ERR (%)": q3c.abstain_rate * 100, "P_EAR (%)": q3c.neg_ear * 100, "P_NRR (%)": q3c.neg_nrr * 100,
                               "P_MIR (%)": q3c.integration_rate * 100, "Exact match (%)": q3c.exact_match * 100,
                               "Relevance (1-5)": q3c.rel_level, "Faithfulness (1-5)": q3c.faith_level, "Truthfulness (1-5)": q3c.truth_level})
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Annex A grades (1 to 5)**")
                ch = grouped_bar(t2, "Setup", ["Relevance (1-5)", "Faithfulness (1-5)", "Truthfulness (1-5)"], "Grade (1 to 5)", [0, 5])
                st.altair_chart(ch, width="stretch") if ch is not None else st.caption("No grades available.")
            with g2:
                st.markdown("**Rates (%)**")
                ch = grouped_bar(t2, "Setup", ["P_ERR (%)", "P_EAR (%)", "P_NRR (%)", "P_MIR (%)", "Exact match (%)"], "Percent", [0, 100])
                st.altair_chart(ch, width="stretch") if ch is not None else st.caption("No rates available.")
            cols_t2 = ["chunker", "embedder", "mode", "P_ERR (%)", "P_EAR (%)", "P_NRR (%)", "P_MIR (%)", "Exact match (%)",
                       "Relevance (1-5)", "Faithfulness (1-5)", "Truthfulness (1-5)"]
            st.dataframe(t2[cols_t2].round(2).reset_index(drop=True), width="stretch", hide_index=True)
            st.download_button("Download answer-quality table (CSV)", t2[["corpus"] + cols_t2].round(3).to_csv(index=False), "f748_answer_quality.csv", "text/csv")
            with st.expander("Annex A grading scales used by the LLM judge"):
                for title_, rows_ in F748_SCALES.items():
                    st.markdown(f"**{title_}**")
                    st.dataframe(pd.DataFrame(rows_, columns=["Grade", "Meaning"]), width="stretch", hide_index=True)

    # ---------------- 4. performance
    st.markdown("## 4. Throughput and response time (clauses 9.2.3 and 9.2.4)")
    if perf_df is None or corp_i is None:
        st.info("No performance results found. Run the F.748.52 performance cell in the notebook (it writes outputs/f748_performance.csv).")
    else:
        pc = perf_df[perf_df.corpus == corp_i] if "corpus" in perf_df.columns else perf_df
        if pc.empty:
            st.info("No performance results for this corpus.")
        else:
            r0 = pc.iloc[0]
            st.caption(f"Measured on {r0.get('chunker', '?')} with {r0.get('embedder', '?')}, on a small CPU test. Treat small differences as noise.")
            tps_cols = [c for c in pc.columns if str(c).startswith("retrieval_TPS")]
            p1, p2, p3 = st.columns(3)
            with p1:
                st.markdown("**Retrieval response time (ms)**")
                st.altair_chart(alt.Chart(pc).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("mode:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("retrieval_response_ms:Q", title="Milliseconds (median)"),
                    color=alt.Color("mode:N", legend=None, scale=alt.Scale(scheme="tableau10")),
                    tooltip=["mode", alt.Tooltip("retrieval_response_ms:Q", format=".1f")]).properties(height=300), width="stretch")
            with p2:
                st.markdown("**Throughput (queries per second)**")
                ch = grouped_bar(pc, "mode", tps_cols, "Queries per second") if tps_cols else None
                st.altair_chart(ch, width="stretch") if ch is not None else st.caption("No throughput values.")
            with p3:
                st.markdown("**End-to-end response time (s)**")
                if "end_to_end_response_s" in pc.columns and pc.end_to_end_response_s.notna().any():
                    st.altair_chart(alt.Chart(pc.dropna(subset=["end_to_end_response_s"])).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                        x=alt.X("mode:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("end_to_end_response_s:Q", title="Seconds (median)"),
                        color=alt.Color("mode:N", legend=None, scale=alt.Scale(scheme="tableau10")),
                        tooltip=["mode", alt.Tooltip("end_to_end_response_s:Q", format=".2f")]).properties(height=300), width="stretch")
                else:
                    st.caption("No end-to-end timing. It needs an LLM configured when the notebook cell runs.")
            st.dataframe(pc.round(2).reset_index(drop=True), width="stretch", hide_index=True)

    # ---------------- 5. human vs LLM
    st.markdown("## 5. Human reviewers versus the LLM judge (Annex A)")
    if hv_df is None:
        st.info("No human-rating comparison yet. In the notebook: run export_rating_sheet(), have two or more reviewers fill "
                "eval/human_ratings_<name>.csv using the Annex A scales, then run human_vs_llm().")
    else:
        step(1, "<b>Reviewer agreement</b> (weighted kappa): how much the human reviewers agree with each other beyond chance. 1 is perfect agreement.")
        step(2, "<b>LLM vs human</b> (Spearman and weighted kappa): how closely the LLM judge follows the average human grade. Higher is better.")
        st.dataframe(hv_df.round(2), width="stretch", hide_index=True)

    # ---------------- 6. alignment
    st.markdown("## 6. Clause-by-clause alignment (all 20 clauses)")
    lc, rc = st.columns([1, 1.6])
    with lc:
        cdf = cnt.rename_axis("Status").reset_index(name="Clauses")
        bar = alt.Chart(cdf).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            y=alt.Y("Status:N", sort=list(STATUS_COLORS), title=None), x=alt.X("Clauses:Q", title="Number of clauses"),
            color=alt.Color("Status:N", legend=None, scale=alt.Scale(domain=list(STATUS_COLORS), range=list(STATUS_COLORS.values()))),
            tooltip=["Status", "Clauses"])
        lbl = bar.mark_text(align="left", dx=4, fontWeight="bold").encode(text="Clauses:Q")
        st.altair_chart((bar + lbl).properties(height=200), width="stretch")
    with rc:
        callout(f"<b>{int(cnt.get('Implemented', 0))}</b> of {len(align)} clauses are fully implemented and <b>{int(cnt.get('Partial', 0))}</b> partly. "
                f"<b>{int(cnt.get('Out of scope', 0))}</b> are out of scope for a research prototype and <b>{int(cnt.get('Not implemented', 0))}</b> are not implemented "
                "(context relevance, Table A.4, and robustness to noisy or counterfactual documents).")
    pick = st.multiselect("Show clauses with status", list(STATUS_COLORS), default=list(STATUS_COLORS), key="itu_status")
    view = align[align.Status.isin(pick)].reset_index(drop=True)
    st.dataframe(style_status(view), width="stretch", hide_index=True, height=min(740, 60 + 38 * len(view)),
                 column_config={"Coverage": st.column_config.TextColumn("How this study covers it", width="large")})
    st.download_button("Download alignment table (CSV)", align.to_csv(index=False), "f748_alignment.csv", "text/csv")

    with st.expander("Notes on how the standard was read"):
        st.markdown("- Clause 8.2.2 points to Table A.1, which we read as Table A.2.\n"
                    "- \"Number of related documents\" in eq. (4) is read as the number of gold evidence spans. A retrieved passage counts as relevant for "
                    "average precision only the first time it reaches a new evidence span.\n"
                    "- \"Multiple reviewers\" is implemented through the human rating sheets in section 5.")

# ================================================================= TAB 3: ask
with tab_ask:
    st.markdown("## Ask your documents")
    st.caption("Choose a pipeline, ask a question, then check the answer against the exact passages it was built from.")
    corpora = list_corpora()
    chunk_ids, emb_ids = list(CHUNKERS), list(EMBEDDERS)

    # ---- 1. pipeline
    with st.container(border=True):
        st.markdown("#### 1. Choose the pipeline")
        c1, c2, c3, c4 = st.columns([1, 1.3, 1.3, 1])
        corpus = c1.selectbox("Corpus", corpora or ["(no corpus found)"], key="ask_corpus", disabled=not corpora)
        c1.caption("Folder under data/")
        chunker_name = c2.selectbox("Chunking method", chunk_ids, index=chunk_ids.index("fixed_256_o64"), key="ask_chunker")
        c2.caption(CHUNKERS[chunker_name]["info"])
        emb_name = c3.selectbox("Embedding model", emb_ids, key="ask_emb")
        c3.caption(EMBEDDERS[emb_name]["info"])
        mode = c4.radio("Retrieval", ["dense", "hybrid"], horizontal=True, key="ask_mode")
        c4.caption("dense = meaning only. hybrid = meaning plus BM25 keywords.")
        eff_emb = emb_name if (HAS_ST or EMBEDDERS[emb_name]["kind"] == "hash") else "hashing-bow"
        if eff_emb != emb_name:
            st.warning(f"sentence-transformers is not installed, so {emb_name} cannot run. Using the offline hashing embedder instead. "
                       "Install it with: pip install sentence-transformers")
        idx = build_index(corpus, chunker_name, eff_emb) if corpora else None
        if idx is not None:
            sp = idx["spans"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Documents", len({x[0] for x in sp}))
            m2.metric("Chunks indexed", f"{len(sp):,}")
            m3.metric("Average words per chunk", f"{np.mean([len(x[3].split()) for x in sp]):.0f}")
            m4.metric("Vector size", idx["X"].shape[1])
        rb1, rb2 = st.columns([5, 1.4])
        who = provider if provider == OFFLINE else f"{provider}: {model_id or 'no model chosen'}"
        rb1.markdown(f'<span class="pill">{chunker_name}</span><span class="pill">{eff_emb}</span><span class="pill">{mode}</span>'
                     f'<span class="pill muted">Answering with: {_html.escape(who)}</span>', unsafe_allow_html=True)
        if rb2.button("Rebuild index", help="Use after adding or editing files in data/", width="stretch"):
            build_index.clear(); st.rerun()

    if idx is None:
        st.info("No documents found. Create a folder data/<corpus>/ and add .md, .txt, .html or .pdf files.")
    else:
        st.session_state.setdefault("ask_history", [])
        hist = st.session_state["ask_history"]

        # ---- 2. question
        with st.container(border=True):
            st.markdown("#### 2. Ask a question")
            with st.form("ask_form", clear_on_submit=True, border=False):
                q = st.text_area("Your question", height=90, label_visibility="collapsed",
                                 placeholder="Ask something your documents answer, for example: What does the policy say about data retention?")
                f1, f2 = st.columns([1, 6])
                submitted = f1.form_submit_button("Ask", type="primary", width="stretch")
            if submitted and q.strip():
                with st.spinner("Retrieving passages and writing the answer..."):
                    units = retrieve(idx, q, mode)
                    t0 = time.time(); err = False
                    try:
                        ans = call_llm(provider, api_key, model_id, build_prompt(q, units), units, q, base_url)
                    except Exception as e:
                        ans, err = f"The model call failed: {e}", True
                    ms = (time.time() - t0) * 1000
                hist.append({"q": q, "a": ans.strip() or "(empty answer)", "units": units, "ms": ms, "err": err,
                             "model": who, "mode": mode, "chunker": chunker_name, "embedder": eff_emb})

        # ---- 3. answers (newest first)
        st.markdown("#### 3. Answers and sources")
        if not hist:
            callout("Answers cite their sources as [1], [2]. Open a source to read the passage the answer came from; "
                    "the words from your question are highlighted so you can check it quickly.")
        else:
            if st.button("Clear history"):
                st.session_state["ask_history"] = []; st.rerun()
            for t in reversed(hist):
                with st.container(border=True):
                    st.markdown(f'<div class="q">{_html.escape(t["q"])}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="lbl">Answer</div>', unsafe_allow_html=True)
                    (st.error if t["err"] else st.markdown)(t["a"])
                    st.markdown(f'<span class="pill">{t["chunker"]}</span><span class="pill">{t["embedder"]}</span>'
                                f'<span class="pill">{t["mode"]}</span><span class="pill muted">{_html.escape(t["model"])}</span>'
                                f'<span class="pill muted">{t["ms"]:.0f} ms</span>', unsafe_allow_html=True)
                    st.markdown(f'<div class="lbl">Sources ({len(t["units"])})</div>', unsafe_allow_html=True)
                    show_sources(t["units"], t["q"])