import numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="RAG Config Advisor", layout="wide")
st.title("RAG Config Advisor")
st.caption("Recommendations come from our chunking x embedding x retrieval experiments (outputs/results.csv).")

df = pd.read_csv("outputs/results.csv")
df = df[(df.stage != "3b")].drop_duplicates("key").dropna(subset=["f_ctx"])

corpus = st.sidebar.selectbox("Document type", sorted(df.corpus.unique()))
st.sidebar.markdown("**Priorities**")
w_acc = st.sidebar.slider("Retrieval accuracy", 0.0, 1.0, 0.6)
w_lat = st.sidebar.slider("Low latency", 0.0, 1.0, 0.2)
w_sz = st.sidebar.slider("Small index / low cost", 0.0, 1.0, 0.2)
modes = st.sidebar.multiselect("Allowed retrieval modes", sorted(df["mode"].unique()), default=sorted(df["mode"].unique()))
embs = st.sidebar.multiselect("Allowed embedders", sorted(df.embedder.unique()), default=sorted(df.embedder.unique()))

d = df[(df.corpus == corpus) & df["mode"].isin(modes) & df.embedder.isin(embs)].copy()
def norm(s, inv=False):
    s = s.astype(float); r = s.max() - s.min()
    z = (s - s.min()) / r if r > 0 else pd.Series(1.0, index=s.index)
    return 1 - z if inv else z
tot = max(w_acc + w_lat + w_sz, 1e-9)
d["score"] = (w_acc * norm(d.f_ctx) + w_lat * norm(np.log1p(d.lat_p50_ms), True) + w_sz * norm(np.log1p(d.index_mb), True)) / tot
d = d.sort_values("score", ascending=False)

if d.empty:
    st.warning("No configuration matches these filters.")
else:
    best = d.iloc[0]
    st.subheader("Recommended setup")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunking", best.chunker); c2.metric("Embedding", best.embedder)
    c3.metric("Retrieval", best["mode"]); c4.metric("Vector size", f"{int(best.dim)}-d {best.quant if isinstance(best.quant, str) and best.quant else ''}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("f_ctx", f"{best.f_ctx:.3f}"); m2.metric("Recall", f"{best.recall:.3f}")
    m3.metric("Median latency (ms)", f"{best.lat_p50_ms:.1f}"); m4.metric("Index size (MB)", f"{best.index_mb:.2f}")
    st.subheader("Top 10 alternatives")
    st.dataframe(d[["chunker", "embedder", "mode", "dim", "quant", "f_ctx", "recall", "precision", "lat_p50_ms", "index_mb", "score"]].head(10).round(3), use_container_width=True)
