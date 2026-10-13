# Prompt Engineering Mini Project — RAG Chunking × Embedding × Retrieval Study

Systematic study of **chunking strategies × embedding models × retrieval modes** for
Retrieval-Augmented Generation (RAG), evaluated on Python `tech_docs` with a Streamlit
**RAG Config Advisor** app.

## Key Finding
`structure_300 + e5-small + hybrid` → **f_ctx 0.808, recall 0.913, ~26.7 ms, 0.52 MB**
(best accuracy / speed / balanced — see `outputs/decision_matrix.csv`).

## Repo Layout
```
advisor_app.py                  # Streamlit RAG Config Advisor (reads outputs/results.csv)
RAG_Chunking_Embedding_Study_fixed.ipynb  # Full experiment notebook
data/tech_docs/                 # Source corpus (8 Python stdlib docs, markdown)
eval/tech_docs_gold.csv         # 75-question gold set
outputs/                        # results.csv, decision_matrix.csv, figures
docs/                           # Report (IEEE), literature survey, presentation, guidelines
```

## Quickstart
```bash
pip install -r requirements.txt
streamlit run advisor_app.py
```

## Reproduce (notebook)
1. Open `RAG_Chunking_Embedding_Study_fixed.ipynb`
2. Set env (see `.env.example`): `LLM_PROVIDER`, `GEMINI_MODEL`
3. Run all cells — results regenerate into `outputs/` (cached embeddings go to `cache/`, git-ignored).

## Results Summary (tech_docs)
| Priority | Chunker | Embedder | Mode | f_ctx | Recall | Lat p50 (ms) | Index (MB) |
|---|---|---|---|---|---|---|---|
| accuracy | structure_300 | e5-small | hybrid | 0.808 | 0.913 | 26.7 | 0.52 |
| speed | structure_300 | e5-small | hybrid | 0.808 | 0.913 | 26.7 | 0.52 |
| low_cost | fixed_256_o0 | e5-small | hybrid | 0.804 | 0.886 | 73.8 | 0.43 |
| balanced | structure_300 | e5-small | hybrid | 0.808 | 0.913 | 26.7 | 0.52 |

See `outputs/heatmap_tech_docs.png`, `outputs/pareto_tech_docs.png`, `outputs/pipeline_tech_docs.png`.
