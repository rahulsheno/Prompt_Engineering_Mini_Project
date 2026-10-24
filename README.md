# RAG Chunking and Embedding Study

A study of how chunking method, embedding model and retrieval mode change the quality of retrieval-augmented generation (RAG), with a Streamlit app to explore the results and to ask questions of your own documents.

The evaluation follows the measures in **ITU-T F.748.52 (03/2025)**, *Requirements and evaluation methods for retrieval augmented generation of large scale pre-trained models*. The study is **aligned with** the standard, not conformant.

---

## Contents

| Path | What it is |
|---|---|
| `RAG_Chunking_Embedding_Study_fast.ipynb` | The study notebook: corpora, gold questions, all stages, plots and the F.748.52 evaluation |
| `rag_studio.py` | The Streamlit app: results, the F.748.52 tab, and a document Q&A tab |
| `requirements.txt` | Python packages for the app |
| `README.md` | This file |
| `data/<corpus>/` | Documents for each corpus. Created and filled by the notebook, or by you |
| `docs/` | Supporting documentation for the study (see the folder for the individual files) |
| `eval/<corpus>_gold.csv` | Gold questions and the evidence spans that answer them |
| `outputs/results.csv` | One row per run, written by the notebook and read by the app |
| `outputs/f748_*.csv` | F.748.52 tables written by the notebook: retrieval quality, answer quality, alignment, performance, and human vs LLM |


---

## What the study compares

**Chunking methods (9):** how documents are cut into pieces.

| Name | Method |
|---|---|
| `fixed_128_o0`, `fixed_128_o32` | Windows of 128 words, with 0 or 32 words of overlap |
| `fixed_256_o0`, `fixed_256_o64` | Windows of 256 words, with 0 or 64 words of overlap |
| `sent_4_o1`, `sent_8_o2` | Groups of 4 or 8 whole sentences, with 1 or 2 sentences of overlap |
| `semantic_p90` | Splits where the topic changes most (90th percentile of sentence-to-sentence distance) |
| `structure_300` | Follows headings and paragraphs, up to 300 words per chunk |
| `parent_child` | Searches 100-word pieces, returns the 400-word section they belong to |

**Embedding models (4 plus an offline baseline):** `bge-small`, `e5-small`, `gte-small`, `minilm`, and `hashing-bow` (keyword hashing, no download).

**Retrieval modes (3):** `dense` (meaning only), `hybrid` (dense plus BM25 keyword search, fused with reciprocal rank), and `hybrid+rerank`.

Index variants such as reduced dimensions or compression are also tested, labelled with `@dim` or a quantisation tag.

---

## Study stages

1. **Stage 1:** every chunking method with one embedding model (bge-small), dense retrieval.
2. **Stage 2:** the top chunking methods from Stage 1 with every embedding model. This gives the heatmap. The number of chunkers is set by `TOP_CHUNKERS`.
3. **Stage 3a:** the best setups under each retrieval mode.
4. **Stage 3b:** answer generation and LLM judging on a question subset (needs an LLM key). This gives the answer-quality measures.
5. **Statistics:** bootstrap confidence intervals and paired tests (question-level), plus cross-domain comparison.
6. **Plots:** heatmaps, retrieval-mode graphs and the top-configuration chart.
7. **F.748.52 evaluation:** retrieval measures, answer measures, throughput and response time, human rating sheets, and the clause-by-clause alignment.

### Metrics

- **Span-based metrics:** gold evidence is a character span `(document, start, end)`. Retrieved chunks are scored by how much of the evidence they cover. These give **recall**, **precision**, **f_ctx** (the harmonic mean of recall and precision, the main ranking score), **hit@k**, **MRR** and **density**.
- **F.748.52 retrieval (clause 8.1.2):** **P_A** (accuracy), **R_A** (recall) and **MAP@5**.
- **F.748.52 answers (clauses 8.2 and 9.2):** relevance, faithfulness and truthfulness graded 1 to 5 by an LLM judge (Annex A scales); exact match; negative rejection (**P_ERR**, **P_EAR**, **P_NRR**); information integration (**P_MIR**) on multi-document questions.
- **Performance (clauses 9.2.3 and 9.2.4):** retrieval time, queries per second with one and several threads, and end-to-end response time.

---

## Requirements

- Python 3.10 or newer
- Jupyter (to run the notebook)
- About 2 GB of disk space for the embedding models
- Internet access to download documents and models on first use
- Optional: an API key for an LLM provider (needed for Stage 3b and for the Ask tab)

Install the app packages:

```
pip install -r requirements.txt
```

`sentence-transformers` is large because it installs PyTorch. If you skip it, the app uses the offline `hashing-bow` embedder instead, and the four named embedding models are unavailable.

---

## Running the notebook

1. Open `RAG_Chunking_Embedding_Study_fast.ipynb` in Jupyter.
2. Run the cells in order. The corpus cell downloads the documents listed for each corpus into `data/`, and skips any file that already exists.
3. Check the gold-coverage line printed before Stage 1. It should say `gold coverage OK` for every corpus. If it warns that gold questions reference documents that are not loaded, those questions score 0. Fix the corpus list so it contains every document the gold set refers to.
4. Set `FAST_MODE = True` for a quick run, or `False` for the full study.
5. To run Stage 3b and the LLM judge, choose a provider and enter its key in the notebook's `choose_model()` cell. Without a key these stages are skipped or use offline defaults.
6. Results are written to `outputs/results.csv` and the F.748.52 tables to `outputs/f748_*.csv`.

**Note on stale results:** rows in `results.csv` are replaced only when their key matches. If you change the corpus, documents or gold set, delete `outputs/results.csv` first, then re-run, so that no rows from an earlier corpus remain.

---

## Running the app

```
streamlit run rag_studio.py
```

Run it from the project folder, so the app can find `data/`, `eval/` and `outputs/`. The app has three tabs.

### 1. Results and Advisor

Reads `outputs/results.csv`.

- **Heatmap:** chunking method against embedding model, for any metric. Only chunkers tested with every embedding model are shown, so each cell is a like-for-like comparison. The best cell has a gold outline.
- **Retrieval mode:** compares dense, hybrid and hybrid+rerank on quality and on time per question.
- **Advisor:** choose a priority (accuracy, balanced, speed or low storage). It ranks the tested setups and recommends one.

### 2. ITU-T F.748.52

Shows the standard's evaluation for your results, in six parts: an overview with the clause counts, retrieval quality (MAP@5, P_A, R_A), answer quality (negative rejection, information integration, Annex A grades), throughput and response time, human reviewers against the LLM judge, and the clause-by-clause alignment. Each table can be downloaded as CSV.

Sections 4 and 5 need the performance and human-rating files, which the notebook writes when you run those steps.

### 3. Ask your documents

Ask questions of your own documents.

1. Choose a corpus (a folder under `data/`), a chunking method, an embedding model and a retrieval mode.
2. Type a question and press **Ask**.
3. The answer cites numbered sources. Open a source to read the passage it came from. Words from your question are highlighted.

**Language models:** pick a provider in the sidebar and paste its API key. The key is kept only for the current browser session and is never saved to disk. Click **Load models** to list the models the key can use, or type a model id.

Supported providers: Gemini, OpenRouter, OpenAI, Groq, SiliconFlow, Together AI, DeepSeek, Mistral, Z.ai, Anthropic, and Ollama (local, no key). Without a key, **Offline demo** returns the sentence that best matches the question.

**Adding documents:** create `data/<corpus>/` and put `.md`, `.txt`, `.html` or `.pdf` files in it. Use **Rebuild index** after you add or edit files.

---

## Using your own documents

1. Create a folder `data/<your_corpus>/` and add your files.
2. For the Ask tab, the corpus appears in the dropdown.
3. For the study, add a gold question set in `eval/<your_corpus>_gold.csv`, with evidence spans that point into your documents. Without it, the notebook builds an offline draft, which should be reviewed before you rely on it.

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| A heatmap or chunking method shows `0.000` everywhere | The gold questions point to documents that are not loaded. Check the gold-coverage line and fix the corpus list. Then delete `outputs/results.csv` and re-run. |
| Stage 2 table has only three chunking methods | Stage 2 uses only the top `TOP_CHUNKERS` from Stage 1. Increase the number to test more. |
| The app shows "sentence-transformers is not installed" | Run `pip install sentence-transformers`, or keep using the offline embedder. |
| "Could not load models: ... not valid" | The key is wrong or has spaces or quotes around it. Paste it again. |
| "Cannot reach Ollama" | Start Ollama with `ollama serve`, and check the base URL in the sidebar. |
| Old values still appear after a re-run | Delete `outputs/results.csv` and re-run from the corpus cell. |
| `AttributeError: partially initialized module 'streamlit'` | A file named `streamlit.py` is in the project folder, or the app code was saved inside the Streamlit library. Rename or remove the file and reinstall Streamlit. |

---

## Limitations

- The study is a small, controlled benchmark. Some corpora have few documents, so scores vary and confidence intervals should be read alongside any ranking.
- The LLM judge can make mistakes. Use the human rating sheets to check it, and report the agreement.
- Only text corpora are supported (Markdown, text, HTML and PDF). There is no support for images, video or distributed storage.
- The study is aligned with F.748.52 but not conformant. Context relevance (Table A.4) and robustness to noisy or counterfactual documents are not implemented.
- Results depend on the LLM provider and model used for Stage 3b. Record the model, as the notebook does in `results.csv`.

---

## Reference

ITU-T Recommendation F.748.52 (03/2025). Requirements and evaluation methods for retrieval augmented generation of large scale pre-trained models. https://handle.itu.int/11.1002/1000/16242
