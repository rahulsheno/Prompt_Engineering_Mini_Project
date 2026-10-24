# RAG Chunking and Embedding Study

A study of how chunking method, embedding model and retrieval mode change the quality of retrieval-augmented generation (RAG), with a Streamlit app to explore the results and to ask questions of your own documents.

The evaluation follows the measures in **ITU-T F.748.52 (03/2025)**, *Requirements and evaluation methods for retrieval augmented generation of large scale pre-trained models*. The study is **aligned with** the standard, not conformant.

---

## Team

| Name | SRN |
|---|---|
| Vinayak Kalal | PES2UG25AM809 |
| Samvit Nandakishore | PES2UG24AM144 |
| Sabbir Mondal | PES2UG24AM141 |
| Rahul Shenoy | PES2UG24AM129 |

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
| `.gitignore` | Files and folders Git should not track |

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

## References

1. P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Proc. NeurIPS, 2020.
2. V. Karpukhin et al., "Dense passage retrieval for open-domain question answering," in Proc. EMNLP, 2020.
3. Y. Gao et al., "Retrieval-augmented generation for large language models: A survey," arXiv:2312.10997, 2023.
4. S. Barnett et al., "Seven failure points when engineering a retrieval augmented generation system," in Proc. IEEE/ACM CAIN, 2024, arXiv:2401.05856.
5. N. F. Liu et al., "Lost in the middle: How language models use long contexts," Trans. Assoc. Comput. Linguist., vol. 12, 2024.
6. F. Cuconasu et al., "The power of noise: Redefining retrieval for RAG systems," in Proc. ACM SIGIR, 2024.
7. R. Qu, R. Tu, and F. Bao, "Is semantic chunking worth the computational cost?" in Findings of NAACL, 2025, arXiv:2410.13070.
8. T. Chen et al., "Dense X retrieval: What retrieval granularity should we use?" in Proc. EMNLP, 2024, arXiv:2312.06648.
9. A. J. Yepes et al., "Financial report chunking for effective retrieval augmented generation," arXiv:2402.05131, 2024.
10. P. Finardi et al., "The chronicles of RAG: The retriever, the chunk and the generator," arXiv:2401.07883, 2024.
11. M. Gunther et al., "Late chunking: Contextual chunk embeddings using long-context embedding models," arXiv:2409.04701, 2024.
12. Anthropic, "Introducing Contextual Retrieval," Anthropic blog, Sep. 2024.
13. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in Proc. EMNLP-IJCNLP, 2019.
14. L. Wang et al., "Text embeddings by weakly-supervised contrastive pretraining," arXiv:2212.03533, 2022.
15. S. Xiao et al., "C-Pack: Packed resources for general Chinese embeddings," in Proc. ACM SIGIR, 2024, arXiv:2309.07597.
16. Z. Li et al., "Towards general text embeddings with multi-stage contrastive learning," arXiv:2308.03281, 2023.
17. Z. Nussbaum et al., "Nomic Embed: Training a reproducible long context text embedder," arXiv:2402.01613, 2024.
18. N. Muennighoff et al., "MTEB: Massive text embedding benchmark," in Proc. EACL, 2023.
19. N. Thakur et al., "BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models," in Proc. NeurIPS Datasets and Benchmarks, 2021.
20. A. Kusupati et al., "Matryoshka representation learning," in Proc. NeurIPS, 2022.
21. A. Shakir, T. Aarsen, and S. Lee, "Binary and scalar embedding quantization for significantly faster and cheaper retrieval," Hugging Face blog, Mar. 2024.
22. S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," Found. Trends Inf. Retr., vol. 3, no. 4, 2009.
23. G. V. Cormack, C. L. A. Clarke, and S. Buettcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," in Proc. ACM SIGIR, 2009.
24. R. Nogueira and K. Cho, "Passage re-ranking with BERT," arXiv:1901.04085, 2019.
25. X. Wang et al., "Searching for best practices in retrieval-augmented generation," in Proc. EMNLP, 2024.
26. J. Johnson, M. Douze, and H. Jegou, "Billion-scale similarity search with GPUs," IEEE Trans. Big Data, vol. 7, no. 3, 2021.
27. S. Es et al., "RAGAs: Automated evaluation of retrieval augmented generation," in Proc. EACL (Demos), 2024.
28. L. Zheng et al., "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena," in Proc. NeurIPS Datasets and Benchmarks, 2023.
29. ITU-T Recommendation F.748.52 (03/2025). Requirements and evaluation methods for retrieval augmented generation of large scale pre-trained models. https://handle.itu.int/11.1002/1000/16242
30. Python Software Foundation, "The Python Standard Library," Python 3 documentation, docs.python.org/3/library/, accessed 2026.

---


