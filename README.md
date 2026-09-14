<div align="center">

# UAJY Academic Document RAG Chatbot

**Accurate, Hallucination-Free Academic Question Answering with Streamlit, FAISS, and Google Gemini**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit 1.49](https://img.shields.io/badge/Streamlit-1.49-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://aistudio.google.com/)
[![Gemini Embedding](https://img.shields.io/badge/Embedding-gemini--embedding--001-blue?style=flat-square)](https://ai.google.dev/)
[![Hybrid Retrieval](https://img.shields.io/badge/Retrieval-Hybrid%20BM25%20%2B%20Dense%20(RRF)-8E44AD?style=flat-square)](#-hybrid-retrieval-pipeline)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS%20CPU-26A69A?style=flat-square)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

[Bahasa Indonesia](#-ringkasan-proyek) · [English](#-overview)

<br/>

![UAJY Academic RAG Chatbot](assets/banner.jpg)

</div>

---

## 📌 Overview

**UAJY Academic Document RAG Chatbot** is a production-grade **Retrieval-Augmented Generation (RAG)** assistant engineered to provide verified, hallucination-free answers to students and faculty members based on the official academic handbook (*Buku Pedoman Akademik Fakultas Teknologi Industri Universitas Atma Jaya Yogyakarta 2025/2026*).

Unlike generic language models that are prone to hallucinating administrative deadlines or degree requirements, this system separates **knowledge retrieval** (lightweight local FAISS similarity search) from **response generation** (Google Gemini 2.5 Flash via API), enforcing strict ground-truth citation and explicit refusal for out-of-scope inquiries.

---

## ✨ Key Features

- 🔀 **Hybrid Retrieval:** Dense embeddings capture meaning while BM25 Okapi catches exact terms that academic documents live on — `IPK 3,51`, `144 SKS`, `Pasal 12`. Both rankings are merged with Reciprocal Rank Fusion.
- 🎚️ **LLM Reranker:** A listwise reranker scores every candidate on whether it can actually *answer* the question, not merely resemble it. Measured impact: **MRR 0.858 → 1.000**.
- 💬 **History-Aware Query Rewriting:** Follow-ups like *"berapa maksimalnya?"* are rewritten into standalone questions **before** retrieval runs, so the right context is fetched in the first place.
- 🎯 **Strict Document Grounding:** Answers are generated **exclusively** from retrieved PDF context chunks.
- 📑 **Precise Source Citations:** Chunks are built with per-line page tracking, so **76% of chunks cite a single page** (average 1.29 pages per chunk).
- 🛡️ **Layered Anti-Hallucination Guardrails:** Similarity floor → IDF-weighted lexical coverage → reranker gate → strict system prompt. Out-of-scope refusal is **100%** in the full configuration.
- 🧹 **Extraction Noise Filtering:** Rotated org-chart diagrams and broken font encodings produce garbage text that would otherwise compete for top-k slots. 24% of raw lines are filtered at ingestion.
- 🔍 **Interactive Document Explorer:** Search, filter, and inspect all 219 indexed chunks with their hierarchical section paths.
- 📊 **Ablation Study, Not Just a Benchmark:** Compare dense-only vs hybrid vs hybrid+rerank on page-level relevance, MRR, refusal accuracy, and false-refusal rate.
- 🌙 **Editorial Dark Theme UI:** Modern, clean, high-contrast dark interface built with Streamlit and tailored CSS design tokens.

---

## 🏗️ System Architecture

```
                  ┌─────────────────────────────────────┐
                  │   Official Campus PDF Document      │
                  │ (Buku Pedoman Akademik FTI UAJY)   │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼ (Offline / One-time Ingestion)
                  ┌─────────────────────────────────────┐
                  │          Ingestion Pipeline         │
                  │  • PDF text extraction (pdfplumber) │
                  │  • Noise filtering (24% of lines)   │
                  │  • Page-accurate chunking + heading │
                  │  • Gemini Embedding (Dim: 3072)     │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │  faiss.index · metadata.json · index_info    │
              └──────────────────────┬───────────────────────┘
                                     │
 ────────────────────────────────────┼────────────────────────────────────
                                     │ (Online / Runtime Query Flow)
                                     ▼
                         ┌───────────────────────┐
                         │ User Query (Streamlit)│
                         └───────────┬───────────┘
                                     ▼
                         ┌───────────────────────┐
                         │ Query Rewriting       │  "berapa maksimalnya?"
                         │ (history-aware)       │   → standalone question
                         └───────────┬───────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
      ┌───────────────────────┐            ┌───────────────────────┐
      │ Dense Retrieval       │            │ Lexical Retrieval     │
      │ FAISS cosine, top-20  │            │ BM25 Okapi, top-20    │
      └───────────┬───────────┘            └───────────┬───────────┘
                  └──────────────────┬──────────────────┘
                                     ▼
                         ┌───────────────────────┐
                         │ Reciprocal Rank Fusion│
                         │ (rank-based, no norm) │
                         └───────────┬───────────┘
                                     ▼
                         ┌───────────────────────┐
                         │ LLM Reranker          │  scores 0–10 on
                         │ (listwise, 1 call)    │  answerability
                         └───────────┬───────────┘
                                     ▼
                         ┌───────────────────────┐
                         │ Relevance Gate        │──► Polite refusal
                         │ (layered defense)     │    if nothing relevant
                         └───────────┬───────────┘
                                     ▼
                         ┌───────────────────────┐
                         │ Prompt Builder        │
                         │ + Gemini 2.5 Flash    │
                         └───────────┬───────────┘
                                     ▼
                         ┌───────────────────────┐
                         │ Grounded Answer +     │
                         │ Page & Section Source │
                         └───────────────────────┘
```

---

## 🔀 Hybrid Retrieval Pipeline

**Why two retrieval paths?** Dense embeddings understand meaning but blur exact tokens. Ask *"berapa IPK minimum untuk cum laude?"* and a dense-only index happily returns chunks about graduation in general. BM25 anchors on the literal terms that matter. Neither alone is enough.

**Why fuse by rank instead of score?** Cosine similarity lives in `0–1` while BM25 is unbounded. Normalising them against each other requires arbitrary scaling. [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) sidesteps this entirely by summing `weight / (k + rank)` — only ordering matters.

**Why an LLM reranker instead of a local cross-encoder?** Multilingual cross-encoders weigh hundreds of megabytes and drag in PyTorch, which conflicts with keeping this project CPU-only and deployable on free tiers. One `gemini-3.5-flash-lite` call scores all candidates in ~400 ms.

**How the relevance gate actually works.** An honest note, because measurement contradicted the original design: since embeddings are built with `RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY` task types, cosine scores compress into a narrow high band and **overlap** between relevant and irrelevant questions:

```
in-scope     : 0.762 – 0.866
out-of-scope : 0.729 – 0.766     ← overlaps
```

No single similarity threshold can separate them. The inherited `0.30` threshold passed **every** out-of-scope question. So the threshold was demoted to a sanity floor, and the reranker became the real precision gate — it scores out-of-scope candidates `0–1` and refuses 5/5. Turning the reranker off drops retrieval-level refusal to 0%, which the UI warns about explicitly.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend UI** | [Streamlit](https://streamlit.io/) 1.32+ | Chat interface, document explorer, retrieval debug panel |
| **LLM Generation** | [Google Gemini 2.5 Flash](https://ai.google.dev/) | High-speed, context-rich reasoning in Indonesian & English |
| **Rerank & Rewrite** | [Gemini 3.5 Flash Lite](https://ai.google.dev/) | Listwise reranking and query rewriting (zero thinking tokens) |
| **Embeddings** | [gemini-embedding-001](https://ai.google.dev/) | 3072-dim multilingual vectors, task-type aware |
| **Dense Search** | [FAISS (CPU)](https://github.com/facebookresearch/faiss) | `IndexFlatIP` cosine similarity over normalised vectors |
| **Lexical Search** | BM25 Okapi (in-house) | Exact-term matching, built in-memory at startup — no extra artifact, no dependency |
| **Fusion** | Reciprocal Rank Fusion | Rank-based merge of dense + lexical results |
| **PDF Extraction** | [pdfplumber](https://github.com/jsvine/pdfplumber) | Layout-aware text extraction from PDF |
| **Language** | Python 3.11+ | Core application runtime (`tomllib` for secrets parsing) |

---

## 📁 Repository Structure

```
ChatBot RAG kampus/
├── .streamlit/
│   ├── config.toml           # Streamlit dark theme configuration
│   └── secrets.toml          # Secret API keys (excluded from git)
├── app/
│   ├── __init__.py
│   ├── config.py             # Central config: models + RetrievalConfig dataclass
│   ├── cli_utils.py          # UTF-8 stdout for Windows terminals
│   ├── text_utils.py         # Indonesian tokenizer, stopwords, domain aliases
│   ├── lexical_index.py      # BM25 Okapi + IDF-weighted lexical coverage
│   ├── llm_client.py         # Gemini wrapper (Streamlit-independent) + embed cache
│   ├── query_rewriter.py     # History-aware follow-up rewriting
│   ├── reranker.py           # Listwise LLM reranker
│   ├── retrieval.py          # Hybrid retrieval, RRF fusion, relevance gate
│   ├── prompt_builder.py     # Prompt template with strict guardrails
│   └── main_streamlit.py     # Streamlit multi-page UI application
├── assets/
│   └── banner.jpg            # README hero banner image
├── data/
│   ├── README.md
│   └── Buku-Pedoman-Akademik-Fakultas-Teknologi-Industri-2025-2026.pdf
├── eval/
│   ├── __init__.py
│   ├── test_questions.json   # 22 questions w/ verified page-level ground truth
│   ├── run_eval.py           # Ablation study runner
│   └── eval_results.json     # Generated metrics + per-question detail
├── index/
│   ├── README.md
│   ├── faiss.index           # Persisted FAISS index (219 vectors, 3072-dim)
│   ├── metadata.json         # Chunk text, page range, heading path
│   └── index_info.json       # Build provenance: PDF hash, model, task type
├── ingestion/
│   ├── __init__.py
│   ├── extract_text.py       # PDF parsing + extraction noise filtering
│   ├── chunking.py           # Page-accurate chunker with heading hierarchy
│   └── build_index.py        # Pipeline orchestrator (supports --dry-run)
├── .gitignore
├── PRD_RAG_Chatbot_Dokumen_Kampus.md
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone Repository & Install Dependencies

```bash
git clone https://github.com/BenyRonald77/uajy-academic-rag-chatbot.git
cd uajy-academic-rag-chatbot

pip install -r requirements.txt
```

### 2. Configure Gemini API Key

Obtain a free API key from [Google AI Studio](https://aistudio.google.com/), then create `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_actual_gemini_api_key_here"
```

### 3. Build the Vector Index (One-Time Execution)

Preview chunking first — this costs nothing and calls no API:

```bash
python ingestion/build_index.py --dry-run
```

Then build for real. The `--yes` flag is required because building overwrites the existing index and consumes embedding quota:

```bash
python ingestion/build_index.py --yes
```

### 4. Run the Streamlit Web Application

```bash
python -m streamlit run app/main_streamlit.py
```

Open your browser at **`http://localhost:8501`**.

---

## 📊 Evaluation: Ablation Study

```bash
python eval/run_eval.py                      # all three configurations
python eval/run_eval.py --mode hybrid_rerank # one configuration only
```

### Results

Measured on 22 questions — 17 in-scope with verified page-level ground truth, 5 out-of-scope, 2 conversational follow-ups. `top_k = 4`.

| Metric | Dense only | Hybrid (RRF) | **Hybrid + Rerank** |
|---|---|---|---|
| **Page Recall@4** | 100.0% | 94.1% | **100.0%** |
| **MRR** | 0.858 | 0.912 | **1.000** |
| **Keyword coverage** (all terms required) | 94.1% | 88.2% | **100.0%** |
| **Refusal accuracy** (out-of-scope) | 0.0% | 0.0% | **100.0%** |
| **False refusal rate** ↓ | 0.0% | 0.0% | **0.0%** |
| **Conversational recall** | 100.0% | 100.0% | **100.0%** |
| **Avg latency** ↓ | 707 ms | 638 ms | 2029 ms |

**MRR 1.000 means the correct page ranked first for every single in-scope question.** That is what the reranker buys: hybrid fusion alone widens the candidate pool but does not order it perfectly, and it even displaced one answer (Q2, the `144 SKS` question, dropping recall to 94.1%). The reranker recovered it and put it at rank 1.

The refusal column is the starkest result. Without the reranker, refusal is **0%** — see [the gate explanation above](#-hybrid-retrieval-pipeline) for why similarity thresholds cannot do this job on task-type embeddings. The extra ~1.4 s of latency is the price of that guarantee.

### Why these numbers differ from a typical RAG benchmark

The earlier version of this suite scored a query as correct when **any** expected keyword appeared anywhere in the retrieved text:

```python
keywords_found = any(kw in all_text for kw in expected)   # too lenient to ever fail
```

Under that rule, *"Apa sanksi bagi mahasiswa yang melakukan plagiarisme?"* counted as a success purely because the word "sanksi" appeared somewhere — even though the word "plagiat" **does not exist anywhere in the source PDF**, making the question unanswerable. The reported 100% reflected the looseness of the metric, not the quality of the system.

This suite is deliberately harder to satisfy:

- **Page Recall** checks whether a retrieved chunk comes from a page manually verified to contain the answer. Page-level relevance is objective and cannot be gamed by common words.
- **MRR** distinguishes ranking the answer 1st from ranking it 4th.
- **Keyword coverage** requires *every* expected term, not one.
- **False refusal rate** is tracked so a system cannot inflate its refusal score by rejecting everything.
- **Reranker failures are counted and reported.** If the reranker fails, the refusal figures for that run are explicitly flagged as invalid rather than silently reported as passing.

### Chunking quality

Page-precision is a prerequisite for trustworthy citations, so it is measured too:

| Metric | Before | After |
|---|---|---|
| Max pages claimed by one chunk | 13 | **4** |
| Avg pages per chunk | 1.99 | **1.29** |
| Chunks citing a single page | — | **76%** (166/219) |
| Chunks carrying a section heading | — | **100%** |

---

## 🇮🇩 Ringkasan Proyek (Bahasa Indonesia)

**RAG Chatbot Dokumen Kampus UAJY** adalah asisten tanya jawab berbasis *Retrieval-Augmented Generation* (RAG) yang dirancang untuk menjawab pertanyaan seputar buku pedoman akademik Fakultas Teknologi Industri Universitas Atma Jaya Yogyakarta (FTI UAJY) Tahun Akademik 2025/2026.

### Mengapa Menggunakan RAG?
- **Bebas Halusinasi:** Model LLM diinstruksikan menjawab **hanya** berdasarkan teks yang ditemukan di dalam dokumen PDF resmi.
- **Kutipan Transparan:** Setiap jawaban menyertakan rujukan nomor halaman dan judul bab/pasal.
- **Ringan & Hemat Resource:** Vector store dijalankan secara lokal via FAISS (CPU), sedangkan inferensi dilakukan via API Google Gemini.

### Pencarian Hibrida
Pencarian dilakukan lewat dua jalur sekaligus. Jalur *dense* memakai embedding untuk menangkap kesamaan makna, sedangkan **BM25** menangkap istilah eksak yang justru paling sering ditanyakan mahasiswa: `IPK 3,51`, `144 SKS`, `Pasal 12`. Hasil keduanya digabung dengan *Reciprocal Rank Fusion*, lalu **reranker** menilai ulang kandidat berdasarkan kemampuannya benar-benar menjawab pertanyaan — bukan sekadar mirip kata.

Pertanyaan lanjutan seperti *"berapa maksimalnya?"* ditulis ulang lebih dulu menjadi pertanyaan mandiri sebelum pencarian berjalan, sebab tanpa itu konteks yang salah sudah terambil sejak awal.

### Hasil Pengukuran
Dari 22 pertanyaan uji, konfigurasi lengkap mencapai **MRR 1,000** — halaman yang memuat jawaban selalu berada di peringkat pertama — dengan **penolakan 100%** untuk pertanyaan di luar cakupan dan **tanpa satu pun penolakan salah**. Rinciannya ada di [bagian evaluasi](#-evaluation-ablation-study).

### Panel Debug
Setiap jawaban dilengkapi rincian retrieval yang bisa dibuka: skor dense, BM25, RRF, dan reranker per kandidat, jalur mana yang menemukannya, keputusan gate relevansi, serta latensi tiap tahap. Tujuannya agar proses retrieval bisa diaudit, bukan menjadi kotak hitam.

---

## ⚖️ License & Disclaimer

- **License:** Distributed under the [MIT License](LICENSE).
- **Disclaimer:** Chatbot ini adalah media edukasi dan asisten pencarian informasi akademik. Informasi resmi tetap mengacu pada keputusan Dekanat dan Kantor Administrasi Akademik Universitas Atma Jaya Yogyakarta.

---

## Streamlit

- https://uajyachatbot.streamlit.app/

<div align="center">
  <sub>Developed with ❤️ by <a href="https://github.com/BenyRonald77">Beny Ronald</a> for Universitas Atma Jaya Yogyakarta</sub>
</div>
