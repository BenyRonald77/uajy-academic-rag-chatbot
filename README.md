<div align="center">

# UAJY Academic Document RAG Chatbot

**Accurate, Hallucination-Free Academic Question Answering with Streamlit, FAISS, and Google Gemini**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit 1.49](https://img.shields.io/badge/Streamlit-1.49-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini%203.6%20Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://aistudio.google.com/)
[![Gemini Embedding](https://img.shields.io/badge/Embedding-gemini--embedding--001-blue?style=flat-square)](https://ai.google.dev/)
[![Hybrid Retrieval](https://img.shields.io/badge/Retrieval-Hybrid%20BM25%20%2B%20Dense%20(RRF)-8E44AD?style=flat-square)](#-hybrid-retrieval-pipeline)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS%20CPU-26A69A?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Tests](https://img.shields.io/badge/tests-404%20passing-2ea44f?style=flat-square)](#-testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

[Bahasa Indonesia](#-ringkasan-proyek) · [English](#-overview)

<br/>

![UAJY Academic RAG Chatbot](assets/banner.jpg)

</div>

---

## 📌 Overview

**UAJY Academic Document RAG Chatbot** is a production-grade **Retrieval-Augmented Generation (RAG)** assistant engineered to provide verified, hallucination-free answers to students and faculty members based on the official academic handbook (*Buku Pedoman Akademik Fakultas Teknologi Industri Universitas Atma Jaya Yogyakarta 2025/2026*).

Unlike generic language models that are prone to hallucinating administrative deadlines or degree requirements, this system separates **knowledge retrieval** (lightweight local FAISS similarity search) from **response generation** (Google Gemini via API), enforcing strict ground-truth citation and explicit refusal for out-of-scope inquiries.

---

## ✨ Key Features

- 🔀 **Hybrid Retrieval:** Dense embeddings capture meaning while BM25 Okapi catches exact terms that academic documents live on — `IPK 3,51`, `144 SKS`, `Pasal 12`. Both rankings are merged with Reciprocal Rank Fusion.
- 🎚️ **LLM Reranker:** A listwise reranker scores every candidate on whether it can actually *answer* the question, not merely resemble it. Measured impact: **MRR 0.858 → 1.000**.
- 💬 **History-Aware Query Rewriting:** Follow-ups like *"berapa maksimalnya?"* are rewritten into standalone questions **before** retrieval runs, so the right context is fetched in the first place.
- 🎯 **Strict Document Grounding:** Answers are generated **exclusively** from retrieved PDF context chunks.
- 📑 **Precise Source Citations:** Chunks are built with per-line page tracking, so **76% of chunks cite a single page** (average 1.29 pages per chunk).
- 🛡️ **Layered Anti-Hallucination Guardrails:** Similarity floor → IDF-weighted lexical coverage → reranker gate → strict system prompt. Out-of-scope refusal is **100%** in the full configuration.
- 🧹 **Extraction Noise Filtering:** Rotated org-chart diagrams and broken font encodings produce garbage text that would otherwise compete for top-k slots. 24% of raw lines are filtered at ingestion.
- 📚 **Multi-Document Index:** One index spans many campus documents. Every chunk records its source, because "page 48" means something different in the academic handbook than in a rector's decree. Per-document search filter included.
- 🕒 **Stale Index Detection:** Each document's SHA-256 *and* an ingestion pipeline version are recorded at build time, then re-checked on startup. A source PDF that changed, or chunking logic that improved, both surface as a warning instead of quietly answering from a stale index.
- 🔍 **Interactive Document Explorer:** Search, filter, and inspect all 218 indexed chunks with their hierarchical section paths.
- 📊 **Ablation Study, Not Just a Benchmark:** Compare dense-only vs hybrid vs hybrid+rerank on page-level relevance, MRR, refusal accuracy, and false-refusal rate.
- 🧪 **404 tests, No API Key Required:** Every pure-logic module is covered, including regression guards for six real bugs found by measurement. CI runs on every push.
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
                         │ + Gemini 3.6 Flash    │
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
| **LLM Generation** | [Google Gemini 3.6 Flash](https://ai.google.dev/) | High-speed, context-rich reasoning in Indonesian & English |
| **Rerank & Rewrite** | [Gemini 3.5 Flash Lite](https://ai.google.dev/) | Listwise reranking and query rewriting (zero thinking tokens) |
| **Model Resilience** | Configurable fallback chain | Retired models degrade to the next available one instead of erroring |
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
├── .github/workflows/
│   └── tests.yml             # CI: pytest on Python 3.11 & 3.12, no API key needed
├── app/
│   ├── __init__.py
│   ├── config.py             # Central config: models + RetrievalConfig dataclass
│   ├── cli_utils.py          # UTF-8 stdout for Windows terminals
│   ├── text_utils.py         # Indonesian tokenizer, stopwords, domain aliases
│   ├── lexical_index.py      # BM25 Okapi + IDF-weighted lexical coverage
│   ├── index_status.py       # Stale-index detection via document hashes
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
│   ├── run_eval.py           # Retrieval ablation study runner
│   ├── answer_eval.py        # Groundedness, citation, and refusal checks
│   ├── run_answer_eval.py    # Answer-level evaluation runner
│   └── eval_results.json     # Generated metrics + per-question detail
├── tests/                    # 404 tests, no API key required
│   ├── conftest.py
│   ├── test_regressions.py   # Guards for six real bugs
│   └── test_*.py             # Per-module unit tests
├── index/
│   ├── README.md
│   ├── faiss.index           # Persisted FAISS index (218 vectors, 3072-dim)
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
python ingestion/build_index.py --yes                     # every PDF in data/
python ingestion/build_index.py --pdf data/a.pdf --yes    # specific documents
```

Drop additional PDFs into `data/` and rebuild — one index holds them all, and each chunk remembers which document it came from.

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
| **Page Recall@4** | 100.0% | 100.0% | **100.0%** |
| **MRR** | 0.858 | 0.868 | **1.000** |
| **Keyword coverage** (all terms required) | 88.2% | 88.2% | **94.1%** |
| **Refusal accuracy** (out-of-scope) | 0.0% | 0.0% | **100.0%** |
| **False refusal rate** ↓ | 0.0% | 0.0% | **0.0%** |
| **Conversational recall** | 100.0% | 100.0% | **100.0%** |
| **Avg latency** ↓ | 602 ms | 487 ms | 1746 ms |

Zero reranker failures in this run, so the refusal figures are valid. The single keyword miss is not a retrieval failure: page 48 was retrieved at rank 1, but the phrase "cum laude" — which appears exactly once in the entire document — landed in the adjacent chunk. Page-level relevance and MRR both capture this correctly as a hit.

> **Free-tier quota is per model, not per account:** 20 requests/day/model. Use `--delay 2` so the reranker is not rate-limited mid-run — it is the stage that acts as the refusal gate. The suite counts and reports reranker failures rather than letting them quietly deflate the refusal figure.

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
| Chunks citing a single page | — | **76%** (165/218) |
| Chunks carrying a section heading | — | **100%** |
| Chunks with a meaningless section label | 31% | **0%** |

That last row was found by measurement, not by reading code. The rule "an all-caps line is a heading" also caught sentence fragments: `... Universitas Atma Jaya Yogyakarta (UAJY).` left behind a line reading `UAJY).`, which became the root heading for 59 chunks. Across three such fragments, **68 of 222 chunks (31%)** displayed a meaningless section label in their citation.

Tightening the rule removed those, but revealed the deeper issue: this document marks its sections with capitalisation rather than "BAB", so the all-caps heuristic is load-bearing — and it also catches table column labels. One such label became the root heading for 94 chunks. Rather than tuning heuristics against a single document, citations now display the **deepest** heading instead of the full path:

| Citation label | Questionable |
|---|---|
| Full heading path (its root) | 97/218 (**44%**) |
| Deepest heading only | 18/218 (**8%**) |

The deepest heading is also the more useful one — `H. Cuti Studi` rather than `PROGRAM › ...`. The full path is still embedded with each chunk, where the extra context helps and the noise is diluted, and remains visible in the debug panel for auditing.

---

## 🔬 Answer-Level Evaluation

Retrieval metrics stop halfway. The two largest claims of this project live on the generation side, so they get their own suite:

```bash
python eval/run_answer_eval.py --delay 3          # full run
python eval/run_answer_eval.py --limit 6          # sample, saves quota
python eval/run_answer_eval.py --no-judge         # citations + refusal only
```

| Metric | What it measures | How |
|---|---|---|
| **Groundedness** | Is every factual claim supported by the retrieved context? | LLM-as-judge, 0–10, pass ≥ 7 |
| **Citation validity** | Do the page numbers *written in the answer* come from the context? | Deterministic parser |
| **Citation precision** | What share of cited pages are legitimate? | Deterministic parser |
| **Refusal (end-to-end)** | Does the final answer actually refuse, not just the retrieval gate? | Refusal markers from the system prompt |
| **False refusal** | Are answerable questions being refused? | Same, inverted |

Everything except groundedness is rule-based and unit-tested, so only the one metric that genuinely needs language understanding depends on a model.

The suite is also **honest about its own failures**: API errors are never allowed to pass as answers, and judge failures are counted separately rather than averaged in. This matters more than it sounds — see below.

### Results

| Metric | Result |
|---|---|
| **Groundedness** | **100.0%** (16/16 judged) · avg **10.00/10** |
| **Citation precision** | **100.0%** — no fabricated page numbers |
| **Citation validity** | 93.8% (15/16) |
| **Cites the verified answer page** | 93.8% |
| **Refusal, end-to-end** | **100.0%** (5/5 out-of-scope) |
| **False refusal** ↓ | **0.0%** |

Groundedness at a perfect average is the evidence behind the hallucination-free claim: every factual statement in every answer traced back to the retrieved context.

Two entries fell out of the run, both accounted for rather than hidden. One question hit the 20-requests/day quota ceiling and was excluded from all metrics by the generation-failure mechanism. The other produced a correct, well-grounded answer that ran out of output tokens mid-sentence before it could append its source line — the cause of the one missing citation. `ANSWER_MAX_TOKENS` was raised from 2048 to 4096 in response; on Gemini 3.x, thinking tokens count against that same budget. That fix is not yet re-measured, because the daily quota ran out.

### What this suite caught about itself

Both findings were bugs in the *evaluation code*, not the chatbot — and both would have produced flattering numbers:

**API errors scored as perfect answers.** `call_llm` returns a friendly message on failure, which is right for the chat UI. The eval treated one such message as a real answer, and the groundedness judge awarded it **10/10** — an error message makes no unsupported factual claims, after all. Fixed by adding `raise_on_error=True` for eval callers and excluding generation failures from every metric.

**A section title read as a page range.** The mandated citation format is `Halaman X — [Nama Bagian]`, and section titles frequently begin with a number:

```
📄 Sumber: Halaman 43 — 5. Beban studi dan beban kredit semester
```

The extractor read `43 — 5` as a range and harvested `5` as a cited page, reporting a fabricated citation the model never made. Replaced the regex with a forward parser that stops at a dash unless it is followed by a larger number. The exact string above is now a regression test.

---

## 🧪 Testing

```bash
pip install -r requirements-dev.txt
python -m pytest
```

404 tests, ~1.5 seconds, **no API key and no index required** — every test is self-contained, so CI runs safely on forks and pull requests.

Coverage is concentrated where bugs actually appeared. Six regression guards protect real failures found by measurement, not imagined ones:

| Bug | Symptom | Guard |
|---|---|---|
| Roman-numeral regex matched Indonesian words | `di` treated as a numeral, escaping the stopword filter into the BM25 index | `test_regressions.py::TestRomanNumeralFalsePositive` |
| Overlap leaked page boundaries | One chunk claimed 13 pages despite a 3-page cap, making citations useless | `TestChunkPageSpanLeak` |
| Gibberish filter flagged course codes | `TID 1101 Kalkulus I 3 SKS` discarded as broken text, deleting curriculum tables | `TestGibberishFilterFalsePositive` |
| Valid `str` cluster judged unnatural | `administrasi`, `struktur`, `instruksi`, `herregistrasi` at risk of being filtered out | `test_extract_text.py::TestConsonantRun` |
| Rerank gate failed **open** | Refusal accuracy silently dropped 100% → 0% under rate limits, with no error | `TestRerankGateFailOpen` |
| Section title parsed as page range | Fabricated citations reported in the eval | `test_answer_eval.py` |
| Sentence fragment promoted to heading | `UAJY).` became the section label for 31% of chunks | `TestFragmentPromotedToHeading` |
| Retired model surfaced a raw 404 | Two Gemini models were removed mid-project; students saw an HTTP error instead of an answer | `TestRetiredModelFallback` |
| Index staleness invisible to code changes | Chunking logic improved, index silently stayed old — only manual measurement revealed it | `TestPipelineVersionStaleness` |

The rerank one deserves emphasis. Its fail-safe returned candidates with `rerank_score=None`, and the gate's `score is None or score >= threshold` check let every one of them through. A busy API therefore disabled the anti-hallucination gate without a single log line. The fix reports success explicitly, scores unmentioned candidates as `0`, retries transient errors, and surfaces failures in both the UI and the eval report.

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

### Banyak Dokumen dalam Satu Index
Kampus punya lebih dari satu sumber resmi: pedoman akademik, kalender akademik, SK Rektor. Cukup letakkan PDF-nya di folder `data/` lalu bangun ulang index. Setiap chunk mencatat dokumen asalnya, sebab nomor halaman hanya bermakna dalam konteks dokumennya — "halaman 48" pada pedoman akademik dan pada SK Rektor merujuk hal yang sama sekali berbeda. Pencarian juga bisa dibatasi ke dokumen tertentu lewat sidebar.

### Peringatan Index Usang
Hash setiap dokumen dicatat saat index dibangun, lalu diperiksa ulang setiap aplikasi dimuat. Kalau PDF sumbernya diperbarui, aplikasi memberi tahu bahwa index perlu dibangun ulang. Tanpa pemeriksaan ini, chatbot akan terus menjawab dari dokumen edisi lama dengan sitasi yang tampak sah — kegagalan yang paling sulit disadari pengguna.

### Pengujian
404 test berjalan tanpa API key dan tanpa index, selesai dalam sekitar 1,5 detik. Sembilan di antaranya adalah penjaga terhadap bug yang benar-benar pernah terjadi dan ditemukan lewat pengukuran, termasuk gate reranker yang dulu gagal-terbuka sehingga penolakan diam-diam jatuh dari 100% ke 0% saat API sibuk. Jalankan dengan `python -m pytest`.

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
