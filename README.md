<div align="center">

# UAJY Academic Document RAG Chatbot

**Accurate, Grounded Academic Question Answering with Streamlit, FAISS, BM25, and an OpenAI-Compatible LLM Gateway**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit 1.49](https://img.shields.io/badge/Streamlit-1.49-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Bandel AI Gateway](https://img.shields.io/badge/LLM-OpenAI--Compatible%20Gateway-4285F4?style=flat-square)](https://bandelbanget.xyz/v1)
[![Gemini Embedding](https://img.shields.io/badge/Embedding-Gemini%20(temporary)-blue?style=flat-square)](https://ai.google.dev/)
[![Hybrid Retrieval](https://img.shields.io/badge/Retrieval-Hybrid%20BM25%20%2B%20Dense%20(RRF)-8E44AD?style=flat-square)](#-hybrid-retrieval-pipeline)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS%20CPU-26A69A?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Tests](https://img.shields.io/badge/tests-456%20passing-2ea44f?style=flat-square)](#-testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

[Bahasa Indonesia](#-ringkasan-proyek) · [English](#-overview)

<br/>

![UAJY Academic RAG Chatbot](assets/banner.jpg)

</div>

---

## 📌 Overview

**UAJY Academic Document RAG Chatbot** is a **Retrieval-Augmented Generation (RAG)** assistant designed to provide grounded answers to students and faculty members based on the official academic handbook (*Buku Pedoman Akademik Fakultas Teknologi Industri Universitas Atma Jaya Yogyakarta 2025/2026*).

Unlike generic language models that are prone to hallucinating administrative deadlines or degree requirements, this system separates **knowledge retrieval** (local FAISS + BM25) from **response generation** (an OpenAI-compatible LLM gateway), enforcing strict ground-truth citation and explicit refusal for out-of-scope inquiries. In the current short-term deployment, the gateway handles chat/rerank/rewrite while Gemini is used only for embeddings.

### Current validation status (17 September 2026)

- **Verified:** 456 offline tests pass. Bandel AI uses `deepseek-v4-flash` for chat, rewriting, and reranking, with `glm-5.3-flash` as a tested fallback. Gemini provides 3072-dimensional embeddings.
- **Also verified:** public/operator mode, rate limiting, answer guard, precise page citation parsing, and the PDF viewer. The native PDF viewer button is in commit `8ef96ac`.
- **Still pending:** rerunning retrieval and answer evaluations with the current provider setup; correcting ALL-CAPS heading noise and rebuilding the index; a browser-based end-to-end UI check; and VPS deployment.

The evaluation tables below are historical results and do not describe measured performance of the current provider setup.

---

## ✨ Key Features

- 🔀 **Hybrid Retrieval:** Dense embeddings capture meaning while BM25 Okapi catches exact terms that academic documents live on — `IPK 3,51`, `144 SKS`, `Pasal 12`. Both rankings are merged with Reciprocal Rank Fusion.
- 🎚️ **LLM Reranker:** A listwise reranker scores every candidate on whether it can actually *answer* the question, not merely resemble it. The published MRR improvement is from a historical evaluation; a new run with the current provider setup is pending.
- 💬 **History-Aware Query Rewriting:** Follow-ups like *"berapa maksimalnya?"* are rewritten into standalone questions **before** retrieval runs, so the right context is fetched in the first place.
- 🎯 **Strict Document Grounding:** Answers are generated **exclusively** from retrieved PDF context chunks.
- 📑 **Precise Source Citations:** Chunks are built with per-line page tracking, so **76% of chunks cite a single page** (average 1.29 pages per chunk).
- 🛡️ **Layered Anti-Hallucination Guardrails:** Similarity floor → IDF-weighted lexical coverage → reranker gate → strict system prompt. The published refusal rate is from a historical evaluation and needs to be remeasured.
- 🧹 **Extraction Noise Filtering:** Rotated org-chart diagrams and broken font encodings produce garbage text that would otherwise compete for top-k slots. 24% of raw lines are filtered at ingestion.
- 📚 **Multi-Document Index:** One index spans many campus documents. Every chunk records its source, because "page 48" means something different in the academic handbook than in a rector's decree. Per-document search filter included.
- 🕒 **Stale Index Detection:** Each document's SHA-256 *and* an ingestion pipeline version are recorded at build time, then re-checked on startup. A source PDF that changed, or chunking logic that improved, both surface as a warning instead of quietly answering from a stale index.
- 🔍 **Interactive Document Explorer:** Search, filter, and inspect all 218 indexed chunks with their hierarchical section paths.
- 📊 **Ablation Study, Not Just a Benchmark:** Compare dense-only vs hybrid vs hybrid+rerank on page-level relevance, MRR, refusal accuracy, and false-refusal rate.
- 🧪 **456 offline tests passed:** The suite covers pure-logic modules and regression guards for bugs found through measurement. CI runs on every push.
- 🌤️ **SIATMA-Inspired Light UI:** Light blue page background, white cards, cyan section headers, and a public student mode built with Streamlit.

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
                  │  • Gemini Embedding (temporary)   │
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
                         │ + OpenAI-Compatible LLM   │
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

**Why an LLM reranker instead of a local cross-encoder?** Multilingual cross-encoders weigh hundreds of megabytes and drag in PyTorch, which conflicts with keeping this project CPU-only and deployable on free tiers. One provider-gateway call scores all candidates.

**Current provider split (verified 17 September 2026).** Bandel AI at `https://bandelbanget.xyz/v1` handles chat completions, reranking, and query rewriting. Use the explicit `deepseek-v4-flash` model; `auto` currently returns `403 model_disabled`. The app retries model-specific disabled/out-of-stock errors with `glm-5.3-flash`. Both models passed connection and JSON-mode checks, and DeepSeek passed a grounded-answer smoke test after the prompt was updated to include copyable source lines. The endpoint does not expose `/embeddings`, so Gemini's `gemini-embedding-001` remains the embedding provider (3072 dimensions). Keep the Bandel and Google keys separate. Full retrieval and answer evaluation with this provider setup is still pending.

**How the relevance gate actually works.** An honest note, because measurement contradicted the original design: since embeddings are built with `RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY` task types, cosine scores compress into a narrow high band and **overlap** between relevant and irrelevant questions:

```
in-scope     : 0.762 – 0.866
out-of-scope : 0.729 – 0.766     ← overlaps
```

No single similarity threshold can separate them. The inherited `0.30` threshold passed every out-of-scope question in the earlier measurements, so the threshold was demoted to a sanity floor and the reranker became the precision gate. Those refusal measurements are historical; rerun the evaluation before treating them as current performance results.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend UI** | [Streamlit](https://streamlit.io/) 1.32+ | Chat interface, document explorer, retrieval debug panel |
| **LLM Generation** | OpenAI-compatible gateway (`https://bandelbanget.xyz/v1`) | Chat completions for grounded answers |
| **Rerank & Rewrite** | Same OpenAI-compatible gateway | Listwise reranking and query rewriting |
| **Embeddings (temporary)** | [gemini-embedding-001](https://ai.google.dev/) | 3072-dim multilingual vectors, task-type aware; separate Google key required |
| **Dense Search** | [FAISS (CPU)](https://github.com/facebookresearch/faiss) | `IndexFlatIP` cosine similarity over normalised vectors |
| **Lexical Search** | BM25 Okapi (in-house) | Exact-term matching, built in-memory at startup — no extra artifact, no dependency |
| **Fusion** | Reciprocal Rank Fusion | Rank-based merge of dense + lexical results |
| **PDF Extraction** | [pdfplumber](https://github.com/jsvine/pdfplumber) | Layout-aware text extraction from PDF |
| **PDF Viewer** | [PyMuPDF](https://pymupdf.readthedocs.io/) | Render the cited page as a visible PNG preview in Streamlit |
| **Language** | Python 3.11+ | Core application runtime (`tomllib` for secrets parsing) |

---

## 📁 Repository Structure

```
ChatBot RAG kampus/
├── .streamlit/
│   ├── config.toml           # Light campus-style theme configuration
│   ├── secrets.example.toml  # Provider configuration template
│   └── secrets.toml          # Local secret keys (excluded from git)
├── .github/workflows/
│   └── tests.yml             # CI: pytest on Python 3.11 & 3.12, no API key needed
├── app/
│   ├── __init__.py
│   ├── config.py             # Central config: models + RetrievalConfig dataclass
│   ├── cli_utils.py          # UTF-8 stdout for Windows terminals
│   ├── text_utils.py         # Indonesian tokenizer, stopwords, domain aliases
│   ├── lexical_index.py      # BM25 Okapi + IDF-weighted lexical coverage
│   ├── index_status.py       # Stale-index detection via document hashes
│   ├── llm_client.py         # OpenAI-compatible chat + Gemini embedding adapter
│   ├── public_config.py       # Public/operator mode and runtime settings
│   ├── rate_limit.py          # Per-session public rate limiter
│   ├── answer_guard.py        # Blocks provider responses without citations
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
├── tests/                    # 456 offline tests passed
│   ├── conftest.py
│   ├── test_openai_provider.py # OpenAI-compatible adapter tests
│   ├── test_public_mode.py     # Public mode + rate limit tests
│   └── test_*.py               # Per-module unit tests
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

### 2. Configure Providers

Chat, reranker, dan query rewriting memakai endpoint OpenAI-compatible Bandel AI. Embedding tetap memakai Google Gemini karena endpoint Bandel yang tersedia belum menyediakan `/embeddings`:

```toml
# .streamlit/secrets.toml
LLM_API_KEY = "your_bandel_api_key_here"
LLM_BASE_URL = "https://bandelbanget.xyz/v1"
LLM_MODEL = "deepseek-v4-flash"
UTILITY_MODEL = "deepseek-v4-flash"

# Key Google AI Studio, BERBEDA dari LLM_API_KEY
GEMINI_EMBEDDING_API_KEY = "your_google_embedding_key_here"

# public untuk mahasiswa; operator untuk UI internal
APP_MODE = "public"
PUBLIC_RATE_LIMIT = 20
PUBLIC_RATE_WINDOW_SECONDS = 3600
```

`GEMINI_API_KEY` lama masih dibaca sebagai fallback transisi untuk chat/embedding, tetapi konfigurasi yang direkomendasikan memakai nama terpisah agar key Bandel tidak tertukar dengan key Google. **Jangan jalankan public mode sebelum kedua key tersedia:** `LLM_API_KEY` untuk Bandel dan `GEMINI_EMBEDDING_API_KEY` untuk Google.

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

> **Historical baseline:** These figures come from the earlier Gemini-era evaluation. They are retained for reference, but the retrieval ablation has not yet been rerun with the current Bandel AI provider setup.

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

> Provider rate limits can interrupt a run. Use `--delay` as needed and inspect the reported reranker failures; a run with reranker failures cannot support valid refusal metrics.

In the historical run, **MRR 1.000 meant the correct page ranked first for every in-scope question**. Hybrid fusion alone widened the candidate pool but did not order it perfectly, and it displaced one answer (Q2, the `144 SKS` question, dropping recall to 94.1%). The reranker recovered it and put it at rank 1.

In that historical run, refusal without the reranker was **0%** — see [the gate explanation above](#-hybrid-retrieval-pipeline) for why similarity thresholds cannot do this job on task-type embeddings. The measured latency increase was about 1.4 seconds in that run.

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

### Citation heading cleanup (open)

The current citation headings still have noise from the ALL-CAPS heading heuristic. The latest project status identifies roughly **31% of citations** as affected. The heading detection fix and index rebuild have not been completed, so earlier figures in this README claiming 0% meaningless labels or 8% questionable headings should not be treated as the current result.

After correcting heading detection, rebuild the index and measure citation labels again before publishing updated cleanliness figures. Existing page-span measurements (average 1.29 pages per chunk and 76% single-page chunks) are historical and should also be rechecked after that rebuild.

---

## 👩‍🎓 Public Mode untuk Mahasiswa

Mode aplikasi default adalah `public`. Mahasiswa hanya melihat halaman Tanya Jawab, contoh pertanyaan, sumber halaman, dan tombol membersihkan percakapan. Kontrol internal — explorer, evaluasi, toggle hybrid/reranker/rewrite, slider retrieval, skor debug, jumlah chunk, dan nama model — disembunyikan.

Untuk membuka panel internal pada mesin operator, isi:

```toml
APP_MODE = "operator"
```

Public mode juga memiliki rate limit per sesi. Nilai defaultnya 20 pertanyaan per window satu jam; atur melalui `PUBLIC_RATE_LIMIT` dan `PUBLIC_RATE_WINDOW_SECONDS`. Ini limiter sederhana di proses Streamlit, bukan pengganti Redis/Nginx untuk deployment berskala besar.

### Circuit breaker provider

Jawaban publik wajib memuat sitasi halaman. Bila gateway mengabaikan prompt dan mengembalikan teks acak atau pesan error, aplikasi tidak menampilkannya sebagai jawaban akademik. Mahasiswa menerima pesan layanan yang aman, sementara operator dapat memeriksa masalah provider.

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

> **Historical baseline:** These answer-level figures predate the currently verified Bandel AI setup. Rerun the answer evaluation before using them to describe current groundedness, citation, or refusal performance.

| Metric | Result |
|---|---|
| **Groundedness** | **100.0%** (16/16 judged) · avg **10.00/10** |
| **Citation precision** | **100.0%** — no fabricated page numbers |
| **Citation validity** | 93.8% (15/16) |
| **Cites the verified answer page** | 93.8% |
| **Refusal, end-to-end** | **100.0%** (5/5 out-of-scope) |
| **False refusal** ↓ | **0.0%** |

The historical groundedness score is not evidence for the current provider setup until the evaluation is rerun.

In that historical run, one question hit the provider quota and was excluded from all metrics by the generation-failure mechanism. Another answer ran out of output tokens before it could append its source line. `ANSWER_MAX_TOKENS` was raised from 2048 to 4096 in response; that fix also needs to be remeasured in the next answer-evaluation run.

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

**456 offline tests passed** in the latest run. The suite requires no provider API key or built index, so CI can run safely on forks and pull requests.

Coverage is concentrated where bugs actually appeared. Nine regression guards protect real failures found by measurement, not imagined ones:

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
- **Jawaban Berbasis Dokumen:** Model LLM diinstruksikan menjawab **hanya** berdasarkan teks yang ditemukan di dalam dokumen PDF resmi.
- **Kutipan Transparan:** Setiap jawaban menyertakan rujukan nomor halaman dan judul bab/pasal.
- **Ringan & Hemat Resource:** Vector store dijalankan secara lokal via FAISS (CPU), embedding memakai Gemini, sedangkan jawaban/reranker/query rewriting memakai gateway Bandel AI yang OpenAI-compatible.

### Kesiapan Provider Jangka Pendek

Gateway Bandel AI sudah lolos `test_connection()` dan pengujian reranker JSON: pada query "syarat SKS lulus", chunk relevan mendapat skor 10 dan chunk tidak relevan mendapat skor 0. Gateway ini digunakan untuk chat, query rewriting, dan reranking. Embedding tetap memakai Gemini dengan dimensi 3072. Pengujian ini memverifikasi provider di level modul; uji end-to-end melalui browser masih belum dilakukan.

### Status proyek per 17 September 2026

| Selesai dan terverifikasi | Belum dikerjakan |
|---|---|
| 456 tes offline lulus; mode public/operator, rate limit, dan answer guard | Menjalankan ulang `eval/run_eval.py` dan evaluasi jawaban dengan provider saat ini |
| Provider Bandel untuk chat, rewrite, dan rerank; embedding Gemini 3072 dimensi | Memperbaiki heading ALL-CAPS dan membangun ulang index untuk membersihkan sitasi |
| Parser sitasi presisi dan PDF viewer (termasuk tombol viewer native browser; commit `8ef96ac`) | Uji UI end-to-end di browser dan deployment VPS |
| Reranker JSON diuji langsung dengan skor relevan 10 dan tidak relevan 0 | Dockerfile, HTTPS/reverse proxy, serta pengelola proses untuk VPS |

Hasil evaluasi yang tercantum di atas adalah baseline lama, bukan angka terukur dari konfigurasi provider saat ini. Streaming jawaban, feedback 👍/👎, ekspor riwayat, dan pengembangan lanjutan multi-dokumen juga masih menjadi pekerjaan berikutnya.



Pertanyaan lanjutan seperti *"berapa maksimalnya?"* ditulis ulang lebih dulu menjadi pertanyaan mandiri sebelum pencarian berjalan, sebab tanpa itu konteks yang salah sudah terambil sejak awal.

### Hasil Pengukuran
Angka MRR, groundedness, dan penolakan yang ditampilkan pada bagian evaluasi adalah hasil baseline lama. Evaluasi retrieval dan jawaban dengan provider Bandel belum dijalankan ulang, jadi angka tersebut belum menggambarkan konfigurasi aktif.

### Panel Debug
Setiap jawaban dilengkapi rincian retrieval yang bisa dibuka: skor dense, BM25, RRF, dan reranker per kandidat, jalur mana yang menemukannya, keputusan gate relevansi, serta latensi tiap tahap. Tujuannya agar proses retrieval bisa diaudit, bukan menjadi kotak hitam.

### Banyak Dokumen dalam Satu Index
Kampus punya lebih dari satu sumber resmi: pedoman akademik, kalender akademik, SK Rektor. Cukup letakkan PDF-nya di folder `data/` lalu bangun ulang index. Setiap chunk mencatat dokumen asalnya, sebab nomor halaman hanya bermakna dalam konteks dokumennya — "halaman 48" pada pedoman akademik dan pada SK Rektor merujuk hal yang sama sekali berbeda. Pencarian juga bisa dibatasi ke dokumen tertentu lewat sidebar.

### Peringatan Index Usang
Hash setiap dokumen dicatat saat index dibangun, lalu diperiksa ulang setiap aplikasi dimuat. Kalau PDF sumbernya diperbarui, aplikasi memberi tahu bahwa index perlu dibangun ulang. Tanpa pemeriksaan ini, chatbot akan terus menjawab dari dokumen edisi lama dengan sitasi yang tampak sah — kegagalan yang paling sulit disadari pengguna.

### Pengujian
456 tes offline lulus pada verifikasi terakhir, tanpa API key provider dan tanpa index. Jalankan dengan `python -m pytest`.

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
