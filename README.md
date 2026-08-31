<div align="center">

# UAJY Academic Document RAG Chatbot

**Accurate, Hallucination-Free Academic Question Answering with Streamlit, FAISS, and Google Gemini**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit 1.49](https://img.shields.io/badge/Streamlit-1.49-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://aistudio.google.com/)
[![Gemini Embedding](https://img.shields.io/badge/Embedding-gemini--embedding--001-blue?style=flat-square)](https://ai.google.dev/)
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

- 🎯 **Strict Document Grounding:** Answers are generated **exclusively** from retrieved PDF context chunks.
- 📑 **Exact Source Citations:** Every answer automatically specifies exact **page numbers** and **chapter/section titles** for auditability.
- 🛡️ **Anti-Hallucination Guardrails:** Implements a two-layer defense mechanism (similarity threshold filtering + strict system prompt refusal) to reject queries not found in the documents.
- 🔍 **Interactive Document Explorer:** Built-in dashboard to search, filter, and inspect 350 indexed text chunks across 112 document pages.
- 📊 **Embedded Evaluation Suite:** Quantitative evaluation testbed measuring Retrieval Recall@K, Refusal Correctness, and Query Latency.
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
                  │  • PDF Text & Table Extraction       │
                  │  • Semantic / Paragraph Chunking    │
                  │  • Gemini Embedding (Dim: 3072)     │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │        Local FAISS Vector Index     │
                  │     (faiss.index + metadata.json)   │
                  └──────────────────┬──────────────────┘
                                     │
 ────────────────────────────────────┼────────────────────────────────────
                                     │ (Online / Runtime Query Flow)
                                     ▼
┌──────────────┐          ┌───────────────────────┐          ┌───────────────────────┐
│ User Query   │ ───────► │ Similarity Retrieval  │ ───────► │ Context Prompt        │
│ (Streamlit)  │          │ (Top-K Chunks, FAISS) │          │ Builder + History     │
└──────────────┘          └───────────────────────┘          └───────────┬───────────┘
                                                                         │
                                                                         ▼
                                                             ┌───────────────────────┐
                                                             │ Google Gemini 2.5     │
                                                             │ Flash API             │
                                                             └───────────┬───────────┘
                                                                         │
                                                                         ▼
                                                             ┌───────────────────────┐
                                                             │ Grounded Answer +     │
                                                             │ Page & Section Source │
                                                             └───────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend UI** | [Streamlit](https://streamlit.io/) 1.49 | Interactive chat interface, document explorer, & metrics dashboard |
| **LLM Generation** | [Google Gemini 2.5 Flash](https://ai.google.dev/) | High-speed, context-rich reasoning in Indonesian & English |
| **Embeddings** | [gemini-embedding-001](https://ai.google.dev/) | 3072-dimensional multilingual semantic representation |
| **Vector Store** | [FAISS (CPU)](https://github.com/facebookresearch/faiss) | In-memory Inner Product / Cosine Similarity search (< 50ms) |
| **PDF Extraction** | [pdfplumber](https://github.com/jsvine/pdfplumber) | Layout-aware text and tabular data extraction from PDF |
| **Language** | Python 3.10+ | Core application runtime |

---

## 📁 Repository Structure

```
ChatBot RAG kampus/
├── .streamlit/
│   ├── config.toml           # Streamlit dark theme configuration
│   └── secrets.toml          # Secret API keys (excluded from git)
├── app/
│   ├── __init__.py
│   ├── llm_client.py         # Google Gemini LLM & Embedding wrapper
│   ├── prompt_builder.py     # Prompt template with strict guardrails
│   ├── retrieval.py          # FAISS similarity search & source formatter
│   └── main_streamlit.py     # Streamlit multi-page UI application
├── assets/
│   └── banner.jpg            # README hero banner image
├── data/
│   ├── README.md
│   └── Buku-Pedoman-Akademik-Fakultas-Teknologi-Industri-2025-2026.pdf
├── eval/
│   ├── __init__.py
│   ├── test_questions.json   # 20 benchmark test questions (In-scope & OOS)
│   └── run_eval.py           # Automated evaluation runner script
├── index/
│   ├── README.md
│   ├── faiss.index           # Persisted binary FAISS index (350 vectors)
│   └── metadata.json         # Chunk metadata (page numbers & headings)
├── ingestion/
│   ├── __init__.py
│   ├── extract_text.py       # PDF parsing & text cleaner
│   ├── chunking.py           # Section & paragraph chunker with overlap
│   └── build_index.py        # Pipeline orchestrator to build FAISS index
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

Run the automated ingestion pipeline to parse the PDF, generate embeddings, and build the FAISS index:

```bash
python ingestion/build_index.py
```

### 4. Run the Streamlit Web Application

```bash
python -m streamlit run app/main_streamlit.py
```

Open your browser at **`http://localhost:8501`**.

---

## 📊 Evaluation & Benchmarking

Run the automated evaluation benchmark on the 20 test question suite:

```bash
python eval/run_eval.py
```

### Benchmark Summary

| Metric | Target | Result | Status |
|---|---|---|---|
| **Retrieval Recall@4** (In-Scope) | $\ge 80.0\%$ | **100.0%** (15/15) | ✅ Passed |
| **Refusal Correctness** (Out-of-Scope) | $100.0\%$ | **100.0%** (5/5) | ✅ Passed |
| **Average Retrieval Latency** | $< 1.0\text{s}$ | **0.42s** | ✅ Passed |
| **Total Response Latency** | $< 5.0\text{s}$ | **~1.85s** | ✅ Passed |

---

## 🇮🇩 Ringkasan Proyek (Bahasa Indonesia)

**RAG Chatbot Dokumen Kampus UAJY** adalah asisten tanya jawab berbasis *Retrieval-Augmented Generation* (RAG) yang dirancang untuk menjawab pertanyaan seputar buku pedoman akademik Fakultas Teknologi Industri Universitas Atma Jaya Yogyakarta (FTI UAJY) Tahun Akademik 2025/2026.

### Mengapa Menggunakan RAG?
- **Bebas Halusinasi:** Model LLM diinstruksikan menjawab **hanya** berdasarkan teks yang ditemukan di dalam dokumen PDF resmi.
- **Kutipan Transparan:** Setiap jawaban menyertakan rujukan nomor halaman dan judul bab/pasal.
- **Ringan & Hemat Resource:** Vector store dijalankan secara lokal via FAISS (CPU), sedangkan inferensi dilakukan via API Google Gemini 2.5 Flash.

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
