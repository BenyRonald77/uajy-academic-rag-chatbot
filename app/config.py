"""
config.py — Konfigurasi terpusat untuk seluruh pipeline RAG.

Satu sumber kebenaran untuk nama model dan parameter retrieval, dipakai
bersama oleh UI Streamlit, script ingestion, dan runner evaluasi.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


# ──────────────────────────────────────────────
# Path
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "index"
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "eval"

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"
INDEX_INFO_PATH = INDEX_DIR / "index_info.json"


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

#: Model utama untuk menghasilkan jawaban akhir.
LLM_MODEL = "gemini-2.5-flash"

#: Model murah & cepat untuk tugas bantu (rerank, query rewriting).
UTILITY_MODEL = "gemini-2.5-flash-lite"

#: Model embedding untuk chunk dokumen dan query.
EMBEDDING_MODEL = "gemini-embedding-001"

#: Suhu rendah untuk jawaban faktual.
ANSWER_TEMPERATURE = 0.2
ANSWER_MAX_TOKENS = 2048

#: Tugas bantu harus deterministik.
UTILITY_TEMPERATURE = 0.0


# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────

CHUNK_SIZE = 1400          # karakter (~350 token)
CHUNK_OVERLAP = 250        # karakter (~62 token)
MIN_CHUNK_SIZE = 120       # chunk di bawah ini dibuang


# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class RetrievalConfig:
    """
    Parameter satu kali jalan pipeline retrieval.

    Pipeline lengkapnya:

        query
          └─ (opsional) query rewriting berbasis riwayat chat
               ├─ dense search  (FAISS cosine)      → top `dense_candidates`
               └─ lexical search (BM25 Okapi)       → top `lexical_candidates`
                    └─ Reciprocal Rank Fusion       → top `fusion_candidates`
                         └─ (opsional) LLM reranker → top `top_k`
                              └─ gate relevansi     → konteks atau penolakan
    """

    # Jumlah chunk final yang dikirim ke LLM sebagai konteks.
    top_k: int = 4

    # Berapa kandidat yang diambil tiap jalur sebelum fusi.
    dense_candidates: int = 20
    lexical_candidates: int = 20

    # Berapa kandidat hasil fusi yang diteruskan ke reranker.
    fusion_candidates: int = 10

    # Toggle tahap pipeline.
    use_hybrid: bool = True
    use_rerank: bool = True
    use_query_rewrite: bool = True

    # Konstanta Reciprocal Rank Fusion. 60 adalah nilai standar dari paper
    # Cormack et al. (2009); makin besar makin datar bobot antar peringkat.
    rrf_k: int = 60

    # Bobot relatif tiap jalur saat fusi.
    dense_weight: float = 1.0
    lexical_weight: float = 1.0

    # ── Gate relevansi (pertahanan anti-halusinasi lapis pertama) ──
    # Cosine similarity minimum agar jalur dense dianggap menemukan sesuatu.
    dense_threshold: float = 0.30

    # Porsi bobot IDF token query yang harus muncul di chunk agar jalur
    # leksikal dianggap cocok. Menjaga pertanyaan di luar cakupan tetap
    # ditolak walaupun BM25 mengembalikan skor tidak nol.
    lexical_coverage_threshold: float = 0.60

    # Skor reranker minimum (skala 0-10) agar kandidat dipakai sebagai konteks.
    rerank_min_score: float = 4.0

    def with_overrides(self, **kwargs) -> "RetrievalConfig":
        """Kembalikan salinan config dengan sebagian field diganti."""
        known = {k: v for k, v in kwargs.items() if v is not None}
        unknown = set(known) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Field config tidak dikenal: {sorted(unknown)}")
        return replace(self, **known)


#: Konfigurasi default aplikasi.
DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()

#: Preset untuk perbandingan/ablation study di eval.
ABLATION_PRESETS: dict[str, RetrievalConfig] = {
    "dense": RetrievalConfig(use_hybrid=False, use_rerank=False, use_query_rewrite=False),
    "hybrid": RetrievalConfig(use_hybrid=True, use_rerank=False, use_query_rewrite=False),
    "hybrid_rerank": RetrievalConfig(use_hybrid=True, use_rerank=True, use_query_rewrite=False),
}


# ──────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────

EMBED_BATCH_SIZE = 10      # chunk per request embedding
EMBED_BATCH_DELAY = 2.0    # detik jeda antar batch (hindari rate limit)
EMBED_MAX_RETRIES = 5
