"""
config.py — Konfigurasi terpusat seluruh pipeline RAG.

Konfigurasi provider dipisahkan dengan sengaja:

- **LLM provider** memakai endpoint OpenAI-compatible milik Bandel AI untuk
  jawaban, reranker, dan query rewriting.
- **Embedding provider** tetap memakai Gemini hanya untuk embedding karena
  endpoint Bandel yang tersedia tidak menyediakan `/embeddings`.

Jangan memakai key Bandel sebagai key Gemini embedding. Keduanya memiliki
format, endpoint, dan hak akses yang berbeda.
"""

from __future__ import annotations

import os
import tomllib
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
# Provider
# ──────────────────────────────────────────────

def _read_model_setting(name: str, default: str) -> str:
    """Read a model ID from environment or local Streamlit secrets."""
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()

    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    try:
        with secrets_path.open("rb") as secrets_file:
            secrets = tomllib.load(secrets_file)
    except (OSError, tomllib.TOMLDecodeError):
        return default

    value = secrets.get(name)
    return str(value).strip() if value and str(value).strip() else default

#: Provider chat OpenAI-compatible. Base URL dapat dioverride lewat
#: `LLM_BASE_URL` atau `OPENAI_BASE_URL` di environment/secrets.
LLM_BASE_URL = "https://bandelbanget.xyz/v1"
LLM_PROVIDER = "openai_compatible"

#: Pilih model eksplisit yang sudah diuji. Alias `auto` saat ini dapat
#: diarahkan ke model yang dinonaktifkan/stok habis oleh gateway.
LLM_MODEL = _read_model_setting("LLM_MODEL", "deepseek-v4-flash")
UTILITY_MODEL = _read_model_setting("UTILITY_MODEL", "deepseek-v4-flash")
MODEL_FALLBACKS: dict[str, tuple[str, ...]] = {
    "auto": ("deepseek-v4-flash", "glm-5.3-flash"),
    "deepseek-v4-flash": ("glm-5.3-flash",),
    "glm-5.3-flash": ("deepseek-v4-flash",),
    # Keep old explicit settings recoverable when a listed model is disabled.
    "gpt-5.6": ("deepseek-v4-flash", "glm-5.3-flash"),
    "claude-sonnet-5": ("deepseek-v4-flash", "glm-5.3-flash"),
}

#: Embedding tetap Gemini karena endpoint Bandel yang diuji tidak menyediakan
#: `/embeddings`. Jangan ubah tanpa membangun ulang FAISS index.
EMBEDDING_PROVIDER = "gemini"
EMBEDDING_MODEL = "gemini-embedding-001"

#: Task type embedding. Index dan query harus memakai ruang embedding yang
#: sama.
TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_TYPE_QUERY = "RETRIEVAL_QUERY"

#: Suhu rendah untuk jawaban faktual.
ANSWER_TEMPERATURE = 0.2
ANSWER_MAX_TOKENS = 4096

#: Tugas bantu harus deterministik.
UTILITY_TEMPERATURE = 0.0


# ──────────────────────────────────────────────
# Aplikasi publik/operator
# ──────────────────────────────────────────────

PUBLIC_APP_MODE = "public"
OPERATOR_APP_MODE = "operator"


# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 250
MIN_CHUNK_SIZE = 120

#: Versi logika pipeline ingestion (ekstraksi + chunking).
INGESTION_PIPELINE_VERSION = 3


# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class RetrievalConfig:
    """Parameter satu kali jalan pipeline retrieval."""

    top_k: int = 4
    dense_candidates: int = 20
    lexical_candidates: int = 20
    fusion_candidates: int = 10

    use_hybrid: bool = True
    use_rerank: bool = True
    use_query_rewrite: bool = True

    # Batasi pencarian ke dokumen tertentu; tuple kosong berarti semua.
    document_filter: tuple[str, ...] = ()

    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0

    #: Dengan task-type embedding, skor cosine berada di pita tinggi dan
    #: overlap dengan pertanyaan OOS. Ini hanya lantai kewajaran; reranker
    #: adalah gate presisi.
    dense_threshold: float = 0.70
    lexical_coverage_threshold: float = 0.60
    rerank_min_score: float = 4.0

    def with_overrides(self, **kwargs) -> "RetrievalConfig":
        """Kembalikan salinan config dengan sebagian field diganti."""
        known = {k: v for k, v in kwargs.items() if v is not None}
        unknown = set(known) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Field config tidak dikenal: {sorted(unknown)}")
        return replace(self, **known)


DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig()

ABLATION_PRESETS: dict[str, RetrievalConfig] = {
    "dense": RetrievalConfig(use_hybrid=False, use_rerank=False, use_query_rewrite=False),
    "hybrid": RetrievalConfig(use_hybrid=True, use_rerank=False, use_query_rewrite=False),
    "hybrid_rerank": RetrievalConfig(use_hybrid=True, use_rerank=True, use_query_rewrite=False),
}


# ──────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────

EMBED_BATCH_SIZE = 10
EMBED_BATCH_DELAY = 2.0
EMBED_MAX_RETRIES = 5
