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
#:
#: Pendahulunya, ``gemini-2.5-flash``, kini menolak permintaan dengan 404
#: ("no longer available to new users") — persis seperti yang lebih dulu
#: terjadi pada ``gemini-2.5-flash-lite``. Model Gemini dihapus tanpa
#: pemberitahuan di dalam kode, jadi nama model tunggal bukan konfigurasi yang
#: aman. Lihat `LLM_MODEL_FALLBACKS` di bawah.
LLM_MODEL = "gemini-3.6-flash"

#: Model murah & cepat untuk tugas bantu (rerank, query rewriting).
#:
#: Pendahulunya, ``gemini-2.5-flash-lite``, kini menolak permintaan dengan
#: 404 ("no longer available to new users") — padahal ``models.list()`` masih
#: melaporkannya tersedia, sehingga kegagalannya hanya terlihat saat
#: dipanggil sungguhan. Karena reranker dan query rewriter sengaja dibuat
#: fail-safe, keduanya diam-diam berhenti bekerja tanpa pesan error apa pun.
#:
#: ``gemini-3.5-flash-lite`` justru lebih cocok untuk peran ini: token
#: *thinking*-nya nol secara default, jadi lebih cepat, lebih murah, dan
#: deterministik. Catatan: model ini MENOLAK parameter ``thinking_config``
#: dengan 400 INVALID_ARGUMENT, jadi parameter itu tidak boleh dikirim.
UTILITY_MODEL = "gemini-3.5-flash-lite"

#: Model embedding untuk chunk dokumen dan query.
#:
#: JANGAN diubah tanpa membangun ulang index. Query harus di-embed oleh model
#: yang sama dengan yang membangun vektornya, kalau tidak seluruh skor
#: kemiripan menjadi tidak bermakna. Karena itu model embedding tidak diberi
#: rantai fallback: berpindah diam-diam justru merusak retrieval.
EMBEDDING_MODEL = "gemini-embedding-001"

#: Model pengganti bila model utama tidak tersedia, dicoba berurutan.
#:
#: Dua model sudah dihapus di tengah masa hidup proyek ini, dan `models.list()`
#: tetap melaporkannya tersedia sehingga kegagalannya baru muncul saat
#: dipanggil sungguhan. Tanpa rantai ini, penghapusan model berikutnya akan
#: menampilkan galat 404 mentah kepada mahasiswa di jendela chat.
#:
#: Perpindahan hanya dilakukan untuk galat yang khas per-model (404 model
#: hilang, 503 model kelebihan beban). Galat kuota 429 bersifat akun, bukan
#: model, jadi berpindah model tidak menolong dan hanya menghabiskan sisa
#: kuota lebih cepat.
MODEL_FALLBACKS: dict[str, tuple[str, ...]] = {
    "gemini-3.6-flash": ("gemini-3.5-flash", "gemini-3.8-flash"),
    "gemini-3.5-flash-lite": ("gemini-3.6-flash", "gemini-3.5-flash"),
}

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

    # Batasi pencarian ke dokumen tertentu. Tuple kosong berarti semua
    # dokumen. Berupa tuple, bukan list, agar dataclass ini tetap frozen.
    document_filter: tuple[str, ...] = ()

    # Konstanta Reciprocal Rank Fusion. 60 adalah nilai standar dari paper
    # Cormack et al. (2009); makin besar makin datar bobot antar peringkat.
    rrf_k: int = 60

    # Bobot relatif tiap jalur saat fusi.
    dense_weight: float = 1.0
    lexical_weight: float = 1.0

    # ── Gate relevansi (pertahanan anti-halusinasi) ──
    #
    # PENTING — hasil pengukuran pada index saat ini:
    # Sejak embedding dibangun dengan task type RETRIEVAL_DOCUMENT/QUERY,
    # cosine similarity menyempit ke pita tinggi dan TUMPANG TINDIH antara
    # pertanyaan yang relevan dan yang di luar cakupan:
    #
    #     in-scope     : 0,762 – 0,866
    #     out-of-scope : 0,729 – 0,766
    #
    # Artinya tidak ada satu nilai ambang pun yang mampu memisahkan keduanya.
    # Ambang 0,30 warisan index lama (embedding generik) meloloskan SEMUA
    # pertanyaan di luar cakupan — akurasi penolakannya terukur 0%.
    #
    # Karena itu ambang ini diturunkan perannya menjadi *lantai kewajaran*
    # untuk menangkap query patologis, BUKAN pemisah relevansi. Yang menjadi
    # gate presisi sebenarnya adalah reranker: pada pengukuran yang sama ia
    # menolak 5/5 pertanyaan di luar cakupan dengan memberi skor 0–1.
    #
    # Konsekuensinya: mematikan `use_rerank` melemahkan pertahanan
    # anti-halusinasi di tingkat retrieval secara signifikan.
    dense_threshold: float = 0.70

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
