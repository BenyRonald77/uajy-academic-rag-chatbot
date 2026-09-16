"""
retrieval.py — Hybrid retrieval: dense (FAISS) + leksikal (BM25) + fusi RRF
+ reranking LLM, dilengkapi gate relevansi anti-halusinasi.

Alur satu query:

    query pengguna
      │
      ├─ (opsional) query rewriting berbasis riwayat chat
      │      "berapa syaratnya?" → "berapa syarat pengajuan cuti kuliah?"
      │
      ├─ Jalur dense   : embedding Gemini → FAISS cosine  → top-N
      ├─ Jalur leksikal: BM25 Okapi                        → top-N
      │
      ├─ Reciprocal Rank Fusion (RRF) menggabungkan kedua peringkat
      │
      ├─ (opsional) reranker LLM menilai ulang relevansi tiap kandidat
      │
      └─ Gate relevansi → konteks untuk LLM, atau penolakan sopan

Pertahanan anti-halusinasi berlapis:

1. **Gate murah** — cosine similarity minimum ATAU cakupan istilah leksikal
   yang tinggi. Menyaring pertanyaan yang jelas di luar cakupan tanpa biaya
   tambahan.
2. **Gate reranker** — kandidat dengan skor relevansi rendah dibuang. Ini
   lapisan yang menangkap kasus sulit: pertanyaan di luar cakupan yang
   kebetulan berbagi kosakata dengan dokumen (mis. "Siapa presiden
   Indonesia?" cocok secara leksikal dengan "Peraturan Presiden Republik
   Indonesia" di dokumen).
3. **Gate prompt** — instruksi sistem yang ketat pada LLM penjawab.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from app.config import (
    DEFAULT_RETRIEVAL_CONFIG,
    INDEX_DIR,
    RetrievalConfig,
)
from app.lexical_index import LexicalIndex
from app.text_utils import content_tokens

#: Sebuah kandidat yang hanya ditemukan lewat BM25 tetap harus lolos ambang
#: kemiripan semantik minimum ini (dinyatakan sebagai rasio terhadap
#: `dense_threshold`) sebelum boleh "menyelamatkan" query dari penolakan.
LEXICAL_RESCUE_DENSE_FLOOR_RATIO = 0.7

#: Minimum jumlah istilah query yang harus benar-benar cocok agar jalur
#: leksikal boleh menyelamatkan query. Satu kata yang cocok terlalu lemah.
LEXICAL_RESCUE_MIN_TERMS = 2


# ──────────────────────────────────────────────
# Struktur hasil
# ──────────────────────────────────────────────

@dataclass
class RetrievalCandidate:
    """
    Satu chunk kandidat lengkap dengan skor dari setiap tahap pipeline.

    Menyimpan skor per tahap membuat proses retrieval bisa diaudit: panel
    debug di UI dan runner evaluasi dapat menunjukkan persis mengapa sebuah
    chunk terpilih.
    """

    chunk_index: int
    text: str
    page_numbers: list[int]
    section_title: str = ""
    heading_path: list[str] = field(default_factory=list)
    source_document: str = ""

    # Jalur dense.
    dense_score: float = 0.0
    dense_rank: int | None = None

    # Jalur leksikal.
    lexical_score: float = 0.0
    lexical_rank: int | None = None
    matched_terms: tuple[str, ...] = ()
    lexical_coverage: float = 0.0

    # Fusi dan reranking.
    fusion_score: float = 0.0
    rerank_score: float | None = None
    final_rank: int = 0

    @property
    def similarity_score(self) -> float:
        """Alias kompatibilitas untuk kode lama yang membaca cosine similarity."""
        return self.dense_score

    @property
    def page_label(self) -> str:
        """Rentang halaman yang ringkas, mis. "12" atau "12-14"."""
        if not self.page_numbers:
            return "-"
        first, last = min(self.page_numbers), max(self.page_numbers)
        return str(first) if first == last else f"{first}-{last}"

    @property
    def heading_label(self) -> str:
        """
        Judul bagian untuk ditampilkan pada sitasi.

        Hanya heading **terdalam**, bukan seluruh jalurnya. Pengukuran pada
        dokumen ini menunjukkan alasannya: judul akar berasal dari aturan
        "baris huruf kapital adalah judul", yang pada dokumen tanpa penanda
        "BAB" ikut menangkap label kolom tabel. Akibatnya 44% chunk memiliki
        akar yang tidak bermakna, sedangkan heading terdalamnya hanya 8% —
        dan justru yang terdalam itulah yang paling spesifik serta paling
        berguna bagi pembaca ("H. Cuti Studi", bukan "PROGRAM › ...").

        Jalur lengkapnya tetap dipakai untuk embedding dan tersedia di panel
        debug lewat `heading_path_label`.
        """
        if self.section_title:
            return self.section_title
        return self.heading_path[-1] if self.heading_path else ""

    @property
    def heading_path_label(self) -> str:
        """Jalur heading lengkap, untuk audit di panel debug."""
        if self.heading_path:
            return " › ".join(self.heading_path)
        return self.section_title

    @property
    def document_label(self) -> str:
        """
        Nama dokumen tanpa ekstensi, untuk ditampilkan pada sitasi.

        Kosong pada index dokumen tunggal versi lama, sehingga sitasinya tetap
        ringkas seperti sebelumnya.
        """
        if not self.source_document:
            return ""
        return self.source_document.rsplit(".", 1)[0]

    @property
    def retrieved_by(self) -> str:
        """Jalur mana yang menemukan chunk ini."""
        if self.dense_rank is not None and self.lexical_rank is not None:
            return "dense+bm25"
        if self.dense_rank is not None:
            return "dense"
        if self.lexical_rank is not None:
            return "bm25"
        return "-"

    @property
    def display_score(self) -> float:
        """Skor tunggal untuk ditampilkan ke pengguna (0.0–1.0)."""
        if self.rerank_score is not None:
            return self.rerank_score / 10.0
        return self.dense_score


#: Nama lama yang masih dipakai modul lain.
RetrievalResult = RetrievalCandidate


@dataclass
class RetrievalOutcome:
    """Hasil lengkap satu kali jalan pipeline retrieval."""

    original_query: str
    effective_query: str
    contexts: list[RetrievalCandidate] = field(default_factory=list)
    candidates: list[RetrievalCandidate] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    was_rewritten: bool = False

    #: True bila tahap reranking diminta tetapi gagal dijalankan. Penting
    #: untuk diketahui: saat ini reranker adalah satu-satunya tahap yang
    #: mampu menolak pertanyaan di luar cakupan, jadi kegagalannya berarti
    #: pertahanan anti-halusinasi tingkat retrieval sedang tidak aktif.
    rerank_failed: bool = False

    timings_ms: dict[str, float] = field(default_factory=dict)
    gate: dict[str, float | bool | None] = field(default_factory=dict)
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())


# ──────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[int]],
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
) -> dict[int, float]:
    """
    Gabungkan beberapa daftar peringkat menjadi satu skor per dokumen.

    RRF (Cormack et al., 2009) menjumlahkan ``weight / (k + rank)`` dari
    setiap daftar. Karena hanya memakai *peringkat*, bukan skor mentah, RRF
    tidak butuh normalisasi antar sistem yang skalanya berbeda jauh — cocok
    untuk menggabungkan cosine similarity (0–1) dengan BM25 (tak terbatas).

    Args:
        ranked_lists: Nama jalur → daftar ``doc_id`` terurut dari terbaik.
        weights: Bobot per jalur. Default 1.0.
        rrf_k: Konstanta peredam. Makin besar, makin datar bobot antar peringkat.

    Returns:
        ``doc_id`` → skor fusi.
    """
    weights = weights or {}
    scores: dict[int, float] = {}

    for source, doc_ids in ranked_lists.items():
        weight = weights.get(source, 1.0)
        for rank, doc_id in enumerate(doc_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rrf_k + rank)

    return scores


# ──────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────

class HybridRetriever:
    """
    Retriever yang menggabungkan FAISS dense search dengan BM25 leksikal.

    Index leksikal dibangun di memori dari ``metadata.json`` saat inisialisasi,
    sehingga tidak ada artefak tambahan yang perlu disimpan dan tidak ada
    biaya API.
    """

    def __init__(
        self,
        index_dir: str | Path | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.index_dir = Path(index_dir) if index_dir else INDEX_DIR
        self.config = config or DEFAULT_RETRIEVAL_CONFIG

        self.index: faiss.Index | None = None
        self.metadata: list[dict] | None = None
        self.index_info: dict = {}
        self.lexical_index: LexicalIndex | None = None

        self._load_index()
        self._build_lexical_index()

    # ── Loading ────────────────────────────────

    def _load_index(self) -> None:
        """Muat FAISS index, metadata, dan info build dari disk."""
        index_path = self.index_dir / "faiss.index"
        metadata_path = self.index_dir / "metadata.json"
        info_path = self.index_dir / "index_info.json"

        for path, label in ((index_path, "FAISS index"), (metadata_path, "Metadata")):
            if not path.exists():
                raise FileNotFoundError(
                    f"{label} tidak ditemukan di {path}. "
                    "Jalankan `python ingestion/build_index.py` terlebih dahulu."
                )

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    self.index_info = json.load(f)
            except (OSError, json.JSONDecodeError):
                self.index_info = {}

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index dan metadata tidak sinkron: {self.index.ntotal} vector "
                f"vs {len(self.metadata)} entri metadata. Bangun ulang index."
            )

    def _build_lexical_index(self) -> None:
        self.lexical_index = LexicalIndex.from_metadata(self.metadata or [])

    @property
    def embedding_task_type(self) -> str | None:
        """
        Task type embedding yang dipakai saat index dibangun.

        Query harus di-embed dengan task type yang cocok. Index versi lama
        (sebelum `index_info.json` ada) dibangun tanpa task type, jadi
        default-nya ``None``.
        """
        return self.index_info.get("embedding_task_type")

    # ── Pencarian per jalur ────────────────────

    def allowed_doc_ids(self, document_filter: tuple[str, ...]) -> set[int] | None:
        """
        Kumpulan ``doc_id`` yang lolos filter dokumen.

        Returns:
            ``None`` bila tidak ada filter, sehingga pemanggil bisa melewati
            penyaringan sama sekali.
        """
        if not document_filter or not self.metadata:
            return None

        diizinkan = set(document_filter)
        return {
            doc_id
            for doc_id, meta in enumerate(self.metadata)
            if meta.get("source_document", "") in diizinkan
        }

    def dense_search(
        self,
        query_embedding: list[float],
        top_n: int,
        allowed: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Cari lewat FAISS.

        Args:
            query_embedding: Embedding pertanyaan.
            top_n: Jumlah hasil yang diinginkan.
            allowed: Bila diberikan, hanya ``doc_id`` ini yang disertakan.

        Returns:
            Daftar ``(doc_id, cosine_score)`` terurut dari paling mirip.
        """
        query_vector = self._as_unit_vector(query_embedding)

        # FAISS IndexFlatIP tidak mendukung penyaringan bawaan, jadi saat filter
        # aktif hasilnya diambil lebih banyak lalu disaring. Tanpa ini, filter
        # dokumen bisa menyisakan kandidat yang terlalu sedikit.
        fetch = top_n if allowed is None else min(top_n * 5, self.index.ntotal)
        scores, indices = self.index.search(query_vector, min(fetch, self.index.ntotal))

        hasil: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc_id = int(idx)
            if allowed is not None and doc_id not in allowed:
                continue
            hasil.append((doc_id, float(score)))
            if len(hasil) >= top_n:
                break
        return hasil

    def lexical_search(self, query: str, top_n: int, allowed: set[int] | None = None):
        """Cari lewat BM25, dengan penyaringan dokumen opsional."""
        if self.lexical_index is None:
            return []

        fetch = top_n if allowed is None else top_n * 5
        hits = self.lexical_index.search(query, top_n=fetch)
        if allowed is not None:
            hits = [h for h in hits if h.doc_id in allowed]
        return hits[:top_n]

    def dense_scores_for(
        self,
        query_embedding: list[float],
        doc_ids: list[int],
    ) -> dict[int, float]:
        """
        Hitung cosine similarity untuk dokumen tertentu.

        Kandidat yang hanya ditemukan BM25 biasanya berada di luar top-N
        dense, jadi skor dense-nya tidak diketahui. Karena FAISS memakai
        `IndexFlatIP`, vector aslinya bisa direkonstruksi sehingga setiap
        kandidat tetap punya profil skor lengkap — dibutuhkan oleh gate
        relevansi maupun panel debug.
        """
        if not doc_ids:
            return {}

        query_vector = self._as_unit_vector(query_embedding)[0]
        result: dict[int, float] = {}
        for doc_id in doc_ids:
            try:
                stored = self.index.reconstruct(int(doc_id))
            except (RuntimeError, AttributeError):
                continue
            result[doc_id] = float(np.dot(query_vector, stored))
        return result

    @staticmethod
    def _as_unit_vector(embedding: list[float]) -> np.ndarray:
        vector = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        return vector

    # ── Pipeline lengkap ───────────────────────

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
        chat_history: list[dict] | None = None,
        api_key: str | None = None,
    ) -> RetrievalOutcome:
        """
        Jalankan pipeline retrieval lengkap untuk sebuah pertanyaan.

        Args:
            query: Pertanyaan mentah dari pengguna.
            config: Override konfigurasi retrieval.
            chat_history: Riwayat chat untuk query rewriting.
            api_key: Override API key Gemini.

        Returns:
            `RetrievalOutcome` berisi konteks terpilih, seluruh kandidat
            beserta skornya, status penolakan, dan waktu tiap tahap.
        """
        cfg = config or self.config
        timings: dict[str, float] = {}

        # Tahap 1 — query rewriting.
        effective_query, was_rewritten = self._maybe_rewrite(
            query, cfg, chat_history, api_key, timings
        )

        # Tahap 2 — pencarian dua jalur.
        started = time.perf_counter()
        from app.llm_client import get_query_embedding

        allowed = self.allowed_doc_ids(cfg.document_filter)

        query_embedding = get_query_embedding(
            effective_query,
            task_type=self.embedding_task_type,
            api_key=api_key,
        )
        dense_hits = self.dense_search(query_embedding, cfg.dense_candidates, allowed)
        timings["dense_ms"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        lexical_hits = (
            self.lexical_search(effective_query, cfg.lexical_candidates, allowed)
            if cfg.use_hybrid else []
        )
        timings["lexical_ms"] = (time.perf_counter() - started) * 1000

        # Tahap 3 — fusi.
        started = time.perf_counter()
        candidates = self._fuse(
            query=effective_query,
            dense_hits=dense_hits,
            lexical_hits=lexical_hits,
            query_embedding=query_embedding,
            cfg=cfg,
        )
        timings["fusion_ms"] = (time.perf_counter() - started) * 1000

        outcome = RetrievalOutcome(
            original_query=query,
            effective_query=effective_query,
            candidates=candidates,
            was_rewritten=was_rewritten,
            timings_ms=timings,
            config=cfg,
        )

        if not candidates:
            outcome.refused = True
            outcome.refusal_reason = "Tidak ada kandidat chunk yang ditemukan."
            outcome.gate = {"dense_pass": False, "lexical_pass": False}
            return outcome

        # Tahap 4 — gate murah.
        gate = self._evaluate_gate(candidates, effective_query, cfg)
        outcome.gate = gate
        if not gate["passed"]:
            outcome.refused = True
            outcome.refusal_reason = (
                f"Skor kemiripan tertinggi ({gate['max_dense_score']:.2f}) di bawah ambang "
                f"{cfg.dense_threshold:.2f} dan cakupan istilah "
                f"({gate['max_lexical_coverage']:.2f}) di bawah "
                f"{cfg.lexical_coverage_threshold:.2f}."
            )
            return outcome

        # Tahap 5 — reranking.
        if cfg.use_rerank and len(candidates) > 1:
            started = time.perf_counter()
            candidates, rerank_ok = self._rerank(
                effective_query, candidates, cfg, api_key
            )
            timings["rerank_ms"] = (time.perf_counter() - started) * 1000
            outcome.candidates = candidates
            outcome.rerank_failed = not rerank_ok

            if rerank_ok:
                passed = [
                    c for c in candidates
                    if (c.rerank_score or 0.0) >= cfg.rerank_min_score
                ]
                if not passed:
                    best = max((c.rerank_score or 0.0) for c in candidates)
                    outcome.refused = True
                    outcome.refusal_reason = (
                        f"Reranker menilai semua kandidat tidak relevan "
                        f"(skor tertinggi {best:.1f} < {cfg.rerank_min_score:.1f})."
                    )
                    outcome.gate["rerank_pass"] = False
                    return outcome

                outcome.gate["rerank_pass"] = True
                candidates = passed
            else:
                # Reranking tidak pernah terjadi. Jawaban tetap diberikan dari
                # urutan fusi, tetapi gate TIDAK boleh mengaku sudah lolos —
                # kalau tidak, kegagalan API akan tampak seperti keberhasilan.
                outcome.gate["rerank_pass"] = None

        # Tahap 6 — ambil top-k final.
        contexts = candidates[: cfg.top_k]
        for rank, candidate in enumerate(contexts, start=1):
            candidate.final_rank = rank
        outcome.contexts = contexts
        return outcome

    def _maybe_rewrite(
        self,
        query: str,
        cfg: RetrievalConfig,
        chat_history: list[dict] | None,
        api_key: str | None,
        timings: dict[str, float],
    ) -> tuple[str, bool]:
        if not (cfg.use_query_rewrite and chat_history):
            return query, False

        from app.query_rewriter import rewrite_query

        started = time.perf_counter()
        rewritten = rewrite_query(query, chat_history, api_key=api_key)
        timings["rewrite_ms"] = (time.perf_counter() - started) * 1000
        return rewritten, rewritten.strip() != query.strip()

    def _fuse(
        self,
        query: str,
        dense_hits: list[tuple[int, float]],
        lexical_hits: list,
        query_embedding: list[float],
        cfg: RetrievalConfig,
    ) -> list[RetrievalCandidate]:
        """Gabungkan hasil kedua jalur menjadi daftar kandidat terurut."""
        dense_by_id = {doc_id: score for doc_id, score in dense_hits}
        dense_rank_by_id = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_hits, 1)}
        lexical_by_id = {hit.doc_id: hit for hit in lexical_hits}
        lexical_rank_by_id = {hit.doc_id: rank for rank, hit in enumerate(lexical_hits, 1)}

        if cfg.use_hybrid:
            fusion_scores = reciprocal_rank_fusion(
                ranked_lists={
                    "dense": [doc_id for doc_id, _ in dense_hits],
                    "lexical": [hit.doc_id for hit in lexical_hits],
                },
                weights={"dense": cfg.dense_weight, "lexical": cfg.lexical_weight},
                rrf_k=cfg.rrf_k,
            )
        else:
            fusion_scores = {
                doc_id: 1.0 / (cfg.rrf_k + rank)
                for doc_id, rank in dense_rank_by_id.items()
            }

        # Lengkapi skor dense untuk kandidat yang hanya ditemukan BM25.
        missing_dense = [doc_id for doc_id in fusion_scores if doc_id not in dense_by_id]
        dense_by_id.update(self.dense_scores_for(query_embedding, missing_dense))

        query_terms = content_tokens(query)

        candidates: list[RetrievalCandidate] = []
        for doc_id, fusion_score in fusion_scores.items():
            meta = self.metadata[doc_id]
            lexical_hit = lexical_by_id.get(doc_id)

            coverage = lexical_hit.coverage if lexical_hit else (
                self.lexical_index.coverage(query_terms, doc_id) if self.lexical_index else 0.0
            )
            matched = lexical_hit.matched_terms if lexical_hit else ()

            candidates.append(RetrievalCandidate(
                chunk_index=meta.get("chunk_index", doc_id),
                text=meta["text"],
                page_numbers=meta.get("page_numbers", []),
                section_title=meta.get("section_title", ""),
                heading_path=meta.get("heading_path", []) or [],
                source_document=meta.get("source_document", ""),
                dense_score=dense_by_id.get(doc_id, 0.0),
                dense_rank=dense_rank_by_id.get(doc_id),
                lexical_score=lexical_hit.score if lexical_hit else 0.0,
                lexical_rank=lexical_rank_by_id.get(doc_id),
                matched_terms=matched,
                lexical_coverage=coverage,
                fusion_score=fusion_score,
            ))

        candidates.sort(key=lambda c: (c.fusion_score, c.dense_score), reverse=True)
        return candidates[: cfg.fusion_candidates]

    def _evaluate_gate(
        self,
        candidates: list[RetrievalCandidate],
        query: str,
        cfg: RetrievalConfig,
    ) -> dict:
        """
        Gate relevansi lapis pertama.

        Lolos bila salah satu terpenuhi:

        - **Jalur dense**: ada kandidat dengan cosine ≥ ``dense_threshold``.
        - **Penyelamatan leksikal**: ada kandidat dengan cakupan istilah
          tinggi, minimal dua istilah cocok, DAN kemiripan semantik masih di
          atas lantai lunak. Syarat ganda ini mencegah pertanyaan di luar
          cakupan lolos hanya karena berbagi beberapa kata umum dengan
          dokumen.
        """
        max_dense = max((c.dense_score for c in candidates), default=0.0)
        max_coverage = max((c.lexical_coverage for c in candidates), default=0.0)

        dense_pass = max_dense >= cfg.dense_threshold

        dense_floor = cfg.dense_threshold * LEXICAL_RESCUE_DENSE_FLOOR_RATIO
        lexical_pass = any(
            c.lexical_coverage >= cfg.lexical_coverage_threshold
            and len(c.matched_terms) >= LEXICAL_RESCUE_MIN_TERMS
            and c.dense_score >= dense_floor
            for c in candidates
        )

        return {
            "passed": dense_pass or lexical_pass,
            "dense_pass": dense_pass,
            "lexical_pass": lexical_pass,
            "max_dense_score": max_dense,
            "max_lexical_coverage": max_coverage,
            "dense_threshold": cfg.dense_threshold,
            "lexical_coverage_threshold": cfg.lexical_coverage_threshold,
        }

    def _rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        cfg: RetrievalConfig,
        api_key: str | None,
    ) -> tuple[list[RetrievalCandidate], bool]:
        from app.reranker import rerank_candidates

        return rerank_candidates(query, candidates, api_key=api_key)

    # ── Kompatibilitas & info ──────────────────

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
        threshold: float = 0.30,
    ) -> list[RetrievalCandidate]:
        """
        Pencarian dense-only berdasarkan embedding yang sudah dihitung.

        Dipertahankan untuk baseline evaluasi dan pemanggil lama. Untuk
        pipeline lengkap gunakan `retrieve()`.
        """
        results: list[RetrievalCandidate] = []
        for rank, (doc_id, score) in enumerate(self.dense_search(query_embedding, top_k), 1):
            if score < threshold:
                continue
            meta = self.metadata[doc_id]
            results.append(RetrievalCandidate(
                chunk_index=meta.get("chunk_index", doc_id),
                text=meta["text"],
                page_numbers=meta.get("page_numbers", []),
                section_title=meta.get("section_title", ""),
                heading_path=meta.get("heading_path", []) or [],
                source_document=meta.get("source_document", ""),
                dense_score=score,
                dense_rank=rank,
                final_rank=rank,
            ))
        return results

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal if self.index else 0

    @property
    def source_documents(self) -> list[str]:
        """
        Nama dokumen yang ada di dalam index, terurut.

        Index dokumen tunggal versi lama tidak menyimpan ``source_document``,
        sehingga daftarnya kosong dan UI menyembunyikan pemilih dokumen.
        """
        if not self.metadata:
            return []
        nama = {
            meta.get("source_document", "")
            for meta in self.metadata
            if meta.get("source_document")
        }
        return sorted(nama)

    @property
    def document_info(self) -> dict:
        """Ringkasan dokumen yang terindeks, untuk ditampilkan di UI."""
        if not self.metadata:
            return {}

        all_pages: set[int] = set()
        all_sections: set[str] = set()
        per_document: dict[str, dict] = {}

        for meta in self.metadata:
            pages = meta.get("page_numbers", [])
            all_pages.update(pages)

            section = meta.get("section_title", "")
            if section:
                all_sections.add(section)

            nama = meta.get("source_document", "")
            entri = per_document.setdefault(nama, {"chunks": 0, "pages": set()})
            entri["chunks"] += 1
            entri["pages"].update(pages)

        page_spans = [len(meta.get("page_numbers", [])) for meta in self.metadata]

        info = {
            "total_chunks": len(self.metadata),
            "total_pages": len(all_pages),
            "page_range": f"{min(all_pages)}-{max(all_pages)}" if all_pages else "N/A",
            "sections": sorted(all_sections),
            "max_pages_per_chunk": max(page_spans) if page_spans else 0,
            "avg_pages_per_chunk": (
                round(sum(page_spans) / len(page_spans), 2) if page_spans else 0
            ),
            "documents": [
                {"name": nama or "(tanpa nama)",
                 "chunks": data["chunks"],
                 "pages": len(data["pages"])}
                for nama, data in sorted(per_document.items())
            ],
            "total_documents": len(per_document),
        }
        if self.lexical_index:
            info["lexical"] = self.lexical_index.stats()
        info["index_info"] = self.index_info
        return info

    def freshness(self):
        """
        Periksa apakah index masih sesuai dengan dokumen sumbernya.

        Delegasi ke `app.index_status`, memakai catatan hash yang ditulis saat
        build. Tanpa pemeriksaan ini, PDF yang diperbarui akan dijawab dari
        index lama tanpa peringatan apa pun.
        """
        from app.index_status import check_index_freshness

        return check_index_freshness(self.index_info)


#: Nama lama yang masih dipakai UI dan evaluasi.
DocumentRetriever = HybridRetriever


# ──────────────────────────────────────────────
# Formatting sumber
# ──────────────────────────────────────────────

def _format_page_ranges(pages: set[int]) -> str:
    """Format halaman terurut sebagai rentang ringkas."""
    ordered = sorted(pages)
    if not ordered:
        return "-"

    ranges: list[str] = []
    start = previous = ordered[0]
    for page in ordered[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(str(start) if start == previous else f"{start}–{previous}")
        start = previous = page
    ranges.append(str(start) if start == previous else f"{start}–{previous}")
    return ", ".join(ranges)


def build_source_references(
    results: list[RetrievalCandidate],
    cited_pages: set[int] | None = None,
) -> list[dict]:
    """
    Buat data sumber terstruktur untuk kartu UI dan PDF viewer.

    Setiap item aman disimpan di `st.session_state` karena hanya berisi string,
    angka, dan list — tidak membawa object FAISS atau dataclass retrieval.
    """
    if not results:
        return []

    selected = results
    if cited_pages:
        selected = [
            result for result in results
            if set(result.page_numbers) & cited_pages
        ]
    if not selected:
        return []

    grouped: dict[str, dict] = {}
    for result in selected:
        document = result.source_document or ""
        entry = grouped.setdefault(document, {"pages": set(), "best": result})
        pages = set(result.page_numbers)
        if cited_pages:
            pages &= cited_pages
        entry["pages"].update(pages)
        if result.display_score > entry["best"].display_score:
            entry["best"] = result

    references: list[dict] = []
    for document, entry in grouped.items():
        best = entry["best"]
        references.append({
            "source_document": document,
            "document_label": best.document_label,
            "pages": sorted(entry["pages"]),
            "page_label": _format_page_ranges(entry["pages"]),
            "section_title": best.heading_label,
            "score": round(best.display_score, 4),
        })
    return references


def format_sources(
    results: list[RetrievalCandidate],
    cited_pages: set[int] | None = None,
    include_scores: bool = True,
) -> str:
    """Susun sumber terstruktur menjadi teks kartu yang ringkas."""
    references = build_source_references(results, cited_pages=cited_pages)
    if not references:
        return ""

    many_documents = len(references) > 1
    lines: list[str] = []
    for reference in references:
        source = "📄 "
        if many_documents and reference["document_label"]:
            source += f"{reference['document_label']} · "
        source += f"Halaman {reference['page_label']}"
        if reference["section_title"]:
            source += f" — *{reference['section_title']}*"
        if include_scores:
            source += f" (relevansi: {reference['score']:.0%})"
        lines.append(source)
    return "\n".join(lines)
