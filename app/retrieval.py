"""
retrieval.py — Modul retrieval: cari chunk dokumen yang paling relevan.

Memuat FAISS index + metadata, lalu melakukan similarity search
berdasarkan embedding query pengguna.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from dataclasses import dataclass


# ──────────────────────────────────────────────
# Konfigurasi
# ──────────────────────────────────────────────

INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
DEFAULT_TOP_K = 4
SIMILARITY_THRESHOLD = 0.30  # di bawah ini, dianggap tidak relevan


@dataclass
class RetrievalResult:
    """Hasil retrieval satu chunk."""
    text: str
    page_numbers: list[int]
    section_title: str
    similarity_score: float
    chunk_index: int


class DocumentRetriever:
    """Retriever yang menggunakan FAISS index untuk similarity search."""

    def __init__(self, index_dir: str | Path | None = None):
        """
        Inisialisasi retriever.

        Args:
            index_dir: Path ke folder index. Default: ../index/
        """
        self.index_dir = Path(index_dir) if index_dir else INDEX_DIR
        self.index = None
        self.metadata = None
        self._load_index()

    def _load_index(self) -> None:
        """Load FAISS index dan metadata dari disk."""
        index_path = self.index_dir / "faiss.index"
        metadata_path = self.index_dir / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index tidak ditemukan di {index_path}. "
                "Jalankan `python ingestion/build_index.py` terlebih dahulu."
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata tidak ditemukan di {metadata_path}. "
                "Jalankan `python ingestion/build_index.py` terlebih dahulu."
            )

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = DEFAULT_TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[RetrievalResult]:
        """
        Cari chunk paling relevan berdasarkan query embedding.

        Args:
            query_embedding: Embedding vector dari pertanyaan user.
            top_k: Jumlah chunk teratas yang diambil.
            threshold: Ambang batas similarity minimum.

        Returns:
            List of RetrievalResult, diurutkan dari yang paling relevan.
            Kosong jika semua hasil di bawah threshold.
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index belum di-load. Panggil _load_index() dulu.")

        # Siapkan query vector
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        # Search
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue

            if score < threshold:
                continue

            meta = self.metadata[idx]
            results.append(RetrievalResult(
                text=meta["text"],
                page_numbers=meta["page_numbers"],
                section_title=meta.get("section_title", ""),
                similarity_score=float(score),
                chunk_index=meta["chunk_index"],
            ))

        return results

    @property
    def total_chunks(self) -> int:
        """Jumlah total chunk dalam index."""
        return self.index.ntotal if self.index else 0

    @property
    def document_info(self) -> dict:
        """Informasi ringkas tentang dokumen yang di-index."""
        if not self.metadata:
            return {}

        all_pages = set()
        all_sections = set()
        for meta in self.metadata:
            all_pages.update(meta.get("page_numbers", []))
            section = meta.get("section_title", "")
            if section:
                all_sections.add(section)

        return {
            "total_chunks": len(self.metadata),
            "total_pages": len(all_pages),
            "page_range": f"{min(all_pages)}-{max(all_pages)}" if all_pages else "N/A",
            "sections": sorted(all_sections),
        }


def format_sources(results: list[RetrievalResult]) -> str:
    """
    Format sumber referensi untuk ditampilkan ke user.

    Args:
        results: List of RetrievalResult.

    Returns:
        String format sumber yang rapi.
    """
    if not results:
        return ""

    sources = []
    seen_pages = set()

    for r in results:
        pages_str = ", ".join(str(p) for p in r.page_numbers)
        key = (pages_str, r.section_title)

        if key not in seen_pages:
            seen_pages.add(key)
            source = f"📄 Halaman {pages_str}"
            if r.section_title:
                source += f" — *{r.section_title}*"
            source += f" (relevansi: {r.similarity_score:.0%})"
            sources.append(source)

    return "\n".join(sources)
