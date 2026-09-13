"""
lexical_index.py — Pencarian leksikal BM25 Okapi untuk jalur sparse pada
hybrid retrieval.

Kenapa perlu jalur leksikal?
    Embedding padat (dense) bagus menangkap makna, tapi lemah pada
    pencocokan istilah eksak. Dokumen akademik justru penuh istilah eksak
    yang tidak boleh salah: "IPK 3,50", "144 SKS", "Pasal 12", "BAB IV",
    kode mata kuliah. BM25 mengunggulkan chunk yang benar-benar memuat
    istilah tersebut, dan hasil kedua jalur digabung dengan RRF.

Kenapa diimplementasikan sendiri, bukan pakai `rank-bm25`?
    Korpusnya kecil (ratusan chunk), rumusnya pendek, dan kita butuh dua hal
    yang tidak disediakan library: tokenisasi khusus Bahasa Indonesia dan
    perhitungan *IDF-weighted coverage* untuk gate relevansi. Menghindari
    satu dependency lagi juga menjaga deployment tetap ringan.

Index dibangun di memori saat startup langsung dari `index/metadata.json`,
jadi TIDAK ada artefak tambahan yang perlu di-commit dan tidak ada biaya
API sama sekali.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from app.text_utils import content_tokens, tokenize

#: Parameter standar Okapi BM25.
#: k1 mengatur seberapa cepat term frequency jenuh; b mengatur kekuatan
#: normalisasi panjang dokumen.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass(frozen=True)
class LexicalHit:
    """Satu hasil pencarian BM25."""

    doc_id: int
    score: float
    matched_terms: tuple[str, ...]
    coverage: float


class LexicalIndex:
    """
    Index BM25 Okapi in-memory.

    Args:
        documents: Teks tiap dokumen, indeksnya dipakai sebagai ``doc_id``.
        k1: Parameter saturasi term frequency.
        b: Parameter normalisasi panjang dokumen.
    """

    def __init__(
        self,
        documents: list[str],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self.k1 = k1
        self.b = b

        self.doc_freqs: list[Counter[str]] = [Counter(tokenize(doc)) for doc in documents]
        self.doc_lengths: list[int] = [sum(freqs.values()) for freqs in self.doc_freqs]
        self.num_docs: int = len(documents)
        self.avg_doc_length: float = (
            sum(self.doc_lengths) / self.num_docs if self.num_docs else 0.0
        )

        # Document frequency tiap term.
        self.doc_frequency: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            self.doc_frequency.update(freqs.keys())

        # Peta term → daftar doc_id, supaya scoring hanya menyentuh dokumen
        # yang benar-benar memuat salah satu term query.
        self.postings: dict[str, list[int]] = {}
        for doc_id, freqs in enumerate(self.doc_freqs):
            for term in freqs:
                self.postings.setdefault(term, []).append(doc_id)

        self._idf_cache: dict[str, float] = {}

        # IDF untuk term yang tidak ada di korpus (df = 0). Nilainya adalah
        # batas atas IDF, sehingga istilah asing menekan skor coverage
        # dengan keras — inilah yang membuat pertanyaan di luar cakupan
        # tetap tertolak.
        self.max_idf: float = self._compute_idf(0)

    @classmethod
    def from_metadata(cls, metadata: list[dict], **kwargs) -> "LexicalIndex":
        """
        Bangun index dari `metadata.json`.

        Teks yang diindeks menyertakan heading path bila tersedia, supaya
        query seperti "BAB IV kurikulum" bisa cocok lewat judul bagian, bukan
        hanya isi paragraf.
        """
        documents = [cls.indexable_text(entry) for entry in metadata]
        return cls(documents, **kwargs)

    @staticmethod
    def indexable_text(entry: dict) -> str:
        """Gabungkan heading path/section title dengan isi chunk."""
        heading_path = entry.get("heading_path") or []
        if heading_path:
            heading = " > ".join(heading_path)
        else:
            heading = entry.get("section_title", "")
        text = entry.get("text", "")
        return f"{heading}\n{text}" if heading else text

    # ──────────────────────────────────────────
    # IDF
    # ──────────────────────────────────────────

    def _compute_idf(self, doc_frequency: int) -> float:
        """
        IDF varian Lucene/BM25+: ``log(1 + (N - df + 0.5) / (df + 0.5))``.

        Selalu positif, jadi term yang muncul di lebih dari separuh dokumen
        tidak pernah memberi kontribusi negatif seperti pada rumus Okapi asli.
        """
        n = self.num_docs
        return math.log(1.0 + (n - doc_frequency + 0.5) / (doc_frequency + 0.5))

    def idf(self, term: str) -> float:
        """IDF sebuah term. Term di luar korpus mendapat IDF maksimum."""
        cached = self._idf_cache.get(term)
        if cached is None:
            cached = self._compute_idf(self.doc_frequency.get(term, 0))
            self._idf_cache[term] = cached
        return cached

    # ──────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────

    def score(self, query_tokens: list[str], doc_id: int) -> float:
        """Skor BM25 satu dokumen terhadap token query."""
        freqs = self.doc_freqs[doc_id]
        doc_length = self.doc_lengths[doc_id]
        denominator_length = self.k1 * (
            1 - self.b + self.b * (doc_length / self.avg_doc_length if self.avg_doc_length else 0)
        )

        total = 0.0
        for term in query_tokens:
            tf = freqs.get(term, 0)
            if not tf:
                continue
            total += self.idf(term) * (tf * (self.k1 + 1)) / (tf + denominator_length)
        return total

    def coverage(self, query_content_tokens: list[str], doc_id: int) -> float:
        """
        Porsi bobot IDF token query yang benar-benar muncul di dokumen.

        Nilainya 0.0–1.0 dan berperan sebagai gate relevansi leksikal.
        Berbeda dari skor BM25 yang tak punya batas atas, angka ini bisa
        dibandingkan dengan threshold tetap.

        Contoh: untuk query "cara memasak nasi goreng", token
        memasak/nasi/goreng tidak ada di korpus sehingga masing-masing
        menyumbang IDF maksimum ke penyebut tapi nol ke pembilang —
        coverage jatuh ke sekitar 0,1 dan gate menolak.
        """
        if not query_content_tokens:
            return 0.0

        freqs = self.doc_freqs[doc_id]
        matched_weight = 0.0
        total_weight = 0.0

        for term in query_content_tokens:
            weight = self.idf(term)
            total_weight += weight
            if term in freqs:
                matched_weight += weight

        return matched_weight / total_weight if total_weight else 0.0

    def search(self, query: str, top_n: int = 20) -> list[LexicalHit]:
        """
        Cari dokumen paling relevan menurut BM25.

        Returns:
            Daftar `LexicalHit` terurut dari skor tertinggi, panjangnya
            maksimal ``top_n``. Dokumen tanpa satu pun term yang cocok
            tidak disertakan.
        """
        query_tokens = tokenize(query)
        if not query_tokens or not self.num_docs:
            return []

        unique_tokens = content_tokens(query)

        # Hanya nilai dokumen yang punya minimal satu term cocok.
        candidate_ids: set[int] = set()
        for term in unique_tokens:
            candidate_ids.update(self.postings.get(term, ()))

        hits: list[LexicalHit] = []
        for doc_id in candidate_ids:
            score = self.score(query_tokens, doc_id)
            if score <= 0:
                continue
            freqs = self.doc_freqs[doc_id]
            matched = tuple(t for t in unique_tokens if t in freqs)
            hits.append(LexicalHit(
                doc_id=doc_id,
                score=score,
                matched_terms=matched,
                coverage=self.coverage(unique_tokens, doc_id),
            ))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_n]

    @property
    def vocabulary_size(self) -> int:
        """Jumlah term unik dalam korpus."""
        return len(self.doc_frequency)

    def stats(self) -> dict:
        """Ringkasan index untuk ditampilkan di UI."""
        return {
            "documents": self.num_docs,
            "vocabulary": self.vocabulary_size,
            "avg_tokens_per_doc": round(self.avg_doc_length, 1),
            "k1": self.k1,
            "b": self.b,
        }
