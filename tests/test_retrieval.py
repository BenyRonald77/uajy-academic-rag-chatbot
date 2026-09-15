"""test_retrieval.py — Fusi RRF, gate relevansi, dan format sitasi."""

from __future__ import annotations

import pytest

from app.config import RetrievalConfig
from app.retrieval import (
    LEXICAL_RESCUE_DENSE_FLOOR_RATIO,
    LEXICAL_RESCUE_MIN_TERMS,
    HybridRetriever,
    RetrievalCandidate,
    RetrievalOutcome,
    format_sources,
    reciprocal_rank_fusion,
)


class TestReciprocalRankFusion:
    def test_dokumen_di_dua_jalur_diunggulkan(self):
        """Inti RRF: kesepakatan antar jalur lebih bernilai."""
        scores = reciprocal_rank_fusion(
            {"dense": [10, 20, 30], "lexical": [30, 40, 10]}, rrf_k=60
        )
        assert scores[10] > scores[20]
        assert scores[30] > scores[40]

    def test_peringkat_lebih_tinggi_skor_lebih_besar(self):
        scores = reciprocal_rank_fusion({"dense": [1, 2, 3]}, rrf_k=60)
        assert scores[1] > scores[2] > scores[3]

    def test_bobot_per_jalur_dihormati(self):
        seimbang = reciprocal_rank_fusion(
            {"dense": [1], "lexical": [2]}, rrf_k=60
        )
        assert seimbang[1] == pytest.approx(seimbang[2])

        berat_dense = reciprocal_rank_fusion(
            {"dense": [1], "lexical": [2]},
            weights={"dense": 2.0, "lexical": 1.0},
            rrf_k=60,
        )
        assert berat_dense[1] > berat_dense[2]

    def test_rrf_k_meredam_selisih(self):
        """k besar membuat bobot antar peringkat makin datar."""
        kecil = reciprocal_rank_fusion({"a": [1, 2]}, rrf_k=1)
        besar = reciprocal_rank_fusion({"a": [1, 2]}, rrf_k=1000)
        assert (kecil[1] / kecil[2]) > (besar[1] / besar[2])

    def test_daftar_kosong(self):
        assert reciprocal_rank_fusion({}) == {}
        assert reciprocal_rank_fusion({"dense": []}) == {}

    def test_tidak_bergantung_skor_mentah(self):
        """
        RRF hanya memakai peringkat, sehingga cosine (0-1) dan BM25 (tak
        terbatas) bisa disatukan tanpa normalisasi apa pun.
        """
        a = reciprocal_rank_fusion({"dense": [7, 8]}, rrf_k=60)
        b = reciprocal_rank_fusion({"lexical": [7, 8]}, rrf_k=60)
        assert a == b


class TestRetrievalCandidate:
    def test_page_label_satu_halaman(self):
        assert RetrievalCandidate(chunk_index=0, text="x", page_numbers=[48]).page_label == "48"

    def test_page_label_rentang(self):
        c = RetrievalCandidate(chunk_index=0, text="x", page_numbers=[39, 40, 41])
        assert c.page_label == "39-41"

    def test_heading_label_mengutamakan_heading_path(self):
        c = RetrievalCandidate(
            chunk_index=0, text="x", page_numbers=[1],
            section_title="diabaikan", heading_path=["BAB I", "Pasal 2"],
        )
        assert c.heading_label == "BAB I › Pasal 2"

    def test_heading_label_jatuh_ke_section_title(self):
        c = RetrievalCandidate(
            chunk_index=0, text="x", page_numbers=[1], section_title="H. Cuti Studi"
        )
        assert c.heading_label == "H. Cuti Studi"

    @pytest.mark.parametrize("dense_rank,lexical_rank,expected", [
        (1, 1, "dense+bm25"),
        (1, None, "dense"),
        (None, 1, "bm25"),
        (None, None, "-"),
    ])
    def test_retrieved_by(self, dense_rank, lexical_rank, expected):
        c = RetrievalCandidate(
            chunk_index=0, text="x", page_numbers=[1],
            dense_rank=dense_rank, lexical_rank=lexical_rank,
        )
        assert c.retrieved_by == expected

    def test_display_score_mengutamakan_rerank(self):
        c = RetrievalCandidate(
            chunk_index=0, text="x", page_numbers=[1],
            dense_score=0.8, rerank_score=10.0,
        )
        assert c.display_score == pytest.approx(1.0)

    def test_display_score_jatuh_ke_dense(self):
        c = RetrievalCandidate(chunk_index=0, text="x", page_numbers=[1], dense_score=0.8)
        assert c.display_score == pytest.approx(0.8)

    def test_similarity_score_kompatibilitas_lama(self):
        c = RetrievalCandidate(chunk_index=0, text="x", page_numbers=[1], dense_score=0.77)
        assert c.similarity_score == 0.77


class TestRelevanceGate:
    """
    Gate lapis pertama. Diuji lewat metode langsung tanpa memuat FAISS,
    sebab yang dinilai murni logika keputusannya.
    """

    @staticmethod
    def _gate(candidates, config=None):
        retriever = HybridRetriever.__new__(HybridRetriever)
        return retriever._evaluate_gate(candidates, "pertanyaan", config or RetrievalConfig())

    def test_lolos_lewat_jalur_dense(self, candidate_factory):
        config = RetrievalConfig(dense_threshold=0.5)
        gate = self._gate([candidate_factory(dense_score=0.8)], config)
        assert gate["passed"] is True
        assert gate["dense_pass"] is True

    def test_gagal_saat_semua_di_bawah_ambang(self, candidate_factory):
        config = RetrievalConfig(dense_threshold=0.9)
        gate = self._gate([candidate_factory(dense_score=0.4)], config)
        assert gate["passed"] is False

    def test_penyelamatan_leksikal_butuh_tiga_syarat(self, candidate_factory):
        """
        Cakupan istilah tinggi saja tidak cukup. Tanpa syarat ganda,
        pertanyaan di luar cakupan bisa lolos hanya karena berbagi beberapa
        kata umum dengan dokumen.
        """
        config = RetrievalConfig(dense_threshold=0.9, lexical_coverage_threshold=0.6)
        floor = config.dense_threshold * LEXICAL_RESCUE_DENSE_FLOOR_RATIO

        cukup = candidate_factory(
            dense_score=floor + 0.01,
            lexical_coverage=0.9,
            matched_terms=("pasal", "12"),
        )
        assert self._gate([cukup], config)["lexical_pass"] is True

        # Cakupan tinggi tapi kemiripan semantik di bawah lantai.
        semantik_lemah = candidate_factory(
            dense_score=floor - 0.05,
            lexical_coverage=0.9,
            matched_terms=("pasal", "12"),
        )
        assert self._gate([semantik_lemah], config)["lexical_pass"] is False

        # Cakupan tinggi tapi hanya satu istilah yang cocok.
        satu_istilah = candidate_factory(
            dense_score=floor + 0.01,
            lexical_coverage=0.9,
            matched_terms=("pasal",),
        )
        assert self._gate([satu_istilah], config)["lexical_pass"] is False

    def test_kandidat_kosong(self):
        gate = self._gate([])
        assert gate["passed"] is False
        assert gate["max_dense_score"] == 0.0

    def test_gate_melaporkan_ambang_yang_dipakai(self, candidate_factory):
        config = RetrievalConfig(dense_threshold=0.65, lexical_coverage_threshold=0.55)
        gate = self._gate([candidate_factory(dense_score=0.7)], config)
        assert gate["dense_threshold"] == 0.65
        assert gate["lexical_coverage_threshold"] == 0.55

    def test_konstanta_penyelamatan_masuk_akal(self):
        assert 0 < LEXICAL_RESCUE_DENSE_FLOOR_RATIO <= 1
        assert LEXICAL_RESCUE_MIN_TERMS >= 2


class TestFormatSources:
    def test_kosong_untuk_tanpa_hasil(self):
        assert format_sources([]) == ""

    def test_memuat_halaman_dan_bagian(self, candidate_factory):
        c = candidate_factory(
            page_numbers=[48], dense_score=0.87,
            heading_path=["BAB IV", "M. Yudisium"],
        )
        hasil = format_sources([c])
        assert "48" in hasil
        assert "BAB IV › M. Yudisium" in hasil
        assert "87%" in hasil

    def test_sumber_duplikat_digabung(self, candidate_factory):
        a = candidate_factory(chunk_index=0, page_numbers=[48], section_title="Yudisium")
        b = candidate_factory(chunk_index=1, page_numbers=[48], section_title="Yudisium")
        assert len(format_sources([a, b]).splitlines()) == 1

    def test_halaman_berbeda_jadi_baris_berbeda(self, candidate_factory):
        a = candidate_factory(chunk_index=0, page_numbers=[48], section_title="A")
        b = candidate_factory(chunk_index=1, page_numbers=[27], section_title="B")
        assert len(format_sources([a, b]).splitlines()) == 2


class TestRetrievalOutcome:
    def test_total_ms_menjumlahkan_tahapan(self):
        outcome = RetrievalOutcome(
            original_query="q", effective_query="q",
            timings_ms={"dense_ms": 500.0, "lexical_ms": 2.0, "rerank_ms": 400.0},
        )
        assert outcome.total_ms == pytest.approx(902.0)

    def test_default_tidak_menolak(self):
        outcome = RetrievalOutcome(original_query="q", effective_query="q")
        assert outcome.refused is False
        assert outcome.rerank_failed is False


class TestRetrievalConfig:
    def test_with_overrides_menghasilkan_salinan(self):
        base = RetrievalConfig()
        diubah = base.with_overrides(top_k=8)
        assert diubah.top_k == 8
        assert base.top_k != 8, "config asli harus tetap utuh (frozen dataclass)"

    def test_nilai_none_diabaikan(self):
        base = RetrievalConfig(top_k=4)
        assert base.with_overrides(top_k=None).top_k == 4

    def test_field_tidak_dikenal_ditolak(self):
        with pytest.raises(ValueError, match="tidak dikenal"):
            RetrievalConfig().with_overrides(field_ngawur=1)
