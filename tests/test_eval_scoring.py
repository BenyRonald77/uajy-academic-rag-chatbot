"""
test_eval_scoring.py — Penilaian dan agregasi metrik evaluasi.

Test di sini menjaga bahwa metriknya tetap **ketat**. Versi awal suite
evaluasi menganggap sebuah query berhasil bila salah satu kata kunci muncul
di mana pun pada teks yang terambil, sehingga hampir tidak mungkin gagal dan
angka 100% yang dihasilkan tidak bermakna. Kelonggaran itu tidak boleh
kembali diam-diam.
"""

from __future__ import annotations

import pytest

from app.retrieval import RetrievalCandidate, RetrievalOutcome
from eval.run_eval import QuestionResult, aggregate, score_question


def outcome_with(contexts, **kwargs) -> RetrievalOutcome:
    return RetrievalOutcome(
        original_query=kwargs.pop("query", "pertanyaan"),
        effective_query=kwargs.pop("effective_query", "pertanyaan"),
        contexts=contexts,
        candidates=kwargs.pop("candidates", contexts),
        **kwargs,
    )


def candidate(chunk_index: int, pages: list[int], text: str = "", dense: float = 0.8):
    return RetrievalCandidate(
        chunk_index=chunk_index, text=text, page_numbers=pages, dense_score=dense
    )


class TestPageRelevance:
    """
    Relevansi diukur dari irisan halaman, bukan pencocokan kata.

    Halaman yang memuat jawaban sudah diverifikasi manual, sehingga ukuran
    ini objektif dan tidak bisa dicurangi oleh kata yang umum.
    """

    def test_halaman_tepat_terhitung_kena(self):
        question = {"id": 1, "question": "q", "category": "in_scope",
                    "expected_pages": [48], "expected_answer_contains": []}
        result = score_question(question, outcome_with([candidate(0, [48])]), 100.0)
        assert result.page_hit is True
        assert result.first_relevant_rank == 1
        assert result.reciprocal_rank == pytest.approx(1.0)

    def test_halaman_salah_terhitung_luput(self):
        question = {"id": 1, "question": "q", "category": "in_scope",
                    "expected_pages": [48], "expected_answer_contains": []}
        result = score_question(question, outcome_with([candidate(0, [12])]), 100.0)
        assert result.page_hit is False
        assert result.reciprocal_rank == 0.0

    def test_peringkat_relevan_pertama_dicatat(self):
        """MRR harus membedakan peringkat 1 dari peringkat 3."""
        question = {"id": 1, "question": "q", "category": "in_scope",
                    "expected_pages": [48], "expected_answer_contains": []}
        contexts = [candidate(0, [12]), candidate(1, [27]), candidate(2, [48])]
        result = score_question(question, outcome_with(contexts), 100.0)
        assert result.first_relevant_rank == 3
        assert result.reciprocal_rank == pytest.approx(1 / 3)

    def test_irisan_sebagian_terhitung_kena(self):
        """Chunk yang membentang beberapa halaman cukup beririsan satu."""
        question = {"id": 1, "question": "q", "category": "in_scope",
                    "expected_pages": [40], "expected_answer_contains": []}
        result = score_question(question, outcome_with([candidate(0, [39, 40, 41])]), 100.0)
        assert result.page_hit is True

    def test_halaman_terambil_dikumpulkan_tanpa_duplikat(self):
        question = {"id": 1, "question": "q", "category": "in_scope",
                    "expected_pages": [48], "expected_answer_contains": []}
        contexts = [candidate(0, [48, 49]), candidate(1, [49, 50])]
        result = score_question(question, outcome_with(contexts), 100.0)
        assert result.retrieved_pages == [48, 49, 50]


class TestKeywordStrictness:
    def test_semua_kata_kunci_wajib_ada(self):
        question = {"id": 1, "question": "q", "category": "in_scope",
                    "expected_pages": [48],
                    "expected_answer_contains": ["cum laude", "3,51"]}
        teks = "Predikat CUM LAUDE berlaku bagi lulusan dengan IPK 3,51 sampai 4,00."
        result = score_question(question, outcome_with([candidate(0, [48], teks)]), 100.0)
        assert result.keyword_hit is True
        assert result.missing_keywords == []

    def test_kata_kunci_sebagian_dianggap_gagal(self):
        """
        Inilah yang membedakan suite ini dari versi lama.

        Metrik `any()` yang lama akan meloloskan kasus ini.
        """
        question = {"id": 1, "question": "q", "category": "in_scope",
                    "expected_pages": [48],
                    "expected_answer_contains": ["cum laude", "3,51"]}
        teks = "Predikat CUM LAUDE ditetapkan bagi lulusan terbaik."
        result = score_question(question, outcome_with([candidate(0, [48], teks)]), 100.0)
        assert result.keyword_hit is False
        assert result.missing_keywords == ["3,51"]

    def test_pencocokan_tidak_peka_huruf_besar(self):
        question = {"id": 1, "question": "q", "category": "in_scope",
                    "expected_pages": [48], "expected_answer_contains": ["CUM LAUDE"]}
        result = score_question(
            question, outcome_with([candidate(0, [48], "predikat cum laude")]), 100.0
        )
        assert result.keyword_hit is True

    def test_tanpa_kata_kunci_tidak_dihitung_lolos(self):
        question = {"id": 1, "question": "q", "category": "in_scope",
                    "expected_pages": [48], "expected_answer_contains": []}
        result = score_question(question, outcome_with([candidate(0, [48], "apa pun")]), 100.0)
        assert result.keyword_hit is False


class TestRefusalScoring:
    def test_penolakan_dicatat(self):
        question = {"id": 17, "question": "Apa rumus pythagoras?",
                    "category": "out_of_scope",
                    "expected_pages": [], "expected_answer_contains": []}
        outcome = outcome_with([], refused=True, refusal_reason="di luar cakupan")
        result = score_question(question, outcome, 50.0)
        assert result.refused is True
        assert result.refusal_reason == "di luar cakupan"

    def test_kegagalan_rerank_diteruskan(self):
        question = {"id": 1, "question": "q", "category": "in_scope",
                    "expected_pages": [48], "expected_answer_contains": []}
        outcome = outcome_with([candidate(0, [48])], rerank_failed=True)
        assert score_question(question, outcome, 100.0).rerank_failed is True


class TestConversational:
    def test_pertanyaan_dengan_riwayat_ditandai(self):
        question = {"id": 21, "question": "Berapa maksimalnya?", "category": "in_scope",
                    "expected_pages": [40], "expected_answer_contains": [],
                    "chat_history": [{"role": "user", "content": "cuti"}]}
        result = score_question(question, outcome_with([candidate(0, [40])]), 100.0)
        assert result.is_conversational is True

    def test_penulisan_ulang_dicatat(self):
        question = {"id": 21, "question": "Berapa maksimalnya?", "category": "in_scope",
                    "expected_pages": [40], "expected_answer_contains": [],
                    "chat_history": [{"role": "user", "content": "cuti"}]}
        outcome = outcome_with(
            [candidate(0, [40])],
            effective_query="Berapa maksimal cuti studi?",
            was_rewritten=True,
        )
        result = score_question(question, outcome, 100.0)
        assert result.was_rewritten is True
        assert result.effective_query == "Berapa maksimal cuti studi?"


class TestAggregate:
    @staticmethod
    def _result(**kwargs) -> QuestionResult:
        base = dict(id=1, question="q", category="in_scope", expected_pages=[1],
                    page_hit=True, first_relevant_rank=1,
                    expected_keywords=["x"], keyword_hit=True, latency_ms=100.0)
        base.update(kwargs)
        return QuestionResult(**base)

    def test_page_recall(self):
        metrics = aggregate([
            self._result(id=1, page_hit=True, first_relevant_rank=1),
            self._result(id=2, page_hit=False, first_relevant_rank=None),
        ])
        assert metrics["page_recall"] == pytest.approx(0.5)
        assert metrics["page_recall_hits"] == 1
        assert metrics["page_recall_total"] == 2

    def test_mrr_merata_ratakan_reciprocal_rank(self):
        metrics = aggregate([
            self._result(id=1, first_relevant_rank=1),
            self._result(id=2, first_relevant_rank=2),
        ])
        assert metrics["mrr"] == pytest.approx((1.0 + 0.5) / 2)

    def test_penolakan_salah_dihitung(self):
        """
        Tanpa metrik ini, sistem bisa memoles angka penolakan dengan cara
        menolak segalanya.
        """
        metrics = aggregate([
            self._result(id=1, refused=True),
            self._result(id=2, refused=False),
        ])
        assert metrics["false_refusal_rate"] == pytest.approx(0.5)
        assert metrics["false_refusals"] == 1

    def test_akurasi_penolakan_hanya_dari_out_of_scope(self):
        metrics = aggregate([
            self._result(id=1, category="in_scope", refused=False),
            self._result(id=2, category="out_of_scope", expected_pages=[],
                         expected_keywords=[], refused=True),
            self._result(id=3, category="out_of_scope", expected_pages=[],
                         expected_keywords=[], refused=False),
        ])
        assert metrics["refusal_accuracy"] == pytest.approx(0.5)
        assert metrics["out_of_scope_total"] == 2

    def test_kegagalan_rerank_dihitung(self):
        metrics = aggregate([
            self._result(id=1, rerank_failed=True),
            self._result(id=2, rerank_failed=False),
        ])
        assert metrics["rerank_failures"] == 1

    def test_recall_percakapan_terpisah(self):
        metrics = aggregate([
            self._result(id=1, is_conversational=False, page_hit=True),
            self._result(id=2, is_conversational=True, page_hit=True),
            self._result(id=3, is_conversational=True, page_hit=False,
                         first_relevant_rank=None),
        ])
        assert metrics["conversational_total"] == 2
        assert metrics["conversational_recall"] == pytest.approx(0.5)

    def test_daftar_kosong_tidak_membagi_nol(self):
        metrics = aggregate([])
        assert metrics["page_recall"] == 0.0
        assert metrics["mrr"] == 0.0
        assert metrics["avg_latency_ms"] == 0.0

    def test_statistik_latensi(self):
        metrics = aggregate([
            self._result(id=1, latency_ms=100.0),
            self._result(id=2, latency_ms=200.0),
            self._result(id=3, latency_ms=900.0),
        ])
        assert metrics["avg_latency_ms"] == pytest.approx(400.0)
        assert metrics["median_latency_ms"] == pytest.approx(200.0)
