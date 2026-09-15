"""
test_answer_eval.py — Bagian deterministik dari evaluasi jawaban.

Ekstraksi sitasi dan deteksi penolakan sengaja dibuat berbasis aturan, bukan
LLM, supaya bisa diuji seperti ini. Hanya groundedness yang memerlukan penilai
model — dan parsing balasannya pun tetap diuji di sini.
"""

from __future__ import annotations

import pytest

from eval.answer_eval import (
    GROUNDEDNESS_PASS_SCORE,
    REFUSAL_MARKERS,
    AnswerResult,
    aggregate_answer_metrics,
    build_groundedness_prompt,
    check_citations,
    detect_refusal,
    extract_cited_pages,
    judge_groundedness,
    parse_groundedness_response,
)


class TestDetectRefusal:
    @pytest.mark.parametrize("answer", [
        "Mohon maaf, informasi mengenai hal tersebut tidak ditemukan dalam dokumen yang tersedia.",
        "Maaf, saya hanya dapat menjawab pertanyaan seputar dokumen akademik UAJY.",
        "Pertanyaan tersebut di luar cakupan dokumen pedoman akademik.",
        "Silakan hubungi bagian akademik kampus untuk informasi lebih lanjut.",
        "Informasi tersebut tidak terdapat dalam dokumen yang tersedia.",
    ])
    def test_penolakan_dikenali(self, answer):
        assert detect_refusal(answer) is True

    @pytest.mark.parametrize("answer", [
        "IPK minimum untuk predikat Cum Laude adalah 3,51. 📄 Sumber: Halaman 48",
        "Maksimal cuti studi adalah 2 semester. 📄 Sumber: Halaman 39-41",
        "Beban belajar program Sarjana paling sedikit 144 SKS.",
    ])
    def test_jawaban_sungguhan_tidak_dianggap_menolak(self, answer):
        assert detect_refusal(answer) is False

    def test_kata_maaf_tunggal_tidak_memicu_penolakan(self):
        """
        Penanda penolakan harus berupa frasa, bukan kata tunggal.

        Jawaban yang sah bisa memuat "maaf" sebagai kesopanan, dan menandainya
        sebagai penolakan akan mengarang angka penolakan salah.
        """
        answer = ("Maaf atas keterlambatan informasi ini. Maksimal cuti studi "
                  "adalah 2 semester. 📄 Sumber: Halaman 40")
        assert detect_refusal(answer) is False

    def test_teks_kosong(self):
        assert detect_refusal("") is False

    def test_semua_penanda_berupa_frasa(self):
        """Penanda satu kata terlalu mudah memicu positif palsu."""
        for marker in REFUSAL_MARKERS:
            assert len(marker.split()) >= 2, f"penanda terlalu pendek: {marker!r}"


class TestExtractCitedPages:
    def test_halaman_tunggal(self):
        assert extract_cited_pages("📄 Sumber: Halaman 48 — Yudisium") == [48]

    def test_rentang_halaman(self):
        assert extract_cited_pages("📄 Sumber: Halaman 39-41") == [39, 40, 41]

    def test_rentang_dengan_en_dash(self):
        assert extract_cited_pages("Halaman 39–41") == [39, 40, 41]

    def test_daftar_halaman(self):
        assert extract_cited_pages("Halaman 39, 40 dan 41") == [39, 40, 41]

    def test_beberapa_sitasi(self):
        answer = "Lihat Halaman 27 dan juga halaman 48 untuk rinciannya."
        assert extract_cited_pages(answer) == [27, 48]

    def test_tidak_peka_huruf_besar(self):
        assert extract_cited_pages("HALAMAN 12") == [12]

    def test_tanpa_sitasi(self):
        assert extract_cited_pages("Jawaban tanpa rujukan halaman apa pun.") == []

    def test_teks_kosong(self):
        assert extract_cited_pages("") == []

    def test_angka_lain_tidak_ikut_terambil(self):
        """Hanya angka yang mengikuti kata "halaman" yang merupakan sitasi."""
        answer = "Beban 144 SKS dengan IPK 3,51 selama 8 semester. Halaman 27."
        assert extract_cited_pages(answer) == [27]

    def test_rentang_tidak_wajar_diabaikan(self):
        """Rentang selebar ribuan halaman hampir pasti salah baca."""
        assert extract_cited_pages("Halaman 1-9999") == [1]

    def test_tanda_pisah_judul_bukan_penanda_rentang(self):
        """
        Regresi dari kasus nyata.

        Format sitasi yang diminta prompt adalah ``Halaman X — [Nama Bagian]``,
        dan nama bagian sering diawali angka. Tanda pisah panjang di situ
        memisahkan judul, bukan menandai rentang. Versi sebelumnya membaca
        "43 — 5" sebagai rentang lalu memungut angka 5 sebagai halaman,
        sehingga melaporkan sitasi palsu yang tidak pernah ditulis model.
        """
        answer = ("📄 Sumber: Halaman 25 — C. SISTEM PENYELENGGARAAN PENDIDIKAN; "
                  "Halaman 43 — 5. Beban studi dan beban kredit semester; "
                  "Halaman 42 — 5. Beban studi dan beban kredit semester")
        assert extract_cited_pages(answer) == [25, 42, 43]

    def test_rentang_menaik_tetap_dikenali(self):
        """Perbaikan di atas tidak boleh mematikan rentang yang sah."""
        assert extract_cited_pages("Halaman 39 — 41") == [39, 40, 41]

    def test_judul_berangka_setelah_halaman_tunggal(self):
        answer = "📄 Sumber: Halaman 48 — 3. Predikat Kelulusan"
        assert extract_cited_pages(answer) == [48]

    def test_hasil_terurut_tanpa_duplikat(self):
        answer = "Halaman 48, halaman 27, dan Halaman 48 lagi."
        assert extract_cited_pages(answer) == [27, 48]


class TestCheckCitations:
    def test_sitasi_valid(self, candidate_factory):
        contexts = [candidate_factory(page_numbers=[48])]
        check = check_citations("Jawaban. 📄 Sumber: Halaman 48", contexts)
        assert check.is_valid is True
        assert check.invalid_pages == []
        assert check.precision == pytest.approx(1.0)

    def test_mengutip_halaman_di_luar_konteks(self, candidate_factory):
        """
        Ini bentuk halusinasi yang paling berbahaya.

        Sitasi yang meyakinkan tetapi salah mendorong pembaca memercayainya,
        justru lebih merugikan daripada tidak ada sitasi sama sekali.
        """
        contexts = [candidate_factory(page_numbers=[48])]
        check = check_citations("Jawaban. 📄 Sumber: Halaman 99", contexts)
        assert check.is_valid is False
        assert check.invalid_pages == [99]
        assert check.precision == pytest.approx(0.0)

    def test_sebagian_valid(self, candidate_factory):
        contexts = [candidate_factory(page_numbers=[48])]
        check = check_citations("Halaman 48 dan Halaman 99", contexts)
        assert check.invalid_pages == [99]
        assert check.precision == pytest.approx(0.5)

    def test_tanpa_sitasi_dianggap_tidak_valid(self, candidate_factory):
        """Prompt sistem mewajibkan sitasi; tanpanya jawaban tak bisa diaudit."""
        contexts = [candidate_factory(page_numbers=[48])]
        check = check_citations("Jawaban tanpa rujukan.", contexts)
        assert check.has_citation is False
        assert check.is_valid is False

    def test_halaman_konteks_digabung_dari_semua_kandidat(self, candidate_factory):
        contexts = [
            candidate_factory(chunk_index=0, page_numbers=[48]),
            candidate_factory(chunk_index=1, page_numbers=[39, 40]),
        ]
        check = check_citations("Halaman 40", contexts)
        assert check.context_pages == [39, 40, 48]
        assert check.is_valid is True

    def test_konteks_kosong(self):
        check = check_citations("Halaman 48", [])
        assert check.invalid_pages == [48]


class TestParseGroundednessResponse:
    def test_json_bersih(self):
        verdict = parse_groundedness_response('{"score": 9, "unsupported_claims": []}')
        assert verdict.score == 9.0
        assert verdict.is_grounded is True
        assert verdict.judge_failed is False

    def test_dengan_pernyataan_tak_didukung(self):
        raw = '{"score": 4, "unsupported_claims": ["batas waktu 14 hari"]}'
        verdict = parse_groundedness_response(raw)
        assert verdict.score == 4.0
        assert verdict.is_grounded is False
        assert verdict.unsupported_claims == ["batas waktu 14 hari"]

    def test_code_fence(self):
        verdict = parse_groundedness_response('```json\n{"score": 8}\n```')
        assert verdict.score == 8.0

    def test_teks_tambahan(self):
        verdict = parse_groundedness_response('Penilaian: {"score": 7} selesai.')
        assert verdict.score == 7.0

    def test_skor_dibatasi(self):
        assert parse_groundedness_response('{"score": 42}').score == 10.0
        assert parse_groundedness_response('{"score": -5}').score == 0.0

    @pytest.mark.parametrize("raw", [
        "", "bukan json", "{}", '{"tanpa_skor": 1}', '{"score": "abc"}', "[]",
    ])
    def test_balasan_rusak_ditandai_gagal(self, raw):
        """
        Kegagalan penilai BUKAN skor nol.

        Membedakan keduanya penting: "dinilai tidak setia" dan "tidak pernah
        dinilai" adalah dua hal yang sangat berbeda bagi laporan akhir.
        """
        verdict = parse_groundedness_response(raw)
        assert verdict.judge_failed is True
        assert verdict.score is None
        assert verdict.is_grounded is False

    def test_ambang_kelulusan(self):
        assert parse_groundedness_response(
            f'{{"score": {GROUNDEDNESS_PASS_SCORE}}}'
        ).is_grounded is True
        assert parse_groundedness_response(
            f'{{"score": {GROUNDEDNESS_PASS_SCORE - 0.1}}}'
        ).is_grounded is False


class TestJudgeGroundedness:
    def test_konteks_kosong_dianggap_gagal(self):
        assert judge_groundedness("q", [], "jawaban").judge_failed is True

    def test_jawaban_kosong_dianggap_gagal(self, candidate_factory):
        assert judge_groundedness("q", [candidate_factory()], "").judge_failed is True

    def test_kegagalan_api_ditandai(self, candidate_factory, monkeypatch):
        monkeypatch.setattr(
            "eval.answer_eval.call_utility_llm",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("API mati")),
        )
        verdict = judge_groundedness("q", [candidate_factory()], "jawaban")
        assert verdict.judge_failed is True

    def test_putusan_diteruskan(self, candidate_factory, monkeypatch):
        monkeypatch.setattr(
            "eval.answer_eval.call_utility_llm",
            lambda *a, **k: '{"score": 10, "unsupported_claims": []}',
        )
        verdict = judge_groundedness("q", [candidate_factory()], "jawaban")
        assert verdict.score == 10.0
        assert verdict.is_grounded is True


class TestBuildGroundednessPrompt:
    def test_memuat_pertanyaan_konteks_dan_jawaban(self, candidate_factory):
        contexts = [candidate_factory(page_numbers=[48], text="isi dokumen penting")]
        prompt = build_groundedness_prompt("Berapa IPK?", contexts, "Jawabannya 3,51.")
        assert "Berapa IPK?" in prompt
        assert "isi dokumen penting" in prompt
        assert "Jawabannya 3,51." in prompt
        assert "Halaman 48" in prompt

    def test_konteks_panjang_dipotong(self, candidate_factory):
        contexts = [candidate_factory(text="kata " * 5000)]
        prompt = build_groundedness_prompt("q", contexts, "jawaban")
        assert len(prompt) < 4000


class TestAggregateAnswerMetrics:
    @staticmethod
    def _result(**kwargs) -> AnswerResult:
        base = dict(id=1, question="q", category="in_scope", answer="jawaban")
        base.update(kwargs)
        return AnswerResult(**base)

    def test_groundedness_hanya_dari_yang_dijawab(self):
        """Jawaban yang menolak tidak punya klaim faktual untuk dinilai."""
        metrics = aggregate_answer_metrics([
            self._result(id=1, groundedness_score=10.0, grounded=True),
            self._result(id=2, refused_in_answer=True),
        ])
        assert metrics["judged_total"] == 1
        assert metrics["groundedness_rate"] == pytest.approx(1.0)

    def test_penilai_gagal_tidak_ikut_dihitung(self):
        metrics = aggregate_answer_metrics([
            self._result(id=1, groundedness_score=10.0, grounded=True),
            self._result(id=2, judge_failed=True),
        ])
        assert metrics["judged_total"] == 1
        assert metrics["judge_failures"] == 1
        assert metrics["groundedness_rate"] == pytest.approx(1.0)

    def test_validitas_sitasi(self):
        metrics = aggregate_answer_metrics([
            self._result(id=1, has_citation=True, citation_valid=True, citation_precision=1.0),
            self._result(id=2, has_citation=True, citation_valid=False, citation_precision=0.5),
        ])
        assert metrics["citation_validity"] == pytest.approx(0.5)
        assert metrics["citation_presence"] == pytest.approx(1.0)
        assert metrics["avg_citation_precision"] == pytest.approx(0.75)

    def test_penolakan_end_to_end(self):
        metrics = aggregate_answer_metrics([
            self._result(id=1, category="out_of_scope", refused_in_answer=True),
            self._result(id=2, category="out_of_scope", refused_in_answer=False),
            self._result(id=3, category="in_scope", refused_in_answer=False),
        ])
        assert metrics["refusal_accuracy_e2e"] == pytest.approx(0.5)
        assert metrics["false_refusal_e2e"] == pytest.approx(0.0)

    def test_penolakan_salah_dihitung(self):
        metrics = aggregate_answer_metrics([
            self._result(id=1, category="in_scope", refused_in_answer=True),
            self._result(id=2, category="in_scope", refused_in_answer=False),
        ])
        assert metrics["false_refusal_e2e"] == pytest.approx(0.5)

    def test_penolakan_oleh_gate_retrieval_ikut_dihitung(self):
        metrics = aggregate_answer_metrics([
            self._result(id=1, category="out_of_scope", refused_by_retrieval=True),
        ])
        assert metrics["refusal_accuracy_e2e"] == pytest.approx(1.0)

    def test_daftar_kosong_tidak_membagi_nol(self):
        metrics = aggregate_answer_metrics([])
        assert metrics["groundedness_rate"] == 0.0
        assert metrics["citation_validity"] == 0.0
        assert metrics["refusal_accuracy_e2e"] == 0.0

    def test_kegagalan_generasi_dikeluarkan_dari_metrik(self):
        """
        Regresi dari kasus nyata.

        Pesan error API pernah masuk sebagai "jawaban". Karena pesan itu tidak
        memuat klaim faktual, penilai groundedness memberinya 10/10 dan
        metriknya tampak sempurna padahal jawabannya tidak pernah ada.
        """
        metrics = aggregate_answer_metrics([
            self._result(id=1, has_citation=True, citation_valid=True,
                         groundedness_score=10.0, grounded=True),
            self._result(id=2, generation_failed=True, groundedness_score=10.0,
                         grounded=True, has_citation=False),
        ])
        assert metrics["generation_failures"] == 1
        assert metrics["in_scope_total"] == 1, "yang gagal tidak boleh ikut dihitung"
        assert metrics["citation_presence"] == pytest.approx(1.0)
        assert metrics["judged_total"] == 1

    def test_tanpa_kegagalan_generasi(self):
        metrics = aggregate_answer_metrics([self._result(id=1)])
        assert metrics["generation_failures"] == 0
