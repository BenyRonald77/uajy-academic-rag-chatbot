"""Test circuit breaker public untuk respons provider yang tidak valid."""

from app.answer_guard import has_page_citation, is_safe_answer


class TestAnswerGuard:
    def test_jawaban_dengan_sitasi_lolos(self):
        answer = "Maksimal cuti adalah 2 semester. 📄 Sumber: Halaman 40 — H. Cuti Studi"
        assert has_page_citation(answer) is True
        assert is_safe_answer(answer) is True

    def test_sitasi_huruf_kecil_lolos(self):
        assert has_page_citation("📄 sumber: halaman 40") is True

    def test_kalimat_motivasi_provider_diblokir(self):
        assert is_safe_answer("Success starts with a single decision.") is False

    def test_pesan_error_diblokir(self):
        assert is_safe_answer("⚠️ Batas penggunaan API tercapai.") is False
        assert is_safe_answer("404 NOT_FOUND model") is False

    def test_jawaban_kosong_diblokir(self):
        assert is_safe_answer("") is False
        assert is_safe_answer("   ") is False

    def test_jawaban_tanpa_sitasi_diblokir(self):
        assert is_safe_answer("Cuti studi maksimal dua semester.") is False
