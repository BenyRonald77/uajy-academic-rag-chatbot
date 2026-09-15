"""test_query_rewriter.py — Heuristik dan pembersihan hasil penulisan ulang."""

from __future__ import annotations

import pytest

from app.query_rewriter import (
    _sanitize,
    build_rewrite_prompt,
    needs_rewrite,
    rewrite_query,
)


@pytest.fixture
def history() -> list[dict]:
    return [
        {"role": "user", "content": "Bagaimana ketentuan cuti studi di UAJY?"},
        {"role": "assistant", "content": "Cuti studi dapat diajukan mahasiswa yang "
                                         "telah mengikuti kegiatan akademik minimal "
                                         "2 semester."},
    ]


class TestNeedsRewrite:
    def test_tanpa_riwayat_tidak_perlu(self):
        assert needs_rewrite("Berapa maksimalnya?", []) is False
        assert needs_rewrite("Berapa maksimalnya?", None) is False

    def test_riwayat_tanpa_jawaban_asisten_tidak_perlu(self):
        """Belum ada apa pun yang bisa dirujuk."""
        hanya_user = [{"role": "user", "content": "Halo"}]
        assert needs_rewrite("Berapa maksimalnya?", hanya_user) is False

    @pytest.mark.parametrize("query", [
        "Berapa maksimalnya?",
        "Apa itu?",
        "Berapa lama maksimal?",
        "Bagaimana prosedurnya?",
    ])
    def test_pertanyaan_pendek_perlu_ditulis_ulang(self, query, history):
        assert needs_rewrite(query, history) is True

    @pytest.mark.parametrize("query", [
        "Bagaimana dengan biayanya?",
        "Apa syarat untuk hal tersebut?",
        "Kalau untuk program magister bagaimana ketentuan yang berlaku?",
    ])
    def test_kata_rujukan_perlu_ditulis_ulang(self, query, history):
        assert needs_rewrite(query, history) is True

    @pytest.mark.parametrize("query", [
        "Berapa IPK minimum untuk lulus cum laude?",
        "Apa saja ketentuan herregistrasi bagi mahasiswa baru?",
        "Berapa total SKS minimal untuk lulus program sarjana?",
    ])
    def test_pertanyaan_mandiri_tidak_perlu(self, query, history):
        """
        Menulis ulang setiap pertanyaan berarti satu panggilan API tambahan
        per query, padahal sebagian besar sudah mandiri.
        """
        assert needs_rewrite(query, history) is False

    def test_query_kosong(self, history):
        assert needs_rewrite("", history) is False
        assert needs_rewrite("   ", history) is False


class TestBuildRewritePrompt:
    def test_memuat_riwayat_dan_pertanyaan(self, history):
        prompt = build_rewrite_prompt("Berapa maksimalnya?", history)
        assert "cuti studi" in prompt
        assert "Berapa maksimalnya?" in prompt

    def test_peran_diberi_label_indonesia(self, history):
        prompt = build_rewrite_prompt("q", history)
        assert "Pengguna:" in prompt
        assert "Asisten:" in prompt

    def test_pesan_panjang_dipotong(self):
        panjang = [
            {"role": "user", "content": "x" * 5000},
            {"role": "assistant", "content": "y" * 5000},
        ]
        prompt = build_rewrite_prompt("q", panjang)
        assert len(prompt) < 2000

    def test_hanya_jendela_terakhir_dipakai(self):
        banyak = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"pesan-{i}"}
            for i in range(20)
        ]
        prompt = build_rewrite_prompt("q", banyak)
        assert "pesan-19" in prompt
        assert "pesan-0" not in prompt


class TestSanitize:
    def test_teks_bersih_diteruskan(self):
        assert _sanitize("Berapa maksimal cuti studi?", "asli") == "Berapa maksimal cuti studi?"

    def test_tanda_kutip_dibuang(self):
        assert _sanitize('"Berapa maksimal cuti?"', "asli") == "Berapa maksimal cuti?"

    def test_awalan_dibuang(self):
        assert _sanitize("Pertanyaan mandiri: Berapa maksimal cuti?", "asli") == \
               "Berapa maksimal cuti?"

    def test_hanya_baris_pertama_dipakai(self):
        hasil = _sanitize("Berapa maksimal cuti?\nPenjelasan tambahan.", "asli")
        assert hasil == "Berapa maksimal cuti?"

    def test_balasan_kosong_jatuh_ke_asli(self):
        assert _sanitize("", "query asli") == "query asli"
        assert _sanitize("   ", "query asli") == "query asli"

    def test_balasan_terlalu_panjang_ditolak(self):
        """Model yang malah menjawab pertanyaannya harus diabaikan."""
        assert _sanitize("penjelasan " * 200, "query asli") == "query asli"


class TestRewriteQuery:
    def test_dilewati_saat_heuristik_bilang_tidak_perlu(self, history, monkeypatch):
        def jangan_dipanggil(*args, **kwargs):
            raise AssertionError("LLM tidak boleh dipanggil")

        monkeypatch.setattr("app.query_rewriter.call_utility_llm", jangan_dipanggil)
        mandiri = "Berapa IPK minimum untuk lulus cum laude?"
        assert rewrite_query(mandiri, history) == mandiri

    def test_force_melewati_heuristik(self, history, monkeypatch):
        monkeypatch.setattr(
            "app.query_rewriter.call_utility_llm",
            lambda *a, **k: "Pertanyaan hasil tulis ulang?",
        )
        hasil = rewrite_query("Pertanyaan mandiri yang cukup panjang sekali?",
                              history, force=True)
        assert hasil == "Pertanyaan hasil tulis ulang?"

    def test_menulis_ulang_pertanyaan_lanjutan(self, history, monkeypatch):
        monkeypatch.setattr(
            "app.query_rewriter.call_utility_llm",
            lambda *a, **k: "Berapa maksimal cuti studi yang dapat diajukan?",
        )
        hasil = rewrite_query("Berapa maksimalnya?", history)
        assert hasil == "Berapa maksimal cuti studi yang dapat diajukan?"

    def test_kegagalan_api_jatuh_ke_query_asli(self, history, monkeypatch):
        monkeypatch.setattr(
            "app.query_rewriter.call_utility_llm",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("API mati")),
        )
        assert rewrite_query("Berapa maksimalnya?", history) == "Berapa maksimalnya?"

    def test_tanpa_riwayat_mengembalikan_apa_adanya(self):
        assert rewrite_query("Berapa maksimalnya?", None) == "Berapa maksimalnya?"
