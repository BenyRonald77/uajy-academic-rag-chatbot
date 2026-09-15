"""test_extract_text.py — Penyaringan noise hasil ekstraksi PDF."""

from __future__ import annotations

import pytest

from ingestion.extract_text import (
    ExtractionReport,
    _detect_header_footer_templates,
    _filter_lines,
    _header_footer_template,
    _is_gibberish,
    _is_page_number,
    _longest_consonant_run,
)


class TestPageNumberDetection:
    @pytest.mark.parametrize("line", [
        "42", "1", "Halaman 12", "halaman 3", "Page 7", "hal. 9",
        "- 5 -", "– 11 –", "xii", "IV",
    ])
    def test_nomor_halaman_dikenali(self, line):
        assert _is_page_number(line)

    @pytest.mark.parametrize("line", [
        "Pasal 12",
        "144 SKS",
        "1 85 - 100 A 4,00",
        "Mahasiswa wajib herregistrasi",
    ])
    def test_isi_sungguhan_tidak_dianggap_nomor_halaman(self, line):
        assert not _is_page_number(line)


class TestConsonantRun:
    def test_digraf_dihitung_satu_konsonan(self):
        """"ng" dan "ny" secara fonetis satu konsonan, bukan gugus tak wajar."""
        assert _longest_consonant_run("mengangguk") < 3

    @pytest.mark.parametrize("word", [
        "mahasiswa", "yudisium", "kurikulum", "semester",
        # Regresi: gugus "str" sah dalam Bahasa Indonesia. Sebelum gugus ini
        # dikenali, kosakata yang justru paling sering dipakai dokumen ini
        # dinilai tak wajar dan berisiko terbuang sebagai teks rusak.
        "herregistrasi", "administrasi", "struktur", "instruksi", "pendaftaran",
    ])
    def test_kata_indonesia_wajar(self, word):
        assert _longest_consonant_run(word) < 3, (
            f"kata sah {word!r} dinilai tak wajar"
        )

    def test_teks_rusak_punya_gugus_panjang(self):
        assert _longest_consonant_run("lratsrutkurts") >= 3


class TestGibberishDetection:
    def test_huruf_bertumpuk(self):
        assert _is_gibberish("UUUNNNIIIVVVEEERRRSSS AAATTTMMMAAA JJJAAAYYYAAA")

    def test_kata_pendek_tidak_dinilai(self):
        """Terlalu sedikit kata panjang untuk disimpulkan apa pun."""
        assert not _is_gibberish("A B C")

    def test_singkatan_kapital_dikecualikan(self):
        """Regresi: kode program studi pernah membuat tabel kurikulum terbuang."""
        assert not _is_gibberish("TID TIF MKWU SKS IPK KRS")


class TestHeaderFooterTemplate:
    def test_angka_dinormalkan(self):
        a = _header_footer_template("Susunan Pengurus xi")
        b = _header_footer_template("Susunan Pengurus xiii")
        assert a == b, "varian nomor halaman harus mengerucut ke pola yang sama"

    def test_angka_arab_dinormalkan(self):
        assert _header_footer_template("12 Buku Pedoman UAJY") == \
               _header_footer_template("98 Buku Pedoman UAJY")

    def test_deteksi_berbasis_frekuensi(self):
        """
        Header berjalan dikenali dari frekuensinya antar halaman, bukan dari
        daftar pola yang ditulis manual — jadi berlaku untuk dokumen apa pun.
        """
        unik = [
            "Ketentuan cuti studi", "Prosedur herregistrasi", "Predikat kelulusan",
            "Beban studi mahasiswa", "Evaluasi hasil belajar", "Program remedi",
            "Skema beasiswa", "Pindah program studi", "Ujian pendadaran",
            "Sistem kredit semester", "Wisuda sarjana", "Layanan perpustakaan",
        ]
        pages = [
            (i, [f"{i} Buku Pedoman UAJY", judul, "Penutup"])
            for i, judul in enumerate(unik, start=1)
        ]
        templates = _detect_header_footer_templates(pages)

        # Muncul di setiap halaman -> header berjalan.
        assert _header_footer_template("1 Buku Pedoman UAJY") in templates
        # Hanya muncul sekali -> isi sungguhan.
        assert _header_footer_template("Ketentuan cuti studi") not in templates

    def test_dokumen_terlalu_pendek_tidak_dideteksi(self):
        pages = [(1, ["Judul", "Isi"]), (2, ["Judul", "Isi"])]
        assert _detect_header_footer_templates(pages) == set()


class TestFilterLines:
    def test_kode_glyph_dibuang(self):
        report = ExtractionReport()
        hasil = _filter_lines(["(cid:54)(cid:68)(cid:87) teks rusak"], set(), report)
        assert hasil == []
        assert report.dropped["kode glyph (cid:NN)"] == 1

    def test_baris_terlalu_pendek_dibuang(self):
        report = ExtractionReport()
        hasil = _filter_lines(["g", ".", "B", "e"], set(), report)
        assert hasil == []
        assert report.dropped["baris terlalu pendek"] == 4

    def test_duplikat_berurutan_dibuang(self):
        report = ExtractionReport()
        hasil = _filter_lines(
            ["Mahasiswa aktif", "Mahasiswa aktif", "Baris lain"], set(), report
        )
        assert hasil == ["Mahasiswa aktif", "Baris lain"]
        assert report.dropped["baris duplikat berurutan"] == 1

    def test_isi_sah_dipertahankan(self):
        report = ExtractionReport()
        lines = [
            "Mahasiswa wajib melakukan herregistrasi setiap awal semester.",
            "TID 1101 Kalkulus I 3 SKS",
            "1 85 - 100 A 4,00",
        ]
        assert _filter_lines(lines, set(), report) == lines

    def test_whitespace_dinormalkan(self):
        report = ExtractionReport()
        hasil = _filter_lines(["Mahasiswa     wajib    hadir"], set(), report)
        assert hasil == ["Mahasiswa wajib hadir"]


class TestExtractionReport:
    def test_hitungan_baris_dibuang(self):
        report = ExtractionReport(lines_raw=100, lines_kept=76)
        assert report.lines_dropped == 24

    def test_ringkasan_menangani_laporan_kosong(self):
        assert ExtractionReport().summary_lines()

    def test_ringkasan_memuat_persentase(self):
        report = ExtractionReport(lines_raw=100, lines_kept=76)
        report.dropped["nomor halaman"] = 24
        teks = " ".join(report.summary_lines())
        assert "24" in teks
