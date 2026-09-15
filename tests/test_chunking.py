"""test_chunking.py — Chunking dengan presisi halaman dan hierarki heading."""

from __future__ import annotations

import pytest

from ingestion.chunking import (
    Chunk,
    _detect_heading,
    _join_lines,
    _starts_new_block,
    chunk_pages,
    looks_like_extraction_noise,
)


class TestChunkDataclass:
    def test_page_label_satu_halaman(self):
        assert Chunk(text="x", page_numbers=[48]).page_label == "48"

    def test_page_label_rentang(self):
        assert Chunk(text="x", page_numbers=[39, 40, 41]).page_label == "39-41"

    def test_page_label_tanpa_halaman(self):
        assert Chunk(text="x", page_numbers=[]).page_label == "-"

    def test_embed_text_menyertakan_heading(self):
        """
        Heading disisipkan ke teks yang diembed.

        Tanpa ini, potongan di tengah bab kehilangan satu-satunya petunjuk
        tentang topik yang sedang dibahas.
        """
        chunk = Chunk(text="isi paragraf", heading_path=["BAB IV", "Pasal 12"])
        assert chunk.embed_text.startswith("BAB IV > Pasal 12")
        assert "isi paragraf" in chunk.embed_text

    def test_embed_text_tanpa_heading_sama_dengan_teks(self):
        assert Chunk(text="isi").embed_text == "isi"

    def test_page_start_end(self):
        chunk = Chunk(text="x", page_numbers=[40, 39, 41])
        assert chunk.page_start == 39
        assert chunk.page_end == 41

    def test_to_dict_memuat_field_turunan(self):
        data = Chunk(text="abcd", page_numbers=[5]).to_dict()
        assert data["char_count"] == 4
        assert data["page_start"] == 5
        assert data["page_end"] == 5


class TestHeadingDetection:
    @pytest.mark.parametrize("line,level,structural", [
        ("BAB I", 0, True),
        ("BAB XII", 0, True),
        ("Bagian 2", 0, True),
        ("LAMPIRAN A", 0, True),
        ("Pasal 12", 1, False),
        ("PASAL 3", 1, False),
        ("1.2 Ruang Lingkup", 2, False),
        ("A. Azas", 2, False),
    ])
    def test_heading_dikenali_beserta_levelnya(self, line, level, structural):
        result = _detect_heading(line)
        assert result is not None, f"gagal mengenali heading: {line!r}"
        assert result[0] == level
        assert result[2] is structural

    def test_huruf_kapital_dianggap_heading(self):
        result = _detect_heading("KETENTUAN UMUM")
        assert result is not None
        assert result[0] == 0
        assert result[2] is False

    @pytest.mark.parametrize("line", [
        "ini paragraf biasa yang panjang dan tidak diawali penanda apa pun",
        "ab",
        "",
    ])
    def test_bukan_heading(self, line):
        assert _detect_heading(line) is None

    def test_baris_terlalu_panjang_bukan_heading(self):
        assert _detect_heading("BAB " + "x" * 200) is None


class TestJoinLines:
    def test_baris_disambung_dengan_spasi(self):
        assert _join_lines(["Mahasiswa wajib", "melakukan herregistrasi"]) == (
            "Mahasiswa wajib melakukan herregistrasi"
        )

    def test_kata_terpotong_tanda_hubung_disatukan(self):
        """PDF sering memotong kata di akhir baris."""
        assert _join_lines(["univer-", "sitas"]) == "universitas"

    def test_tanda_hubung_sebelum_huruf_kapital_dipertahankan(self):
        assert _join_lines(["S1-", "S2 UAJY"]) == "S1- S2 UAJY"

    def test_daftar_kosong(self):
        assert _join_lines([]) == ""


class TestStartsNewBlock:
    def test_baris_pertama_selalu_blok_baru(self):
        assert _starts_new_block(None, "teks apa pun", 40.0)

    def test_setelah_titik_blok_baru(self):
        assert _starts_new_block("Kalimat berakhir di sini.", "Kalimat baru", 40.0)

    def test_penanda_daftar_memulai_blok(self):
        assert _starts_new_block(
            "kalimat panjang yang belum selesai dan masih menggantung tanpa titik",
            "1) Butir pertama",
            40.0,
        )

    def test_baris_pendek_menandai_akhir_paragraf(self):
        """Pada teks rata kanan-kiri, baris pendek berarti paragraf berakhir."""
        assert _starts_new_block("pendek", "Kalimat berikutnya", 40.0)

    def test_baris_penuh_yang_menggantung_bukan_blok_baru(self):
        previous = "kalimat panjang yang jelas belum selesai dan masih berlanjut ke"
        assert not _starts_new_block(previous, "baris berikutnya", 40.0)


class TestExtractionNoiseFilter:
    def test_teks_terbalik_terdeteksi_sebagai_noise(self):
        """
        Prosa Indonesia yang sah selalu memuat kata fungsi.

        Sisa teks rusak dari diagram yang dirotasi tidak pernah memuatnya.
        """
        noise = ("satlukasf aatdluakpa arudtakpu lratsrutkurts "
                 "nskaaigleaPB naigaB isartsiniismadrtAsinimdA")
        assert looks_like_extraction_noise(noise)

    def test_prosa_normal_bukan_noise(self):
        assert not looks_like_extraction_noise(
            "Mahasiswa yang mengajukan cuti studi wajib memenuhi seluruh "
            "persyaratan administrasi yang berlaku di fakultas."
        )

    def test_tabel_berangka_tidak_dibuang(self):
        """Tabel minim kata fungsi, jadi dikenali dari kepadatan angkanya."""
        table = "1 85 - 100 A 4,00\n2 80 - 84,99 A- 3,70\n3 75 - 79,99 B+ 3,30"
        assert not looks_like_extraction_noise(table)

    def test_teks_kosong_dianggap_noise(self):
        assert looks_like_extraction_noise("")
        assert looks_like_extraction_noise("!!! ???")


class TestChunkPages:
    def test_heading_bab_disambung_dari_baris_berikutnya(self, pages_two_columns):
        """
        "BAB I" pada satu baris dan "PENDAHULUAN" pada baris berikutnya adalah
        satu judul yang terbelah tata letak, bukan dua heading.
        """
        chunks = chunk_pages(pages_two_columns, min_chunk_size=20, verbose=False)
        assert chunks
        assert chunks[0].heading_path[0] == "BAB I PENDAHULUAN"

    def test_teks_heading_tidak_hilang_dari_isi(self, pages_two_columns):
        """Judul harus ikut terindeks, bukan sekadar menjadi label."""
        chunks = chunk_pages(pages_two_columns, min_chunk_size=20, verbose=False)
        gabungan = " ".join(c.text for c in chunks)
        assert "PENDAHULUAN" in gabungan

    def test_setiap_chunk_punya_heading(self, pages_two_columns):
        chunks = chunk_pages(pages_two_columns, min_chunk_size=20, verbose=False)
        assert all(c.heading_path for c in chunks)

    def test_nomor_halaman_hanya_dari_baris_penyusunnya(self, pages_two_columns):
        """
        Inti dari presisi sitasi: chunk hanya boleh mengklaim halaman yang
        benar-benar menjadi sumber teksnya.
        """
        chunks = chunk_pages(pages_two_columns, min_chunk_size=20, verbose=False)
        for chunk in chunks:
            assert set(chunk.page_numbers) <= {10, 11}
            assert chunk.page_numbers

    def test_chunk_index_berurutan_rapat(self, pages_two_columns):
        chunks = chunk_pages(pages_two_columns, min_chunk_size=20, verbose=False)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chunk_terlalu_kecil_dibuang(self):
        pages = [(1, "Pendek.")]
        assert chunk_pages(pages, min_chunk_size=500, verbose=False) == []

    def test_halaman_kosong(self):
        assert chunk_pages([], verbose=False) == []
        assert chunk_pages([(1, "")], verbose=False) == []

    def test_batas_chunk_size_dihormati(self):
        panjang = " ".join(
            f"Kalimat nomor {i} berisi keterangan administrasi akademik."
            for i in range(200)
        )
        chunks = chunk_pages(
            [(1, panjang)], chunk_size=500, chunk_overlap=50,
            min_chunk_size=50, verbose=False,
        )
        assert chunks
        # Overlap boleh membuat sedikit lebih panjang, tapi tidak berlipat.
        for chunk in chunks:
            assert chunk.char_count <= 500 * 2

    def test_drop_noise_bisa_dimatikan(self):
        noise = ("satlukasf aatdluakpa arudtakpu lratsrutkurts nskaaigleaPB "
                 "naigaB isartsiniismadrtAsinimdA satlukasf aatdluakpa")
        pages = [(1, noise)]
        tanpa_filter = chunk_pages(pages, min_chunk_size=20, drop_noise=False, verbose=False)
        dengan_filter = chunk_pages(pages, min_chunk_size=20, drop_noise=True, verbose=False)
        assert len(tanpa_filter) > len(dengan_filter)
