"""test_text_utils.py — Tokenisasi dan normalisasi Bahasa Indonesia."""

from __future__ import annotations

import pytest

from app.text_utils import (
    INDONESIAN_STOPWORDS,
    content_tokens,
    normalize_text,
    strip_clitic,
    tokenize,
)


class TestNormalizeText:
    def test_huruf_dikecilkan(self):
        assert normalize_text("IPK Minimum") == "ipk minimum"

    def test_aksen_dibuang(self):
        assert normalize_text("Café") == "cafe"

    @pytest.mark.parametrize("phrase,expected", [
        ("mata kuliah", "matakuliah"),
        ("tugas akhir", "tugasakhir"),
        ("cum laude", "cumlaude"),
        ("program studi", "programstudi"),
        ("indeks prestasi kumulatif", "ipk"),
        ("satuan kredit semester", "sks"),
    ])
    def test_frasa_alias_disatukan(self, phrase, expected):
        """
        Alias multi-kata harus jadi satu token.

        Tanpa ini "mata kuliah" terpecah jadi dua token biasa dan kehilangan
        daya bedanya, sebab "mata" dan "kuliah" masing-masing sangat umum.
        """
        assert expected in normalize_text(f"tentang {phrase} di kampus")


class TestStripClitic:
    @pytest.mark.parametrize("word,expected", [
        ("syaratnya", "syarat"),
        ("bukunya", "buku"),
        ("bagaimanakah", "bagaimana"),
        ("biayanya", "biaya"),
    ])
    def test_klitik_dibuang(self, word, expected):
        assert strip_clitic(word) == expected

    @pytest.mark.parametrize("word", ["nya", "kah", "lah", "aku"])
    def test_kata_pendek_tidak_dipotong(self, word):
        """Pemotongan pada kata pendek menghasilkan potongan tak bermakna."""
        assert strip_clitic(word) == word

    @pytest.mark.parametrize("word", ["adakah", "punya", "kupu"])
    def test_penjaga_panjang_minimum_menahan_over_stemming(self, word):
        """
        Klitik hanya dibuang bila sisa katanya masih cukup panjang.

        Konsekuensinya "adakah" tidak menjadi "ada" — imbalan yang murah,
        sebab tanpa penjaga ini "punya" akan terpotong menjadi "pu" dan
        "kupu" menjadi "kup". Presisi lebih penting daripada recall untuk
        korpus sekecil satu buku pedoman.
        """
        assert strip_clitic(word) == word


class TestTokenize:
    def test_stopword_dibuang(self):
        tokens = tokenize("Apa saja syarat yang harus dipenuhi untuk lulus?")
        assert "syarat" in tokens
        assert "lulus" in tokens
        for stopword in ("apa", "saja", "yang", "harus", "untuk"):
            assert stopword not in tokens

    def test_angka_dipertahankan(self):
        """Dokumen akademik penuh rujukan numerik yang tidak boleh hilang."""
        tokens = tokenize("Beban belajar 144 SKS dan IPK 3,51 pada Pasal 12")
        assert "144" in tokens
        assert "12" in tokens
        assert "51" in tokens

    def test_tanpa_stopword_removal(self):
        tokens = tokenize("syarat yang berlaku", remove_stopwords=False)
        assert "yang" in tokens

    @pytest.mark.parametrize("token_alias,canonical", [
        ("skripsi", "tugasakhir"),
        ("prodi", "programstudi"),
        ("matkul", "matakuliah"),
    ])
    def test_alias_satu_kata(self, token_alias, canonical):
        assert canonical in tokenize(f"tentang {token_alias}")

    def test_token_satu_huruf_dibuang(self):
        assert "a" not in tokenize("nilai a dan b")

    def test_string_kosong(self):
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_hanya_tanda_baca(self):
        assert tokenize("!!! ??? ---") == []

    def test_stopword_list_masuk_akal(self):
        for word in ("dan", "yang", "di", "untuk", "dengan", "apa", "bagaimana"):
            assert word in INDONESIAN_STOPWORDS


class TestContentTokens:
    def test_duplikat_dibuang_urutan_dijaga(self):
        tokens = content_tokens("syarat cuti dan syarat yudisium")
        assert tokens == ["syarat", "cuti", "yudisium"]

    def test_kosong_untuk_query_tanpa_isi(self):
        assert content_tokens("dan yang untuk dengan") == []
