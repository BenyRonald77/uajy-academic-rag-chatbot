"""test_lexical_index.py — BM25 Okapi dan cakupan istilah berbobot IDF."""

from __future__ import annotations

import pytest

from app.lexical_index import LexicalIndex
from app.text_utils import content_tokens


@pytest.fixture
def index(sample_metadata) -> LexicalIndex:
    return LexicalIndex.from_metadata(sample_metadata)


class TestConstruction:
    def test_statistik_dasar(self, index, sample_metadata):
        assert index.num_docs == len(sample_metadata)
        assert index.vocabulary_size > 0
        assert index.avg_doc_length > 0

    def test_index_kosong_tidak_meledak(self):
        empty = LexicalIndex([])
        assert empty.num_docs == 0
        assert empty.search("apa pun") == []

    def test_heading_ikut_diindeks(self, sample_metadata):
        """
        Judul bagian harus bisa dicari.

        Query seperti "BAB IV kurikulum" hanya cocok lewat heading, sebab
        kata itu tidak muncul di badan paragrafnya.
        """
        index = LexicalIndex.from_metadata(sample_metadata)
        hits = index.search("BAB IV kurikulum", top_n=3)
        assert hits, "heading seharusnya bisa dicocokkan"
        assert hits[0].doc_id == 0

    def test_indexable_text_tanpa_heading_path(self):
        entry = {"text": "isi dokumen", "section_title": "Judul Bagian"}
        assert "Judul Bagian" in LexicalIndex.indexable_text(entry)

    def test_indexable_text_mengutamakan_heading_path(self):
        entry = {
            "text": "isi",
            "section_title": "diabaikan",
            "heading_path": ["BAB I", "Pasal 2"],
        }
        result = LexicalIndex.indexable_text(entry)
        assert "BAB I > Pasal 2" in result
        assert "diabaikan" not in result


class TestIDF:
    def test_istilah_di_luar_korpus_mendapat_idf_maksimum(self, index):
        """
        Ini yang membuat gate leksikal bisa menolak pertanyaan asing.

        Istilah yang tidak ada di korpus menyumbang bobot terbesar ke penyebut
        cakupan tetapi nol ke pembilang, sehingga skornya jatuh keras.
        """
        assert index.idf("pythagoras") == index.max_idf
        assert index.idf("pythagoras") > index.idf("mahasiswa")

    def test_idf_selalu_positif(self, index):
        """
        Varian Lucene dipakai supaya istilah yang muncul di lebih dari
        separuh dokumen tidak berkontribusi negatif seperti pada Okapi asli.
        """
        for term in index.doc_frequency:
            assert index.idf(term) > 0

    def test_istilah_jarang_lebih_tinggi_dari_yang_umum(self, index):
        assert index.idf("cumlaude") >= index.idf("studi")


class TestSearch:
    def test_menemukan_dokumen_yang_tepat(self, index):
        hits = index.search("Berapa IPK minimum untuk cum laude?", top_n=2)
        assert hits[0].doc_id == 0

    def test_istilah_angka_eksak(self, index):
        """Justru inilah alasan jalur leksikal ada."""
        hits = index.search("144 SKS", top_n=2)
        assert hits[0].doc_id == 1

    def test_query_tanpa_kecocokan_mengembalikan_kosong(self, index):
        assert index.search("pythagoras trigonometri", top_n=5) == []

    def test_hasil_terurut_menurun(self, index):
        hits = index.search("masa studi mahasiswa semester", top_n=4)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_dihormati(self, index):
        assert len(index.search("studi", top_n=1)) <= 1

    def test_matched_terms_dilaporkan(self, index):
        hits = index.search("cuti studi", top_n=1)
        assert "cuti" in hits[0].matched_terms

    def test_query_kosong(self, index):
        assert index.search("", top_n=5) == []


class TestCoverage:
    def test_cakupan_penuh_saat_semua_istilah_ada(self, index):
        tokens = content_tokens("cuti studi")
        assert index.coverage(tokens, 2) == pytest.approx(1.0)

    def test_cakupan_rendah_untuk_pertanyaan_di_luar_cakupan(self, index):
        """
        "cara memasak nasi goreng": hanya "cara" yang mungkin ada di korpus,
        sisanya asing dan masing-masing menyumbang IDF maksimum.
        """
        tokens = content_tokens("cara memasak nasi goreng")
        for doc_id in range(index.num_docs):
            assert index.coverage(tokens, doc_id) < 0.5

    def test_cakupan_nol_untuk_token_kosong(self, index):
        assert index.coverage([], 0) == 0.0

    def test_cakupan_dalam_rentang_valid(self, index):
        tokens = content_tokens("masa studi maksimal program sarjana")
        for doc_id in range(index.num_docs):
            assert 0.0 <= index.coverage(tokens, doc_id) <= 1.0
