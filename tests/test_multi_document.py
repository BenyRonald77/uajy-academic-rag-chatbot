"""
test_multi_document.py — Dukungan banyak dokumen dalam satu index.

Nomor halaman hanya bermakna dalam konteks dokumennya: "halaman 48" pada
pedoman akademik dan pada SK Rektor merujuk hal yang sama sekali berbeda.
Karena itu asal dokumen harus terbawa dari chunking sampai ke sitasi.
"""

from __future__ import annotations

import pytest

from app.config import RetrievalConfig
from app.retrieval import HybridRetriever, RetrievalCandidate, format_sources
from ingestion.build_index import chunks_to_metadata, discover_pdfs
from ingestion.chunking import Chunk


class TestDiscoverPdfs:
    def test_folder_bawaan_dipindai(self, tmp_path):
        (tmp_path / "b.pdf").touch()
        (tmp_path / "a.pdf").touch()
        (tmp_path / "catatan.txt").touch()

        hasil = discover_pdfs(None, data_dir=tmp_path)

        assert [p.name for p in hasil] == ["a.pdf", "b.pdf"], "harus terurut"

    def test_path_eksplisit(self, tmp_path):
        a = tmp_path / "a.pdf"
        a.touch()
        assert discover_pdfs([str(a)]) == [a]

    def test_folder_sebagai_argumen(self, tmp_path):
        (tmp_path / "a.pdf").touch()
        (tmp_path / "b.pdf").touch()
        assert len(discover_pdfs([str(tmp_path)])) == 2

    def test_duplikat_dibuang_urutan_dijaga(self, tmp_path):
        a = tmp_path / "a.pdf"
        a.touch()
        assert len(discover_pdfs([str(a), str(a)])) == 1

    def test_folder_kosong(self, tmp_path):
        assert discover_pdfs(None, data_dir=tmp_path) == []


class TestChunkSourceDocument:
    def test_default_kosong_untuk_kompatibilitas(self):
        """Index dokumen tunggal versi lama tidak punya field ini."""
        assert Chunk(text="x").source_document == ""

    def test_metadata_memuat_source_document(self):
        chunks = [
            Chunk(text="isi pedoman", page_numbers=[48], chunk_index=0,
                  source_document="pedoman.pdf"),
            Chunk(text="isi kalender", page_numbers=[2], chunk_index=1,
                  source_document="kalender.pdf"),
        ]
        metadata = chunks_to_metadata(chunks)
        assert [m["source_document"] for m in metadata] == ["pedoman.pdf", "kalender.pdf"]


class TestCandidateDocumentLabel:
    def test_ekstensi_dibuang(self):
        c = RetrievalCandidate(
            chunk_index=0, text="x", page_numbers=[1], source_document="pedoman.pdf"
        )
        assert c.document_label == "pedoman"

    def test_kosong_saat_tanpa_dokumen(self):
        c = RetrievalCandidate(chunk_index=0, text="x", page_numbers=[1])
        assert c.document_label == ""


class TestFormatSourcesMultiDocument:
    def test_nama_dokumen_muncul_saat_lebih_dari_satu(self):
        results = [
            RetrievalCandidate(chunk_index=0, text="a", page_numbers=[48],
                               source_document="pedoman.pdf", dense_score=0.9),
            RetrievalCandidate(chunk_index=1, text="b", page_numbers=[2],
                               source_document="kalender.pdf", dense_score=0.8),
        ]
        hasil = format_sources(results)
        assert "pedoman" in hasil
        assert "kalender" in hasil

    def test_nama_dokumen_disembunyikan_saat_hanya_satu(self):
        """
        Menyebut nama dokumen setiap kali pada index dokumen tunggal hanya
        menambah keriuhan tanpa menambah informasi.
        """
        results = [
            RetrievalCandidate(chunk_index=0, text="a", page_numbers=[48],
                               source_document="pedoman.pdf", dense_score=0.9),
            RetrievalCandidate(chunk_index=1, text="b", page_numbers=[27],
                               source_document="pedoman.pdf", dense_score=0.8),
        ]
        assert "pedoman" not in format_sources(results)

    def test_halaman_sama_dari_dokumen_berbeda_tetap_terpisah(self):
        """
        Kasus yang paling mudah salah: halaman 48 pada dua dokumen berbeda
        adalah dua sumber, bukan satu.
        """
        results = [
            RetrievalCandidate(chunk_index=0, text="a", page_numbers=[48],
                               source_document="pedoman.pdf", section_title="Bab A"),
            RetrievalCandidate(chunk_index=1, text="b", page_numbers=[48],
                               source_document="sk-rektor.pdf", section_title="Bab A"),
        ]
        assert len(format_sources(results).splitlines()) == 2


class TestDocumentFilter:
    @staticmethod
    def _retriever(metadata: list[dict]) -> HybridRetriever:
        """Retriever tanpa FAISS — hanya bagian penyaringan yang diuji."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.metadata = metadata
        return retriever

    @pytest.fixture
    def metadata(self) -> list[dict]:
        return [
            {"chunk_index": 0, "text": "a", "source_document": "pedoman.pdf"},
            {"chunk_index": 1, "text": "b", "source_document": "pedoman.pdf"},
            {"chunk_index": 2, "text": "c", "source_document": "kalender.pdf"},
        ]

    def test_tanpa_filter_mengembalikan_none(self, metadata):
        """None berarti pemanggil boleh melewati penyaringan sepenuhnya."""
        assert self._retriever(metadata).allowed_doc_ids(()) is None

    def test_filter_satu_dokumen(self, metadata):
        allowed = self._retriever(metadata).allowed_doc_ids(("kalender.pdf",))
        assert allowed == {2}

    def test_filter_beberapa_dokumen(self, metadata):
        allowed = self._retriever(metadata).allowed_doc_ids(
            ("pedoman.pdf", "kalender.pdf")
        )
        assert allowed == {0, 1, 2}

    def test_dokumen_tak_dikenal_menghasilkan_kosong(self, metadata):
        assert self._retriever(metadata).allowed_doc_ids(("tidak_ada.pdf",)) == set()

    def test_config_menyimpan_filter_sebagai_tuple(self):
        """Harus tuple agar dataclass config tetap frozen dan bisa di-hash."""
        config = RetrievalConfig().with_overrides(document_filter=("a.pdf",))
        assert config.document_filter == ("a.pdf",)
        assert RetrievalConfig().document_filter == ()


class TestSourceDocumentsProperty:
    @staticmethod
    def _retriever(metadata):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.metadata = metadata
        return retriever

    def test_daftar_unik_terurut(self):
        metadata = [
            {"source_document": "kalender.pdf"},
            {"source_document": "pedoman.pdf"},
            {"source_document": "pedoman.pdf"},
        ]
        assert self._retriever(metadata).source_documents == [
            "kalender.pdf", "pedoman.pdf"
        ]

    def test_index_lama_menghasilkan_daftar_kosong(self):
        """Tanpa source_document, UI menyembunyikan pemilih dokumen."""
        metadata = [{"text": "a"}, {"text": "b"}]
        assert self._retriever(metadata).source_documents == []
