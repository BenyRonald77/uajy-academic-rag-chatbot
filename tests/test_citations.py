"""Test parser sitasi dan tampilan sumber public."""

from app.citations import (
    citation_pages_are_in_context,
    extract_cited_pages,
    strip_inline_source_lines,
)
from app.retrieval import RetrievalCandidate, format_sources


class TestCitationParsing:
    def test_halaman_tunggal(self):
        assert extract_cited_pages("Sumber: Halaman 39") == [39]

    def test_rentang(self):
        assert extract_cited_pages("Sumber: Halaman 39-40") == [39, 40]

    def test_daftar(self):
        assert extract_cited_pages("Halaman 39, 40 dan 41") == [39, 40, 41]

    def test_dash_judul_tidak_menjadi_halaman(self):
        answer = "Halaman 43 — 5. Beban studi dan beban kredit semester"
        assert extract_cited_pages(answer) == [43]

    def test_citation_valid_dengan_context(self):
        contexts = [RetrievalCandidate(chunk_index=0, text="x", page_numbers=[39, 40])]
        assert citation_pages_are_in_context("Halaman 39-40", contexts) is True

    def test_citation_di_luar_context_ditolak(self):
        contexts = [RetrievalCandidate(chunk_index=0, text="x", page_numbers=[39])]
        assert citation_pages_are_in_context("Halaman 40", contexts) is False


class TestStripInlineSources:
    def test_menghapus_baris_sumber(self):
        answer = "Jawaban utama.\n\n📄 Sumber: Halaman 39 — H. Cuti Studi"
        assert strip_inline_source_lines(answer) == "Jawaban utama."

    def test_menyimpan_isi_jawaban(self):
        answer = "Poin pertama.\nPoin kedua.\n📄 Sumber: Halaman 39"
        assert strip_inline_source_lines(answer) == "Poin pertama.\nPoin kedua."


class TestCleanSourceCard:
    def _candidate(self, index, pages, score, section):
        return RetrievalCandidate(
            chunk_index=index,
            text="x",
            page_numbers=pages,
            dense_score=score,
            section_title=section,
            heading_path=["PROGRAM", section],
        )

    def test_hanya_halaman_yang_dikutip_ditampilkan(self):
        results = [
            self._candidate(0, [39], 0.82, "H. Cuti Studi"),
            self._candidate(1, [40], 0.81, "H. Cuti Studi"),
            self._candidate(2, [107, 108], 0.67, "PERPUSTAKAAN"),
            self._candidate(3, [42], 0.76, "J. Perkuliahan"),
        ]
        result = format_sources(
            results,
            cited_pages={39, 40},
            include_scores=False,
        )
        assert "Halaman 39–40" in result
        assert "PERPUSTAKAAN" not in result
        assert "Perkuliahan" not in result
        assert "relevansi" not in result

    def test_overlap_digabung_menjadi_satu_baris(self):
        results = [
            self._candidate(0, [39], 0.82, "H. Cuti Studi"),
            self._candidate(1, [39, 40], 0.90, "H. Cuti Studi"),
            self._candidate(2, [40], 0.81, "H. Cuti Studi"),
        ]
        result = format_sources(results, cited_pages={39, 40}, include_scores=False)
        assert result.count("Halaman") == 1
        assert "Halaman 39–40" in result

    def test_operator_boleh_melihat_skor(self):
        result = format_sources(
            [self._candidate(0, [39], 0.82, "H. Cuti Studi")],
            cited_pages={39},
            include_scores=True,
        )
        assert "relevansi" in result
        assert "82%" in result

    def test_dokumen_berbeda_tetap_terpisah(self):
        a = self._candidate(0, [39], 0.82, "H. Cuti Studi")
        b = self._candidate(1, [39], 0.81, "SK Rektor")
        a.source_document = "pedoman.pdf"
        b.source_document = "sk-rektor.pdf"
        result = format_sources([a, b], cited_pages={39}, include_scores=False)
        assert "pedoman" in result
        assert "sk-rektor" in result
        assert len(result.splitlines()) == 2
