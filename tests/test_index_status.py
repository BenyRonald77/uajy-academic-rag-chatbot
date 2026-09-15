"""
test_index_status.py — Deteksi index usang dan dukungan multi-dokumen.

Hash dokumen sudah dicatat sejak awal, tetapi sebelum modul ini ada, tidak ada
kode yang pernah membacanya kembali untuk membandingkan. Test di sini menjaga
agar pemeriksaannya benar-benar berjalan.
"""

from __future__ import annotations

import json

import pytest

from app.index_status import (
    DocumentStatus,
    IndexFreshness,
    check_index_freshness,
    file_sha256,
    indexed_documents,
    load_index_info,
)


@pytest.fixture
def data_dir(tmp_path):
    """Folder dokumen tiruan berisi dua PDF palsu."""
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "pedoman.pdf").write_bytes(b"isi pedoman akademik")
    (folder / "kalender.pdf").write_bytes(b"isi kalender akademik")
    return folder


def info_for(data_dir, names: list[str]) -> dict:
    """Catatan build yang cocok dengan isi folder saat ini."""
    return {
        "source_documents": [
            {"name": name, "sha256": file_sha256(data_dir / name)}
            for name in names
        ]
    }


class TestFileSha256:
    def test_isi_sama_hash_sama(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"isi identik")
        b.write_bytes(b"isi identik")
        assert file_sha256(a) == file_sha256(b)

    def test_isi_berbeda_hash_berbeda(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"isi pertama")
        b.write_bytes(b"isi kedua")
        assert file_sha256(a) != file_sha256(b)


class TestIndexedDocuments:
    def test_format_multi_dokumen(self):
        info = {"source_documents": [{"name": "a.pdf"}, {"name": "b.pdf"}]}
        assert len(indexed_documents(info)) == 2

    def test_format_lama_dokumen_tunggal(self):
        """Index versi lama memakai kunci `source_pdf`."""
        info = {"source_pdf": {"name": "pedoman.pdf", "sha256": "abc"}}
        hasil = indexed_documents(info)
        assert len(hasil) == 1
        assert hasil[0]["name"] == "pedoman.pdf"

    def test_tanpa_catatan(self):
        assert indexed_documents({}) == []
        assert indexed_documents({"source_pdf": {}}) == []


class TestLoadIndexInfo:
    def test_berkas_tidak_ada(self, tmp_path):
        assert load_index_info(tmp_path / "tidak_ada.json") == {}

    def test_json_rusak_tidak_meledak(self, tmp_path):
        path = tmp_path / "rusak.json"
        path.write_text("{ bukan json", encoding="utf-8")
        assert load_index_info(path) == {}

    def test_json_valid_terbaca(self, tmp_path):
        path = tmp_path / "info.json"
        path.write_text(json.dumps({"built_at": "2026-01-01"}), encoding="utf-8")
        assert load_index_info(path)["built_at"] == "2026-01-01"


class TestCheckIndexFreshness:
    def test_index_segar(self, data_dir):
        freshness = check_index_freshness(
            info_for(data_dir, ["pedoman.pdf", "kalender.pdf"]), data_dir
        )
        assert freshness.has_info is True
        assert freshness.is_stale is False
        assert freshness.changed == []
        assert freshness.messages() == []

    def test_dokumen_berubah_terdeteksi(self, data_dir):
        """Inti dari modul ini: PDF yang diperbarui harus terdeteksi."""
        info = info_for(data_dir, ["pedoman.pdf", "kalender.pdf"])
        (data_dir / "pedoman.pdf").write_bytes(b"isi pedoman EDISI BARU")

        freshness = check_index_freshness(info, data_dir)

        assert freshness.is_stale is True
        assert [d.name for d in freshness.changed] == ["pedoman.pdf"]
        assert any("pedoman.pdf" in m for m in freshness.messages())

    def test_dokumen_baru_belum_terindeks(self, data_dir):
        info = info_for(data_dir, ["pedoman.pdf"])
        freshness = check_index_freshness(info, data_dir)

        assert freshness.untracked_documents == ["kalender.pdf"]
        assert freshness.is_stale is True

    def test_dokumen_hilang_tidak_membuat_usang(self, data_dir):
        """
        Berkas sumber yang hilang tidak membuat index tidak sahih.

        Index-nya masih benar; yang tidak tersedia hanya berkas untuk
        diperiksa. Menandainya usang akan mendorong rebuild yang sia-sia —
        dan justru tidak mungkin dilakukan tanpa berkasnya.
        """
        info = info_for(data_dir, ["pedoman.pdf", "kalender.pdf"])
        (data_dir / "kalender.pdf").unlink()

        freshness = check_index_freshness(info, data_dir)

        assert [d.name for d in freshness.missing] == ["kalender.pdf"]
        assert freshness.is_stale is False
        assert any("tidak ditemukan" in m for m in freshness.messages())

    def test_tanpa_catatan_build(self, data_dir):
        freshness = check_index_freshness({}, data_dir)
        assert freshness.has_info is False
        assert freshness.is_stale is False
        assert any("dibangun sebelum" in m for m in freshness.messages())

    def test_format_lama_tetap_diperiksa(self, data_dir):
        info = {
            "source_pdf": {
                "name": "pedoman.pdf",
                "sha256": file_sha256(data_dir / "pedoman.pdf"),
            }
        }
        assert check_index_freshness(info, data_dir).changed == []

        (data_dir / "pedoman.pdf").write_bytes(b"diubah")
        assert check_index_freshness(info, data_dir).is_stale is True

    def test_folder_data_tidak_ada(self, tmp_path):
        info = {"source_documents": [{"name": "hilang.pdf", "sha256": "abc"}]}
        freshness = check_index_freshness(info, tmp_path / "tidak_ada")
        assert freshness.missing
        assert freshness.untracked_documents == []


class TestDocumentStatus:
    def test_berubah_saat_hash_berbeda(self):
        status = DocumentStatus(
            name="a.pdf", indexed_sha256="aaa", current_sha256="bbb", exists=True
        )
        assert status.changed is True

    def test_tidak_berubah_saat_hash_sama(self):
        status = DocumentStatus(
            name="a.pdf", indexed_sha256="aaa", current_sha256="aaa", exists=True
        )
        assert status.changed is False

    def test_berkas_hilang_bukan_berubah(self):
        status = DocumentStatus(name="a.pdf", indexed_sha256="aaa", exists=False)
        assert status.changed is False

    def test_tanpa_hash_tercatat_bukan_berubah(self):
        """Index lama bisa saja tidak menyimpan hash; jangan menuduh."""
        status = DocumentStatus(
            name="a.pdf", indexed_sha256="", current_sha256="bbb", exists=True
        )
        assert status.changed is False


class TestIndexFreshnessDefaults:
    def test_kosong_tidak_usang(self):
        assert IndexFreshness().is_stale is False
