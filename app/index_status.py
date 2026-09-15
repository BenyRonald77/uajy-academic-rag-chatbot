"""
index_status.py — Deteksi index yang sudah usang terhadap dokumen sumbernya.

Masalah yang diselesaikan:
    Index dibangun dari PDF pada satu titik waktu. Bila PDF-nya diperbarui —
    misalnya pedoman akademik tahun ajaran baru — index lama tetap terpakai
    tanpa keluhan apa pun. Chatbot lalu menjawab dengan penuh keyakinan dari
    dokumen kedaluwarsa, lengkap dengan sitasi halaman yang tampak sah tetapi
    merujuk terbitan yang salah. Itu kegagalan yang paling sulit disadari.

`build_index.py` sudah mencatat hash SHA-256 setiap dokumen di
`index/index_info.json`. Modul ini yang membacanya kembali dan
membandingkannya dengan berkas yang ada sekarang.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.config import DATA_DIR, INDEX_INFO_PATH, INGESTION_PIPELINE_VERSION


def file_sha256(path: str | Path) -> str:
    """Hash SHA-256 sebuah berkas, dibaca bertahap agar hemat memori."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class DocumentStatus:
    """Status satu dokumen sumber terhadap index."""

    name: str
    indexed_sha256: str = ""
    current_sha256: str | None = None
    exists: bool = True

    @property
    def changed(self) -> bool:
        """True bila berkasnya ada tetapi isinya sudah berbeda."""
        return (
            self.exists
            and self.current_sha256 is not None
            and bool(self.indexed_sha256)
            and self.current_sha256 != self.indexed_sha256
        )


@dataclass
class IndexFreshness:
    """Ringkasan kesegaran index terhadap seluruh dokumen sumber."""

    has_info: bool = False
    documents: list[DocumentStatus] = field(default_factory=list)
    untracked_documents: list[str] = field(default_factory=list)

    #: Versi pipeline yang tercatat pada index, dan versi kode saat ini.
    #: Index yang dibangun sebelum penomoran ini ada dianggap versi 0.
    indexed_pipeline_version: int = 0
    current_pipeline_version: int = INGESTION_PIPELINE_VERSION

    @property
    def pipeline_outdated(self) -> bool:
        """
        True bila index dibangun oleh logika pipeline yang lebih lama.

        Ini menangkap kelas keusangan yang tidak terlihat dari hash dokumen:
        PDF-nya tidak berubah, tetapi aturan ekstraksi atau chunking sudah
        diperbaiki sehingga isi chunk yang dihasilkan berbeda.

        Bernilai False bila catatan build tidak ada sama sekali. Keadaan itu
        sudah dilaporkan lewat pesannya sendiri, dan menandainya usang di sini
        hanya akan memunculkan dua keluhan untuk satu sebab yang sama.
        """
        if not self.has_info:
            return False
        return self.indexed_pipeline_version < self.current_pipeline_version

    @property
    def changed(self) -> list[DocumentStatus]:
        return [d for d in self.documents if d.changed]

    @property
    def missing(self) -> list[DocumentStatus]:
        return [d for d in self.documents if not d.exists]

    @property
    def is_stale(self) -> bool:
        """
        True bila index perlu dibangun ulang.

        Dokumen yang hilang **tidak** membuat index usang: index-nya masih
        sahih, hanya berkas sumbernya yang tidak lagi tersedia untuk diperiksa.
        Yang membuat usang adalah isi dokumen yang berubah, dokumen baru yang
        belum terindeks, atau logika pipeline yang sudah lebih maju daripada
        saat index dibangun.
        """
        return bool(self.changed or self.untracked_documents or self.pipeline_outdated)

    def messages(self) -> list[str]:
        """Pesan siap tampil, satu per masalah yang ditemukan."""
        pesan: list[str] = []

        if not self.has_info:
            pesan.append(
                "Index ini dibangun sebelum pencatatan asal-usul ada, sehingga "
                "kesegarannya tidak dapat diperiksa. Bangun ulang untuk "
                "mengaktifkan deteksi otomatis."
            )
            return pesan

        if self.pipeline_outdated:
            pesan.append(
                f"Pipeline ingestion sudah diperbarui "
                f"(versi {self.indexed_pipeline_version} → "
                f"{self.current_pipeline_version}). Dokumennya tidak berubah, "
                "tetapi aturan ekstraksi atau chunking sudah diperbaiki, "
                "sehingga index sekarang menghasilkan chunk yang berbeda."
            )

        for doc in self.changed:
            pesan.append(
                f"Dokumen **{doc.name}** sudah berubah sejak index dibangun. "
                "Jawaban chatbot masih memakai versi lama."
            )
        for doc in self.missing:
            pesan.append(
                f"Berkas sumber **{doc.name}** tidak ditemukan lagi, jadi "
                "kesegarannya tidak bisa diperiksa."
            )
        for name in self.untracked_documents:
            pesan.append(
                f"Dokumen **{name}** ada di folder data tetapi belum terindeks."
            )
        return pesan


def load_index_info(path: str | Path = INDEX_INFO_PATH) -> dict:
    """Baca `index_info.json`. Kembalikan dict kosong bila tidak ada/rusak."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def indexed_documents(index_info: dict) -> list[dict]:
    """
    Ambil daftar dokumen sumber dari catatan build.

    Menangani dua bentuk: ``source_documents`` (multi-dokumen) dan
    ``source_pdf`` (dokumen tunggal, format lama).
    """
    documents = index_info.get("source_documents")
    if isinstance(documents, list) and documents:
        return documents

    single = index_info.get("source_pdf")
    if isinstance(single, dict) and single.get("name"):
        return [single]

    return []


def check_index_freshness(
    index_info: dict | None = None,
    data_dir: str | Path = DATA_DIR,
) -> IndexFreshness:
    """
    Bandingkan hash dokumen yang tercatat di index dengan berkas saat ini.

    Args:
        index_info: Isi ``index_info.json``. Dibaca sendiri bila ``None``.
        data_dir: Folder tempat dokumen sumber berada.

    Returns:
        `IndexFreshness` berisi dokumen yang berubah, hilang, dan yang belum
        terindeks.
    """
    if index_info is None:
        index_info = load_index_info()

    freshness = IndexFreshness(has_info=bool(index_info))
    if not index_info:
        return freshness

    freshness.indexed_pipeline_version = int(index_info.get("pipeline_version", 0) or 0)

    data_dir = Path(data_dir)
    tercatat: set[str] = set()

    for entry in indexed_documents(index_info):
        name = entry.get("name", "")
        if not name:
            continue
        tercatat.add(name)

        path = data_dir / name
        status = DocumentStatus(
            name=name,
            indexed_sha256=entry.get("sha256", ""),
            exists=path.exists(),
        )
        if status.exists:
            try:
                status.current_sha256 = file_sha256(path)
            except OSError:
                status.exists = False
        freshness.documents.append(status)

    if data_dir.exists():
        freshness.untracked_documents = sorted(
            p.name for p in data_dir.glob("*.pdf") if p.name not in tercatat
        )

    return freshness
