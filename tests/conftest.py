"""
conftest.py — Fixture bersama untuk seluruh test.

Semua fixture di sini murni data, tanpa panggilan jaringan, sehingga seluruh
test unit bisa berjalan tanpa API key maupun kuota.
"""

from __future__ import annotations

import pytest

from app.retrieval import RetrievalCandidate


@pytest.fixture
def sample_metadata() -> list[dict]:
    """
    Metadata tiruan yang meniru bentuk ``index/metadata.json``.

    Isinya dipilih agar menguji hal yang berbeda: istilah eksak dengan angka,
    dokumen yang hanya mirip topiknya, dan dokumen yang sama sekali tidak
    berkaitan.
    """
    return [
        {
            "chunk_index": 0,
            "text": "Predikat kelulusan Pujian atau Excellent dengan IPK 3,51 sampai "
                    "4,00 dapat ditetapkan menjadi predikat CUM LAUDE apabila masa "
                    "studi lulusan maksimal adalah masa studi normal ditambah satu "
                    "semester dan tidak ada mata kuliah dengan nilai kurang dari B.",
            "page_numbers": [48],
            "page_start": 48,
            "page_end": 48,
            "section_title": "M. Evaluasi Hasil Belajar dan Yudisium",
            "heading_path": ["BAB IV KURIKULUM", "M. Evaluasi Hasil Belajar dan Yudisium"],
            "char_count": 260,
            "estimated_tokens": 65,
        },
        {
            "chunk_index": 1,
            "text": "Program Sarjana memiliki beban belajar paling sedikit 144 SKS "
                    "yang dijadwalkan untuk dapat diselesaikan dalam masa studi "
                    "normal selama 8 semester dan paling lama 6 tahun.",
            "page_numbers": [27],
            "page_start": 27,
            "page_end": 27,
            "section_title": "A. Program Sarjana",
            "heading_path": ["BAB II PROGRAM PENDIDIKAN", "A. Program Sarjana"],
            "char_count": 160,
            "estimated_tokens": 40,
        },
        {
            "chunk_index": 2,
            "text": "Maksimal cuti studi yang dapat diberikan kepada mahasiswa adalah "
                    "2 dua semester, baik berurutan maupun tidak. Cuti studi tidak "
                    "diperhitungkan sebagai masa studi.",
            "page_numbers": [39, 40],
            "page_start": 39,
            "page_end": 40,
            "section_title": "H. Cuti Studi",
            "heading_path": ["BAB III ADMINISTRASI", "H. Cuti Studi"],
            "char_count": 150,
            "estimated_tokens": 37,
        },
        {
            "chunk_index": 3,
            "text": "Perpustakaan universitas menyediakan ruang baca, koleksi buku "
                    "tercetak, dan akses jurnal elektronik bagi seluruh mahasiswa.",
            "page_numbers": [93],
            "page_start": 93,
            "page_end": 93,
            "section_title": "PERPUSTAKAAN",
            "heading_path": ["PERPUSTAKAAN"],
            "char_count": 120,
            "estimated_tokens": 30,
        },
    ]


@pytest.fixture
def candidate_factory():
    """Pabrik `RetrievalCandidate` ringkas untuk test."""

    def make(
        chunk_index: int = 0,
        text: str = "teks contoh",
        page_numbers: list[int] | None = None,
        dense_score: float = 0.5,
        rerank_score: float | None = None,
        **kwargs,
    ) -> RetrievalCandidate:
        return RetrievalCandidate(
            chunk_index=chunk_index,
            text=text,
            page_numbers=page_numbers if page_numbers is not None else [1],
            dense_score=dense_score,
            rerank_score=rerank_score,
            **kwargs,
        )

    return make


@pytest.fixture
def pages_two_columns() -> list[tuple[int, str]]:
    """
    Halaman tiruan yang meniru tata letak PDF sungguhan.

    `extract_text` membuang baris kosong, jadi batas paragraf harus
    disimpulkan dari panjang baris dan tanda baca. Data ini sengaja dibuat
    dengan baris yang terpotong di tengah kalimat.
    """
    return [
        (
            10,
            "BAB I\n"
            "PENDAHULUAN\n"
            "Buku pedoman akademik ini disusun sebagai acuan resmi bagi seluruh\n"
            "mahasiswa Fakultas Teknologi Industri dalam menjalani seluruh proses\n"
            "pembelajaran di lingkungan universitas.\n"
            "Pedoman ini mencakup ketentuan umum, struktur kurikulum, serta\n"
            "prosedur administrasi akademik yang berlaku.",
        ),
        (
            11,
            "Pasal 1\n"
            "Pedoman ini berlaku untuk seluruh mahasiswa aktif pada tahun akademik\n"
            "yang sedang berjalan.\n"
            "Pasal 2\n"
            "Mahasiswa wajib mengikuti seluruh peraturan yang ditetapkan oleh\n"
            "universitas maupun fakultas.",
        ),
    ]
