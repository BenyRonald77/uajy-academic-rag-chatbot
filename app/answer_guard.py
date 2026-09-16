"""Pengaman minimal agar respons provider yang tidak valid tidak tampil ke user."""

from __future__ import annotations

import re

# Prompt builder mewajibkan sumber halaman pada jawaban akhir. Jika gateway
# mengabaikan prompt dan mengembalikan teks acak/motivasi, respons itu tidak
# boleh ditampilkan kepada mahasiswa sebagai jawaban akademik.
_PAGE_CITATION_RE = re.compile(r"\bhalaman\s*:?\s*\d{1,4}\b", re.IGNORECASE)
_ERROR_MARKERS = (
    "⚠️",
    "not_found",
    "resource_exhausted",
    "rate limit",
    "api key",
)


def has_page_citation(answer: str) -> bool:
    """True bila jawaban memuat minimal satu sitasi halaman."""
    return bool(answer and _PAGE_CITATION_RE.search(answer))


def is_safe_answer(answer: str) -> bool:
    """
    Validasi minimum sebelum jawaban ditampilkan kepada mahasiswa.

    Ini bukan pengganti groundedness evaluator. Ini hanya circuit breaker
    untuk dua kondisi yang paling berbahaya di public mode:

    1. provider mengembalikan pesan error sebagai teks jawaban;
    2. gateway mengabaikan prompt dan mengembalikan teks umum tanpa sitasi.
    """
    if not answer or not answer.strip():
        return False
    lowered = answer.lower()
    if any(marker.lower() in lowered for marker in _ERROR_MARKERS):
        return False
    return has_page_citation(answer)
