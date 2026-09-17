"""Pengaman minimal agar respons provider yang tidak valid tidak tampil ke user."""

from __future__ import annotations

import re

from app.citations import extract_cited_pages

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


def diagnose_answer(answer: str, context_pages: set[int]) -> tuple[str, str] | None:
    """Return a safe diagnostic code/message when a generated answer is blocked.

    Never include the raw provider response here: error bodies and model output
    can contain user or configuration data. The returned detail only contains
    a category and, for citation mismatches, page numbers.
    """
    if not answer or not answer.strip():
        return "empty_response", "Provider mengembalikan jawaban kosong."

    lowered = answer.lower()
    if any(marker.lower() in lowered for marker in _ERROR_MARKERS):
        if any(token in lowered for token in (
            "quota", "rate limit", "batas penggunaan", "429", "resource_exhausted"
        )):
            return "provider_rate_limit", "Provider menolak permintaan karena kuota atau rate limit (indikasi HTTP 429)."
        if any(token in lowered for token in ("api key", "api_key", "authentication", "401")):
            return "provider_auth", "Provider menolak autentikasi (indikasi API key atau HTTP 401)."
        if any(token in lowered for token in ("404", "not_found", "model not found")):
            return "provider_model_or_endpoint", "Model atau endpoint provider tidak ditemukan (indikasi HTTP 404)."
        if "model_disabled" in lowered:
            return "provider_models_unavailable", "Model provider dinonaktifkan atau stoknya habis; semua model cadangan yang dicoba juga gagal."
        return "provider_error", "Provider mengembalikan error; detail mentah disembunyikan."

    if not has_page_citation(answer):
        return "missing_page_citation", "Jawaban tidak memuat sitasi dengan format 'Halaman N'."

    cited_pages = set(extract_cited_pages(answer))
    invalid_pages = sorted(cited_pages - context_pages)
    if invalid_pages:
        available = ", ".join(map(str, sorted(context_pages))) or "tidak ada"
        invalid = ", ".join(map(str, invalid_pages))
        return (
            "citation_outside_context",
            f"Sitasi halaman {invalid} tidak ada dalam konteks retrieval; halaman tersedia: {available}.",
        )

    return None
