"""Utilitas sitasi yang dipakai runtime UI dan evaluator."""

from __future__ import annotations

import re

_CITATION_START_RE = re.compile(r"halaman\s*:?\s*(?=\d)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d{1,4}")
_SEPARATOR_RE = re.compile(r"\s*(,|dan|s\.?d\.?|[-–—])\s*", re.IGNORECASE)
_MAX_RANGE_SPAN = 40


def extract_cited_pages(answer: str) -> list[int]:
    """
    Ambil nomor halaman dari sitasi jawaban.

    Mendukung "Halaman 48", "Halaman 39-41", dan "Halaman 39, 40 dan 41".
    Dash yang diikuti angka lebih kecil dianggap pemisah judul, bukan rentang;
    ini mencegah `Halaman 43 — 5. Beban studi` dibaca sebagai halaman 5.
    """
    if not answer:
        return []

    pages: set[int] = set()
    for match in _CITATION_START_RE.finditer(answer):
        position = match.end()
        first = _NUMBER_RE.match(answer, position)
        if not first:
            continue

        current = int(first.group())
        pages.add(current)
        position = first.end()

        while True:
            separator = _SEPARATOR_RE.match(answer, position)
            if not separator:
                break
            following_match = _NUMBER_RE.match(answer, separator.end())
            if not following_match:
                break

            following = int(following_match.group())
            is_dash = separator.group(1) in {"-", "–", "—"}
            if is_dash:
                if following <= current or following - current > _MAX_RANGE_SPAN:
                    break
                pages.update(range(current, following + 1))
            else:
                pages.add(following)

            current = following
            position = following_match.end()

    return sorted(pages)


def citation_pages_are_in_context(answer: str, contexts) -> bool:
    """Pastikan setiap halaman yang dikutip berasal dari konteks retrieval."""
    cited = set(extract_cited_pages(answer))
    context_pages = {page for context in contexts for page in context.page_numbers}
    return bool(cited) and cited <= context_pages


def strip_inline_source_lines(answer: str) -> str:
    """
    Hapus baris sumber yang dibuat LLM dari tampilan utama.

    UI menampilkan sumber dalam satu kartu terstruktur di bawah jawaban.
    Menghapus baris inline mencegah sumber muncul dua kali, seperti pada
    tampilan lama: daftar halaman dari LLM lalu kartu "Sumber Referensi" lagi.
    """
    if not answer:
        return answer

    kept: list[str] = []
    for line in answer.splitlines():
        normalized = line.strip().lower()
        is_source_line = (
            "sumber:" in normalized
            and "halaman" in normalized
        )
        if not is_source_line:
            kept.append(line)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
