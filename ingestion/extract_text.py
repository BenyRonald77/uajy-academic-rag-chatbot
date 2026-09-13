"""
extract_text.py — Ekstrak teks dari PDF menggunakan pdfplumber, lengkap
dengan penyaringan noise ekstraksi.

Kenapa penyaringan ini penting untuk kualitas retrieval:
    Buku pedoman ini memuat diagram struktur organisasi dan tabel yang
    dirotasi. Pada bagian itu pdfplumber menghasilkan teks rusak: kode glyph
    `(cid:54)`, huruf berulang seperti "UUUNNNIIIVVV", ratusan baris berisi
    satu karakter, dan dua kolom yang terjalin karakter demi karakter.

    Teks rusak tersebut tetap masuk index, ikut memakan kuota embedding, dan
    yang terburuk: bersaing memperebutkan slot top-k saat retrieval. Membuang
    noise di hulu meningkatkan presisi seluruh pipeline dan jauh lebih murah
    daripada menyaringnya di hilir.

Penyaringan dilakukan per baris, bukan per halaman: pengukuran menunjukkan
metrik level halaman salah menuduh halaman yang sebenarnya bersih (halaman
padat tabel nilai) sementara halaman rusak pun masih memuat baris yang sah.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

# ──────────────────────────────────────────────
# Pola noise
# ──────────────────────────────────────────────

#: Kode glyph yang muncul saat font PDF tidak menyertakan peta karakter.
_CID_RE = re.compile(r"\(cid:\d+\)")

#: Huruf yang sama berulang tiga kali, ciri teks bertumpuk hasil rotasi
#: ("UUUNNNIIIVVVEEERRRSSS").
_TRIPLE_LETTER_RE = re.compile(r"([A-Za-z])\1\1")

#: Baris yang hanya berisi nomor halaman.
_PAGE_NUMBER_RES = (
    re.compile(r"^\d{1,4}$"),
    re.compile(r"^(halaman|page|hal\.?)\s*\d+$", re.IGNORECASE),
    re.compile(r"^[-–—]\s*\d+\s*[-–—]$"),
    re.compile(r"^[ivxlcdm]{1,7}$", re.IGNORECASE),
)

_WORD_RE = re.compile(r"[A-Za-z]+")
_VOWELS = frozenset("aeiou")

#: Digraf Bahasa Indonesia yang secara fonetis berperan sebagai satu konsonan,
#: sehingga tidak boleh dihitung sebagai gugus konsonan tak wajar.
_DIGRAPHS = ("ng", "ny", "sy", "kh", "th", "ch")

#: Minimum karakter alfanumerik agar sebuah baris dianggap membawa informasi.
_MIN_ALNUM_PER_LINE = 3

#: Ambang untuk menuduh sebuah baris sebagai teks rusak.
_GIBBERISH_WORD_RATIO = 0.5
_GIBBERISH_MIN_WORDS = 3
_MAX_CONSONANT_RUN = 3

#: Panjang minimum kata yang ikut dinilai kewajaran gugus konsonannya.
#: Kata pendek terlalu sering memberi sinyal palsu.
_GIBBERISH_MIN_WORD_LENGTH = 4

#: Baris yang polanya muncul di setidaknya porsi halaman ini dianggap
#: header/footer berjalan dan dibuang.
_HEADER_FOOTER_PAGE_RATIO = 0.10
_HEADER_FOOTER_MIN_PAGES = 5
_HEADER_FOOTER_MAX_LENGTH = 90


@dataclass
class ExtractionReport:
    """Ringkasan kualitas ekstraksi, untuk ditampilkan dan diaudit."""

    pages_with_text: int = 0
    pages_empty_after_cleaning: list[int] = field(default_factory=list)
    lines_raw: int = 0
    lines_kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    header_footer_templates: list[str] = field(default_factory=list)
    noisy_pages: list[tuple[int, float]] = field(default_factory=list)

    @property
    def lines_dropped(self) -> int:
        return self.lines_raw - self.lines_kept

    def summary_lines(self) -> list[str]:
        """Ringkasan siap cetak."""
        if not self.lines_raw:
            return ["   ⚠️  Tidak ada baris teks yang terbaca."]

        ratio = self.lines_dropped / self.lines_raw
        lines = [
            f"   🧹 Baris disaring: {self.lines_dropped}/{self.lines_raw} ({ratio:.0%})",
        ]
        for reason, count in self.dropped.most_common():
            lines.append(f"      • {reason}: {count}")
        if self.header_footer_templates:
            preview = ", ".join(repr(t) for t in self.header_footer_templates[:3])
            lines.append(f"      • pola header/footer terdeteksi: {preview}")
        if self.noisy_pages:
            worst = ", ".join(f"hal.{p} ({r:.0%})" for p, r in self.noisy_pages[:6])
            lines.append(f"   ⚠️  Halaman dengan noise tinggi: {worst}")
        if self.pages_empty_after_cleaning:
            lines.append(
                f"   ⚠️  Halaman kosong setelah penyaringan: "
                f"{self.pages_empty_after_cleaning}"
            )
        return lines


# ──────────────────────────────────────────────
# Pendeteksi noise per baris
# ──────────────────────────────────────────────

def _longest_consonant_run(word: str) -> int:
    """
    Panjang gugus konsonan terpanjang dalam sebuah kata.

    Bahasa Indonesia hampir tidak pernah memiliki tiga konsonan berurutan di
    luar digraf, jadi angka tinggi menandakan teks rusak ("lratsrutkurts").
    """
    lowered = word.lower()
    for digraph in _DIGRAPHS:
        lowered = lowered.replace(digraph, "c")

    longest = run = 0
    for char in lowered:
        if char not in _VOWELS:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def _is_page_number(line: str) -> bool:
    return any(pattern.match(line) for pattern in _PAGE_NUMBER_RES)


def _is_gibberish(line: str) -> bool:
    """
    Tandai baris yang hampir pasti hasil ekstraksi rusak.

    Dua sinyal dipakai: huruf berulang tiga kali (teks bertumpuk) dan
    proporsi kata dengan gugus konsonan tak wajar (teks terbalik/terjalin).

    Singkatan huruf kapital dan kata pendek dikecualikan dari penilaian.
    Tanpa pengecualian ini, tabel kurikulum ikut terbuang: baris seperti
    "TID 1101 Kalkulus I 3 SKS" penuh kode program studi yang secara
    fonetis memang "tidak wajar", padahal isinya sah dan justru sering
    ditanyakan mahasiswa.
    """
    if len(_TRIPLE_LETTER_RE.findall(line)) >= 2:
        return True

    words = [
        word for word in _WORD_RE.findall(line)
        if len(word) >= _GIBBERISH_MIN_WORD_LENGTH and not word.isupper()
    ]
    if len(words) < _GIBBERISH_MIN_WORDS:
        return False

    suspicious = sum(
        1 for word in words if _longest_consonant_run(word) >= _MAX_CONSONANT_RUN
    )
    return suspicious / len(words) >= _GIBBERISH_WORD_RATIO


def _alnum_count(line: str) -> int:
    return sum(1 for char in line if char.isalnum())


def _header_footer_template(line: str) -> str:
    """
    Bentuk baku sebuah baris dengan angka dihilangkan.

    Membuat "Susunan Pengurus xi" dan "Susunan Pengurus xiii" mengerucut ke
    pola yang sama sehingga header berjalan bisa dikenali lewat frekuensi.
    """
    template = re.sub(r"\d+", "#", line.lower())
    template = re.sub(r"\b[ivxlcdm]{1,7}\b", "#", template)
    return re.sub(r"\s+", " ", template).strip()


def _detect_header_footer_templates(
    pages_lines: list[tuple[int, list[str]]],
) -> set[str]:
    """
    Kenali header/footer berjalan dari frekuensi kemunculannya antar halaman.

    Lebih tahan banting daripada daftar pola yang ditulis manual: berlaku
    untuk dokumen apa pun tanpa penyesuaian.
    """
    total_pages = len(pages_lines)
    if total_pages < _HEADER_FOOTER_MIN_PAGES:
        return set()

    counts: Counter[str] = Counter()
    for _, lines in pages_lines:
        # Hanya baris pertama dan terakhir tiap halaman yang dipertimbangkan.
        edges = {line for line in (lines[:2] + lines[-2:]) if line}
        for line in edges:
            if len(line) <= _HEADER_FOOTER_MAX_LENGTH:
                counts[_header_footer_template(line)] += 1

    threshold = max(_HEADER_FOOTER_MIN_PAGES, int(total_pages * _HEADER_FOOTER_PAGE_RATIO))
    return {
        template
        for template, count in counts.items()
        if count >= threshold and template
    }


# ──────────────────────────────────────────────
# Ekstraksi
# ──────────────────────────────────────────────

def extract_text_from_pdf(
    pdf_path: str | Path,
    clean: bool = True,
    verbose: bool = True,
) -> list[tuple[int, str]]:
    """
    Ekstrak teks dari setiap halaman PDF.

    Args:
        pdf_path: Path ke file PDF.
        clean: Aktifkan penyaringan noise ekstraksi.
        verbose: Cetak ringkasan hasil ekstraksi.

    Returns:
        Daftar ``(nomor_halaman, teks)``, nomor halaman dimulai dari 1.
        Halaman yang kosong setelah penyaringan tidak disertakan.
    """
    pages, _ = extract_text_with_report(pdf_path, clean=clean, verbose=verbose)
    return pages


def extract_text_with_report(
    pdf_path: str | Path,
    clean: bool = True,
    verbose: bool = True,
) -> tuple[list[tuple[int, str]], ExtractionReport]:
    """
    Sama dengan `extract_text_from_pdf`, tetapi juga mengembalikan laporan
    kualitas ekstraksi.

    Returns:
        Tuple ``(halaman, laporan)``.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File PDF tidak ditemukan: {pdf_path}")

    report = ExtractionReport()
    raw_pages: list[tuple[int, list[str]]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            report.lines_raw += len(lines)
            if lines:
                raw_pages.append((i + 1, lines))

    if not clean:
        pages = [(number, "\n".join(lines)) for number, lines in raw_pages]
        report.lines_kept = report.lines_raw
        report.pages_with_text = len(pages)
        if verbose:
            print(f"✅ Berhasil mengekstrak {len(pages)} halaman dari '{pdf_path.name}' "
                  "(tanpa penyaringan)")
        return pages, report

    header_footer = _detect_header_footer_templates(raw_pages)
    report.header_footer_templates = sorted(header_footer)

    pages: list[tuple[int, str]] = []
    for page_number, lines in raw_pages:
        kept = _filter_lines(lines, header_footer, report)
        report.lines_kept += len(kept)

        if lines:
            noise_ratio = 1 - (len(kept) / len(lines))
            if noise_ratio >= 0.5 and len(lines) >= 20:
                report.noisy_pages.append((page_number, noise_ratio))

        if kept:
            pages.append((page_number, "\n".join(kept)))
        else:
            report.pages_empty_after_cleaning.append(page_number)

    report.pages_with_text = len(pages)
    report.noisy_pages.sort(key=lambda item: item[1], reverse=True)

    if verbose:
        print(f"✅ Berhasil mengekstrak {len(pages)} halaman dari '{pdf_path.name}'")
        for line in report.summary_lines():
            print(line)

    return pages, report


def _filter_lines(
    lines: list[str],
    header_footer: set[str],
    report: ExtractionReport,
) -> list[str]:
    """Terapkan seluruh penyaring noise pada baris-baris satu halaman."""
    kept: list[str] = []
    previous: str | None = None

    for line in lines:
        if _CID_RE.search(line):
            report.dropped["kode glyph (cid:NN)"] += 1
            continue
        if _is_page_number(line):
            report.dropped["nomor halaman"] += 1
            continue
        if _header_footer_template(line) in header_footer:
            report.dropped["header/footer berjalan"] += 1
            continue
        if _alnum_count(line) < _MIN_ALNUM_PER_LINE:
            report.dropped["baris terlalu pendek"] += 1
            continue
        if _is_gibberish(line):
            report.dropped["teks rusak (rotasi/terjalin)"] += 1
            continue

        # Baris identik berurutan muncul saat teks tergambar dua kali.
        normalized = re.sub(r"\s+", " ", line)
        if normalized == previous:
            report.dropped["baris duplikat berurutan"] += 1
            continue

        kept.append(normalized)
        previous = normalized

    return kept


def extract_tables_from_pdf(pdf_path: str | Path, verbose: bool = True) -> list[dict]:
    """
    Ekstrak tabel dari PDF sebagai teks berpembatas pipa.

    Returns:
        Daftar ``{"page": int, "table_index": int, "content": str}``.
    """
    pdf_path = Path(pdf_path)
    tables: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            for j, table in enumerate(page.extract_tables() or []):
                rows = []
                for row in table:
                    cells = [str(cell).strip() if cell else "" for cell in row]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    tables.append({
                        "page": i + 1,
                        "table_index": j,
                        "content": "\n".join(rows),
                    })

    if verbose:
        print(f"✅ Ditemukan {len(tables)} tabel dari '{pdf_path.name}'")
    return tables


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.extract_text <path_to_pdf>")
        sys.exit(1)

    extracted, extraction_report = extract_text_with_report(sys.argv[1])
    print(f"\n📄 Total halaman dengan teks: {len(extracted)}")
    for number, page_text in extracted[:3]:
        print(f"\n--- Halaman {number} (preview 300 karakter) ---")
        print(page_text[:300])
