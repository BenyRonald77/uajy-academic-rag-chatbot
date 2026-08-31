"""
extract_text.py — Ekstrak teks dari PDF menggunakan pdfplumber.

Menghasilkan list of (page_number, text) tuples dengan pembersihan
header/footer berulang dan nomor halaman yang mengganggu.
"""

import re
import pdfplumber
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> list[tuple[int, str]]:
    """
    Ekstrak teks dari setiap halaman PDF.

    Args:
        pdf_path: Path ke file PDF.

    Returns:
        List of (page_number, cleaned_text) tuples.
        page_number dimulai dari 1.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File PDF tidak ditemukan: {pdf_path}")

    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                cleaned = _clean_page_text(text, i + 1)
                if cleaned.strip():
                    pages.append((i + 1, cleaned))

    print(f"✅ Berhasil mengekstrak {len(pages)} halaman dari '{pdf_path.name}'")
    return pages


def _clean_page_text(text: str, page_num: int) -> str:
    """
    Bersihkan teks halaman dari noise umum PDF.

    - Hapus nomor halaman yang berdiri sendiri
    - Hapus baris yang hanya berisi spasi/tab
    - Normalisasi whitespace berlebihan
    - Hapus header/footer berulang yang umum
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip baris kosong
        if not stripped:
            continue

        # Skip baris yang hanya berisi nomor halaman
        if re.match(r"^\d{1,4}$", stripped):
            continue

        # Skip pattern nomor halaman umum: "Halaman 1", "- 1 -", "Page 1"
        if re.match(r"^(halaman|page|hal\.?)\s*\d+$", stripped, re.IGNORECASE):
            continue
        if re.match(r"^[-–—]\s*\d+\s*[-–—]$", stripped):
            continue

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # Normalisasi multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalisasi multiple spasi
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def extract_tables_from_pdf(pdf_path: str) -> list[dict]:
    """
    Ekstrak tabel dari PDF (opsional, untuk dokumen dengan banyak tabel).

    Returns:
        List of dicts: {"page": int, "table_index": int, "content": str}
    """
    pdf_path = Path(pdf_path)
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            if page_tables:
                for j, table in enumerate(page_tables):
                    # Konversi tabel ke string readable
                    rows = []
                    for row in table:
                        cells = [str(cell).strip() if cell else "" for cell in row]
                        rows.append(" | ".join(cells))
                    table_text = "\n".join(rows)

                    tables.append({
                        "page": i + 1,
                        "table_index": j,
                        "content": table_text,
                    })

    print(f"✅ Ditemukan {len(tables)} tabel dari '{pdf_path.name}'")
    return tables


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_text.py <path_to_pdf>")
        print("Contoh: python extract_text.py ../data/dokumen_sumber.pdf")
        sys.exit(1)

    pdf_file = sys.argv[1]
    pages = extract_text_from_pdf(pdf_file)

    print(f"\n📄 Total halaman dengan teks: {len(pages)}")
    for page_num, text in pages[:3]:
        print(f"\n--- Halaman {page_num} (preview 300 karakter) ---")
        print(text[:300])
        print("...")
