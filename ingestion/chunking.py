"""
chunking.py — Membagi teks dokumen menjadi chunk-chunk yang bermakna.

Strategi chunking:
1. Coba split berdasarkan section heading terlebih dahulu.
2. Jika section terlalu panjang, split berdasarkan paragraf.
3. Jika paragraf masih terlalu panjang, split berdasarkan kalimat
   dengan overlap untuk menjaga konteks.
"""

import re
from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    """Representasi satu chunk teks dengan metadata."""
    text: str
    page_numbers: list[int] = field(default_factory=list)
    section_title: str = ""
    chunk_index: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def estimated_tokens(self) -> int:
        """Estimasi kasar jumlah token (1 token ≈ 4 karakter untuk bahasa Indonesia)."""
        return len(self.text) // 4


# ──────────────────────────────────────────────
# Konfigurasi default
# ──────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 1500       # karakter (~375 token)
DEFAULT_CHUNK_OVERLAP = 300     # karakter (~75 token)
MIN_CHUNK_SIZE = 100            # minimum karakter agar chunk bermakna

# Pattern untuk mendeteksi heading/section
HEADING_PATTERNS = [
    r"^(?:BAB|Bab|BAGIAN|Bagian)\s+[IVXLCDM\d]+",     # BAB I, Bagian 2
    r"^(?:Pasal|PASAL)\s+\d+",                          # Pasal 1
    r"^\d+\.\d*\s+[A-Z]",                               # 1. Pendahuluan, 1.1 Sub
    r"^[A-Z][A-Z\s]{5,}$",                              # JUDUL DENGAN HURUF KAPITAL
    r"^(?:Lampiran|LAMPIRAN)\s*",                        # Lampiran
]


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Bagi halaman-halaman teks menjadi chunk-chunk.

    Args:
        pages: List of (page_number, text) dari extract_text.
        chunk_size: Maksimum karakter per chunk.
        chunk_overlap: Jumlah karakter overlap antar chunk.

    Returns:
        List of Chunk objects.
    """
    # Gabungkan halaman menjadi sections berdasarkan heading
    sections = _split_into_sections(pages)

    # Chunk setiap section
    all_chunks = []
    for section in sections:
        section_chunks = _chunk_section(
            text=section["text"],
            page_numbers=section["pages"],
            section_title=section["title"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(section_chunks)

    # Assign index
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i

    # Filter chunk terlalu kecil
    all_chunks = [c for c in all_chunks if c.char_count >= MIN_CHUNK_SIZE]

    print(f"✅ Dihasilkan {len(all_chunks)} chunk dari {len(pages)} halaman")
    _print_chunk_stats(all_chunks)

    return all_chunks


def _split_into_sections(pages: list[tuple[int, str]]) -> list[dict]:
    """
    Split halaman-halaman menjadi sections berdasarkan heading.

    Returns:
        List of {"title": str, "text": str, "pages": list[int]}
    """
    sections = []
    current_section = {
        "title": "Pendahuluan",
        "text": "",
        "pages": [],
    }

    for page_num, text in pages:
        lines = text.split("\n")
        for line in lines:
            heading = _detect_heading(line)
            if heading:
                # Simpan section sebelumnya jika ada isinya
                if current_section["text"].strip():
                    sections.append(current_section)
                # Mulai section baru
                current_section = {
                    "title": heading,
                    "text": "",
                    "pages": [page_num],
                }
            else:
                current_section["text"] += line + "\n"
                if page_num not in current_section["pages"]:
                    current_section["pages"].append(page_num)

    # Simpan section terakhir
    if current_section["text"].strip():
        sections.append(current_section)

    return sections


def _detect_heading(line: str) -> str | None:
    """Deteksi apakah sebuah baris adalah heading/judul section."""
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return None

    for pattern in HEADING_PATTERNS:
        if re.match(pattern, stripped):
            return stripped

    return None


def _chunk_section(
    text: str,
    page_numbers: list[int],
    section_title: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Bagi satu section menjadi chunk-chunk dengan overlap."""
    if len(text) <= chunk_size:
        return [Chunk(
            text=text.strip(),
            page_numbers=page_numbers,
            section_title=section_title,
        )]

    # Split berdasarkan paragraf dulu
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current_text = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Jika menambahkan paragraf ini masih muat
        if len(current_text) + len(para) + 2 <= chunk_size:
            current_text += para + "\n\n"
        else:
            # Simpan chunk saat ini
            if current_text.strip():
                chunks.append(Chunk(
                    text=current_text.strip(),
                    page_numbers=page_numbers,
                    section_title=section_title,
                ))

            # Jika paragraf tunggal lebih besar dari chunk_size, split by sentence
            if len(para) > chunk_size:
                sentence_chunks = _split_long_text(
                    para, page_numbers, section_title,
                    chunk_size, chunk_overlap
                )
                chunks.extend(sentence_chunks)
                current_text = ""
            else:
                # Overlap: ambil akhir chunk sebelumnya
                if current_text and chunk_overlap > 0:
                    overlap_text = current_text.strip()[-chunk_overlap:]
                    current_text = overlap_text + "\n\n" + para + "\n\n"
                else:
                    current_text = para + "\n\n"

    # Sisa terakhir
    if current_text.strip():
        chunks.append(Chunk(
            text=current_text.strip(),
            page_numbers=page_numbers,
            section_title=section_title,
        ))

    return chunks


def _split_long_text(
    text: str,
    page_numbers: list[int],
    section_title: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split teks panjang berdasarkan kalimat dengan overlap."""
    # Split by kalimat (handle tanda baca Indonesia)
    sentences = re.split(r"(?<=[.!?;])\s+", text)
    chunks = []
    current_text = ""

    for sentence in sentences:
        if len(current_text) + len(sentence) + 1 <= chunk_size:
            current_text += sentence + " "
        else:
            if current_text.strip():
                chunks.append(Chunk(
                    text=current_text.strip(),
                    page_numbers=page_numbers,
                    section_title=section_title,
                ))
            # Start new chunk with overlap
            if chunk_overlap > 0 and current_text:
                overlap = current_text.strip()[-chunk_overlap:]
                current_text = overlap + " " + sentence + " "
            else:
                current_text = sentence + " "

    if current_text.strip():
        chunks.append(Chunk(
            text=current_text.strip(),
            page_numbers=page_numbers,
            section_title=section_title,
        ))

    return chunks


def _print_chunk_stats(chunks: list[Chunk]) -> None:
    """Print statistik chunk untuk debugging."""
    if not chunks:
        print("⚠️  Tidak ada chunk yang dihasilkan!")
        return

    sizes = [c.char_count for c in chunks]
    tokens = [c.estimated_tokens for c in chunks]

    print(f"   📊 Statistik chunk:")
    print(f"      Jumlah   : {len(chunks)}")
    print(f"      Karakter : min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")
    print(f"      Token est: min={min(tokens)}, max={max(tokens)}, avg={sum(tokens)//len(tokens)}")


if __name__ == "__main__":
    # Test dengan dummy data
    test_pages = [
        (1, "BAB I\nPENDAHULUAN\n\nIni adalah paragraf pertama dari bab pendahuluan. "
            "Berisi informasi umum tentang pedoman akademik universitas.\n\n"
            "Paragraf kedua menjelaskan ruang lingkup dari pedoman ini."),
        (2, "BAB II\nKETENTUAN UMUM\n\nPasal 1\nPedoman ini berlaku untuk seluruh "
            "mahasiswa aktif.\n\nPasal 2\nMahasiswa wajib mengikuti peraturan yang berlaku."),
    ]

    chunks = chunk_pages(test_pages)
    for chunk in chunks:
        print(f"\n--- Chunk {chunk.chunk_index} (hal. {chunk.page_numbers}) ---")
        print(f"Section: {chunk.section_title}")
        print(f"Text ({chunk.char_count} chars): {chunk.text[:200]}...")
