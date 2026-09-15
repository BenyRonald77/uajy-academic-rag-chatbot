"""
chunking.py — Membagi teks dokumen menjadi chunk yang bermakna, dengan
nomor halaman yang presisi.

Tiga masalah yang diperbaiki dibanding versi awal:

1. **Nomor halaman kabur.** Versi sebelumnya mengumpulkan seluruh halaman
   sebuah section, lalu memberikan daftar halaman yang sama itu ke SEMUA
   chunk di dalamnya. Sebuah section yang membentang 13 halaman membuat
   setiap chunk-nya mengklaim "halaman 80-92", padahal isinya hanya satu
   halaman. Sitasi yang tidak presisi merusak nilai utama chatbot ini.
   Sekarang nomor halaman dilacak per baris, sehingga sebuah chunk hanya
   mengklaim halaman yang benar-benar menjadi sumber teksnya.

2. **Batas paragraf tidak pernah terdeteksi.** `extract_text` membuang semua
   baris kosong, jadi pemisahan berbasis `\\n\\n` selalu gagal dan seluruh
   section terpaksa dipotong per kalimat. Modul ini merekonstruksi batas
   paragraf dari sinyal tata letak (panjang baris, tanda baca akhir, penanda
   daftar) dan menyambung kembali baris yang terpotong akibat pembungkusan
   PDF.

3. **Judul bagian hilang dari teks chunk.** Baris heading dulu dipakai
   sebagai judul saja lalu dibuang dari isi. Akibatnya embedding chunk
   kehilangan sinyal semantik terkuatnya. Sekarang heading disimpan sebagai
   hierarki (`heading_path`) dan disertakan pada teks yang diembed.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass, field

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_SIZE
from app.text_utils import INDONESIAN_STOPWORDS

#: Batas atas jumlah halaman yang boleh diklaim satu chunk. Pengaman untuk
#: halaman yang teksnya sangat sedikit (halaman tabel, halaman pembatas bab):
#: tanpa batas ini satu chunk bisa menelan banyak halaman sekaligus dan
#: sitasinya kembali kabur.
MAX_PAGES_PER_CHUNK = 3

#: Chunk yang lebih kecil dari porsi ini masih boleh digabung dengan bagian
#: berjudul berbeda, supaya "Pasal" yang sangat pendek tidak berdiri sendiri
#: sebagai chunk kerdil.
SMALL_CHUNK_MERGE_RATIO = 0.4

#: Ambang penyaring noise tingkat chunk. Prosa Bahasa Indonesia yang wajar
#: selalu memuat kata fungsi ("dan", "yang", "untuk"), sedangkan sisa teks
#: rusak hasil rotasi PDF tidak pernah memuatnya. Tabel memang minim kata
#: fungsi, karena itu chunk berangka atau bertanda baca tetap diloloskan.
NOISE_MAX_STOPWORD_RATIO = 0.02
NOISE_MAX_DIGIT_RATIO = 0.08


# ──────────────────────────────────────────────
# Deteksi heading
# ──────────────────────────────────────────────

#: Pola heading beserta level hierarkinya. Level kecil berarti lebih tinggi.
#: Heading "struktural" (BAB/BAGIAN/LAMPIRAN) diberi tanda karena judulnya
#: sering terbelah ke baris berikutnya.
_HEADING_PATTERNS: list[tuple[int, bool, re.Pattern]] = [
    (0, True, re.compile(r"^(?:BAB|Bab|BAGIAN|Bagian)\s+[IVXLCDM\d]+\b")),
    (0, True, re.compile(r"^(?:LAMPIRAN|Lampiran)\b")),
    (1, False, re.compile(r"^(?:PASAL|Pasal)\s+\d+\b")),
    (2, False, re.compile(r"^\d+\.\d+(?:\.\d+)?\s+\S")),
    (2, False, re.compile(r"^[A-Z]\.\s+[A-Z]")),
]

#: Baris huruf kapital semua dianggap judul. Dipakai terpisah karena sering
#: menjadi kelanjutan judul BAB pada baris sebelumnya ("BAB I" lalu
#: "PENDAHULUAN"), bukan heading baru.
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z\s\-–—/&(),.']{5,}$")

#: Minimum jumlah huruf agar baris huruf kapital dianggap judul.
#:
#: Tanpa syarat ini, potongan akhir kalimat yang kebetulan berhuruf kapital
#: ikut terangkat menjadi judul tingkat atas. Pada dokumen ini pengukuran
#: menemukan "UAJY)." mencemari 59 chunk, "PKKMB)." 8 chunk, dan "AB-PUI"
#: 1 chunk — totalnya 31% dari seluruh chunk menampilkan label bagian yang
#: tidak bermakna pada sitasinya.
#:
#: Ambang ini HANYA berlaku untuk aturan huruf kapital. Heading struktural
#: seperti "BAB I" hanya punya empat huruf tetapi sudah dikenali lewat pola
#: tersendiri, jadi tidak boleh ikut tersaring.
_MIN_ALL_CAPS_HEADING_LETTERS = 6

#: Baris kepala tabel yang dimulai kolom penomoran ("NO." / "NO URUT").
#: Pada dokumen ini satu baris semacam itu sempat menjadi akar heading bagi
#: 94 chunk, karena judul tingkat atas mewarisi seluruh chunk sesudahnya.
_TABLE_HEADER_RE = re.compile(r"^NO\.?\s", re.IGNORECASE)


def _is_all_caps_heading(line: str) -> bool:
    """
    Tentukan apakah baris huruf kapital layak dianggap judul bagian.

    Aturan huruf kapital bersifat menentukan pada dokumen ini: hampir seluruh
    judul bagiannya memakai huruf kapital, bukan penanda "BAB". Karena itu
    setiap salah tebak berdampak luas — judul tingkat atas menjadi akar
    hierarki bagi semua chunk sesudahnya sampai judul berikutnya muncul.

    Tiga bentuk bukan-judul yang tersaring di sini, semuanya ditemukan lewat
    pengukuran pada dokumen sungguhan:

    - **Pecahan kalimat.** Sisa dari "... (UAJY)." menyisakan baris "UAJY).",
      pendek dan berkurung tutup tanpa pembuka.
    - **Baris kepala tabel.** "NO. FAKULTAS KONSENTRASI / PEMINATAN STATUS"
      sempat menjadi akar heading bagi 94 chunk sekaligus.
    - **Judul terpotong.** Baris yang berakhir dengan "/" atau "-" jelas masih
      bersambung ke baris berikutnya.
    """
    if not _ALL_CAPS_RE.match(line):
        return False

    if sum(1 for c in line if c.isalpha()) < _MIN_ALL_CAPS_HEADING_LETTERS:
        return False

    # Tanda kurung tutup tanpa pembuka menandakan baris ini pecahan kalimat.
    if line.count(")") != line.count("("):
        return False

    # Kolom penomoran adalah penanda paling khas baris kepala tabel.
    if _TABLE_HEADER_RE.match(line):
        return False

    # Judul yang berakhir dengan pemisah masih terpotong, belum utuh.
    if line.rstrip().endswith(("/", "-", "–", "—", ",")):
        return False

    return True

#: Penanda awal butir daftar.
_BULLET_RE = re.compile(r"^(?:[-•*▪o]\s+|\(?\d{1,2}[.)]\s+|[a-z][.)]\s+)")

#: Tanda baca yang menandai akhir kalimat/paragraf.
_SENTENCE_END = (".", "!", "?", ":", ";")

_MAX_HEADING_LENGTH = 120


@dataclass
class Chunk:
    """Satu chunk teks beserta metadata sumbernya."""

    text: str
    page_numbers: list[int] = field(default_factory=list)
    section_title: str = ""
    heading_path: list[str] = field(default_factory=list)
    chunk_index: int = 0

    #: Nama berkas dokumen asal. Kosong berarti index dokumen tunggal versi
    #: lama. Nomor halaman hanya bermakna dalam konteks dokumennya, jadi
    #: sitasi wajib menyertakan asal dokumen begitu ada lebih dari satu.
    source_document: str = ""

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def estimated_tokens(self) -> int:
        """Estimasi kasar jumlah token (1 token ≈ 4 karakter untuk Bahasa Indonesia)."""
        return len(self.text) // 4

    @property
    def page_start(self) -> int | None:
        return min(self.page_numbers) if self.page_numbers else None

    @property
    def page_end(self) -> int | None:
        return max(self.page_numbers) if self.page_numbers else None

    @property
    def page_label(self) -> str:
        """Rentang halaman ringkas, mis. "12" atau "12-14"."""
        if not self.page_numbers:
            return "-"
        start, end = self.page_start, self.page_end
        return str(start) if start == end else f"{start}-{end}"

    @property
    def embed_text(self) -> str:
        """
        Teks yang dikirim ke model embedding.

        Heading path disisipkan di depan isi supaya vektor chunk membawa
        konteks bagiannya. Tanpa ini, potongan di tengah "BAB IV Kurikulum"
        tidak punya petunjuk apa pun bahwa ia membahas kurikulum.
        """
        if not self.heading_path:
            return self.text
        return f"{' > '.join(self.heading_path)}\n\n{self.text}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update({
            "char_count": self.char_count,
            "estimated_tokens": self.estimated_tokens,
            "page_start": self.page_start,
            "page_end": self.page_end,
        })
        return data


@dataclass
class _Line:
    """Satu baris teks beserta halaman asalnya."""

    page: int
    text: str


@dataclass
class _Block:
    """
    Satu paragraf (atau butir daftar) hasil rekonstruksi tata letak.

    Menyimpan baris penyusunnya, bukan hanya teks gabungan, supaya blok yang
    kelewat panjang bisa dipecah tanpa kehilangan ketepatan nomor halaman.
    """

    lines: list[_Line]
    heading_path: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return _join_lines([line.text for line in self.lines])

    @property
    def pages(self) -> list[int]:
        return sorted({line.page for line in self.lines})

    @property
    def char_count(self) -> int:
        return len(self.text)


# ──────────────────────────────────────────────
# Penyambungan baris
# ──────────────────────────────────────────────

def _join_lines(lines: list[str]) -> str:
    """
    Sambung baris yang terpotong akibat pembungkusan PDF menjadi teks utuh.

    Kata yang terpotong tanda hubung di akhir baris ("univer-\\nsitas")
    disatukan kembali tanpa spasi.
    """
    if not lines:
        return ""

    result = lines[0].strip()
    for raw in lines[1:]:
        line = raw.strip()
        if not line:
            continue
        if result.endswith("-") and line[:1].islower():
            result = result[:-1] + line
        else:
            result = f"{result} {line}"
    return result


def _detect_heading(line: str) -> tuple[int, str, bool] | None:
    """
    Deteksi heading pada sebuah baris.

    Returns:
        ``(level, judul, is_structural)`` bila baris merupakan heading, atau
        ``None``. Level 0 adalah yang tertinggi (BAB/Lampiran).
        ``is_structural`` menandai heading BAB/BAGIAN/LAMPIRAN, yang judulnya
        boleh disambung dari baris huruf kapital berikutnya.
    """
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > _MAX_HEADING_LENGTH:
        return None

    for level, is_structural, pattern in _HEADING_PATTERNS:
        if pattern.match(stripped):
            return level, stripped, is_structural

    if _is_all_caps_heading(stripped):
        return 0, stripped, False

    return None


def _starts_new_block(previous: str | None, current: str, short_line_limit: float) -> bool:
    """
    Tentukan apakah `current` memulai paragraf baru.

    `extract_text` sudah membuang baris kosong, jadi batas paragraf harus
    disimpulkan dari tata letak:

    - Butir daftar selalu memulai blok baru.
    - Baris sebelumnya diakhiri tanda baca penutup kalimat.
    - Baris sebelumnya jauh lebih pendek dari lebar baris umumnya, yang pada
      teks rata kanan-kiri berarti baris terakhir sebuah paragraf.
    """
    if previous is None:
        return True

    if _BULLET_RE.match(current):
        return True

    previous = previous.rstrip()
    if previous.endswith(_SENTENCE_END):
        return True
    if len(previous) < short_line_limit:
        return True

    return False


def _to_lines(pages: list[tuple[int, str]]) -> list[_Line]:
    """Ratakan daftar halaman menjadi daftar baris yang membawa nomor halaman."""
    lines: list[_Line] = []
    for page_number, text in pages:
        for raw in text.split("\n"):
            stripped = raw.strip()
            if stripped:
                lines.append(_Line(page=page_number, text=stripped))
    return lines


def _short_line_limit(lines: list[_Line]) -> float:
    """
    Ambang "baris pendek", diturunkan dari lebar baris dokumen itu sendiri.

    Dokumen berbeda punya lebar kolom berbeda, jadi angka absolut tidak
    dapat diandalkan. Median panjang baris memberi patokan yang menyesuaikan
    diri.
    """
    lengths = [len(line.text) for line in lines if len(line.text) > 20]
    if not lengths:
        return 40.0
    return statistics.median(lengths) * 0.6


def _build_blocks(lines: list[_Line]) -> list[_Block]:
    """
    Kelompokkan baris menjadi blok paragraf sambil melacak hierarki heading.

    Heading tetap disertakan sebagai blok tersendiri agar teksnya ikut
    terindeks, bukan sekadar menjadi label.
    """
    short_limit = _short_line_limit(lines)

    blocks: list[_Block] = []
    heading_stack: list[str] = []
    current: list[_Line] = []
    previous_text: str | None = None

    #: Blok heading struktural terakhir, kandidat penyambungan judul.
    pending_structural: _Block | None = None

    def flush() -> None:
        nonlocal current, previous_text
        if current:
            blocks.append(_Block(lines=current, heading_path=tuple(heading_stack)))
            current = []
        previous_text = None

    for line in lines:
        heading = _detect_heading(line.text)

        if heading:
            level, title, is_structural = heading

            # "BAB I" pada satu baris lalu "PENDAHULUAN" pada baris berikutnya
            # adalah satu judul yang terbelah tata letak. Sambung ke blok
            # heading sebelumnya, jangan buat heading baru.
            if (
                pending_structural is not None
                and not is_structural
                and _ALL_CAPS_RE.match(line.text)
                and heading_stack
            ):
                heading_stack[-1] = f"{heading_stack[-1]} {title}"
                pending_structural.lines.append(line)
                pending_structural.heading_path = tuple(heading_stack)
                pending_structural = None
                continue

            flush()
            del heading_stack[level:]
            heading_stack.append(title)
            heading_block = _Block(lines=[line], heading_path=tuple(heading_stack))
            blocks.append(heading_block)
            pending_structural = heading_block if is_structural else None
            continue

        pending_structural = None

        if _starts_new_block(previous_text, line.text, short_limit):
            flush()

        current.append(line)
        previous_text = line.text

    flush()
    return blocks


# ──────────────────────────────────────────────
# Pengemasan blok menjadi chunk
# ──────────────────────────────────────────────

def _page_span(blocks: list[_Block]) -> int:
    """Jumlah halaman berbeda yang dicakup sekumpulan blok."""
    pages: set[int] = set()
    for block in blocks:
        pages.update(block.pages)
    return len(pages)


def _blocks_to_chunk(blocks: list[_Block]) -> Chunk | None:
    """Gabungkan blok-blok menjadi satu Chunk."""
    if not blocks:
        return None

    text = "\n\n".join(block.text for block in blocks if block.text.strip())
    if not text.strip():
        return None

    pages: set[int] = set()
    for block in blocks:
        pages.update(block.pages)

    heading_path = list(blocks[0].heading_path)
    return Chunk(
        text=text.strip(),
        page_numbers=sorted(pages),
        section_title=heading_path[-1] if heading_path else "",
        heading_path=heading_path,
    )


def _split_oversized_block(block: _Block, chunk_size: int) -> list[_Block]:
    """
    Pecah blok yang lebih panjang dari `chunk_size`.

    Pemecahan dilakukan pada batas baris, bukan karakter, sehingga setiap
    pecahan tetap tahu halaman asalnya. Baris tunggal yang masih terlalu
    panjang dipecah lagi per kalimat, mewarisi halaman baris induknya.
    """
    pieces: list[_Block] = []
    current: list[_Line] = []
    current_length = 0

    def flush() -> None:
        nonlocal current, current_length
        if current:
            pieces.append(_Block(lines=current, heading_path=block.heading_path))
            current = []
            current_length = 0

    for line in block.lines:
        line_length = len(line.text) + 1

        if line_length > chunk_size:
            flush()
            for sentence in _split_line_into_sentences(line, chunk_size):
                pieces.append(_Block(lines=[sentence], heading_path=block.heading_path))
            continue

        if current and current_length + line_length > chunk_size:
            flush()

        current.append(line)
        current_length += line_length

    flush()
    return pieces


def _split_line_into_sentences(line: _Line, chunk_size: int) -> list[_Line]:
    """Pecah satu baris yang sangat panjang menjadi beberapa baris per kalimat."""
    sentences = re.split(r"(?<=[.!?;])\s+", line.text)
    result: list[_Line] = []
    buffer = ""

    for sentence in sentences:
        if buffer and len(buffer) + len(sentence) + 1 > chunk_size:
            result.append(_Line(page=line.page, text=buffer.strip()))
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()

    if buffer.strip():
        result.append(_Line(page=line.page, text=buffer.strip()))

    return result or [line]


def _overlap_blocks(blocks: list[_Block], overlap: int) -> list[_Block]:
    """
    Pilih blok terakhir dari sebuah chunk untuk dibawa ke chunk berikutnya.

    Overlap dilakukan pada tingkat blok, bukan potongan karakter mentah,
    supaya teks yang dibawa tetap merupakan paragraf utuh dan halaman yang
    diklaim chunk berikutnya tetap benar. Blok heading tidak dihitung sebagai
    overlap karena akan tetap muncul lewat `heading_path`.
    """
    if overlap <= 0 or len(blocks) <= 1:
        return []

    carried: list[_Block] = []
    total = 0
    for block in reversed(blocks[1:]):
        length = block.char_count
        if total + length > overlap:
            break
        carried.insert(0, block)
        total += length

    return carried


def _pack_blocks(
    blocks: list[_Block],
    chunk_size: int,
    chunk_overlap: int,
    max_pages_per_chunk: int,
) -> list[Chunk]:
    """
    Susun blok menjadi chunk secara berurutan.

    Sebuah chunk ditutup ketika salah satu terjadi:

    - Menambah blok berikutnya melewati `chunk_size`.
    - Menambah blok berikutnya membuat rentang halaman melewati
      `max_pages_per_chunk` (penjaga presisi sitasi).
    - Judul bagian berganti, kecuali chunk saat ini masih terlalu kecil untuk
      berdiri sendiri.
    """
    chunks: list[Chunk] = []
    current: list[_Block] = []
    current_length = 0
    small_chunk_limit = chunk_size * SMALL_CHUNK_MERGE_RATIO

    def flush(carry_overlap: bool = True) -> None:
        nonlocal current, current_length
        chunk = _blocks_to_chunk(current)
        if chunk:
            chunks.append(chunk)

        carried = _overlap_blocks(current, chunk_overlap) if carry_overlap else []

        # Batas rentang halaman harus berlaku pada blok overlap juga. Tanpa ini
        # overlap terus menyeret halaman lama ke chunk berikutnya: setiap kali
        # rentang terlampaui chunk ditutup, tapi blok yang dibawa langsung
        # memasukkan kembali halaman-halaman itu, sehingga satu chunk bisa
        # mengklaim tujuh halaman meski batasnya tiga.
        while carried and _page_span(carried) >= max_pages_per_chunk:
            carried.pop(0)

        current = list(carried)
        current_length = sum(block.char_count for block in current)

    for block in blocks:
        if block.char_count > chunk_size:
            flush()
            for piece in _split_oversized_block(block, chunk_size):
                chunk = _blocks_to_chunk([piece])
                if chunk:
                    chunks.append(chunk)
            current, current_length = [], 0
            continue

        if current:
            heading_changed = block.heading_path != current[0].heading_path
            too_long = current_length + block.char_count > chunk_size
            too_many_pages = _page_span(current + [block]) > max_pages_per_chunk

            if too_long or too_many_pages:
                flush()
            elif heading_changed and current_length >= small_chunk_limit:
                # Jangan biarkan satu chunk melintasi batas bagian, kecuali
                # bagian saat ini memang terlalu pendek untuk berdiri sendiri.
                flush(carry_overlap=False)

        current.append(block)
        current_length += block.char_count

    flush(carry_overlap=False)
    return chunks


# ──────────────────────────────────────────────
# Penyaring noise tingkat chunk
# ──────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z]+")


def looks_like_extraction_noise(text: str) -> bool:
    """
    Deteksi chunk yang isinya sisa teks rusak hasil ekstraksi PDF.

    Penyaring di `extract_text` bekerja per baris dan menangkap sebagian
    besar noise, tetapi teks terbalik dari diagram struktur organisasi
    ("arudtakpu", "satlukasf") lolos karena secara statistik huruf-per-huruf
    masih tampak wajar.

    Pada tingkat chunk ada sinyal yang jauh lebih kuat: **prosa Bahasa
    Indonesia yang sah selalu memuat kata fungsi.** Teks acak tidak pernah.
    Agar tabel tidak ikut terbuang, chunk yang kaya angka atau memuat tanda
    baca akhir kalimat tetap diloloskan.

    Returns:
        True bila chunk sebaiknya dibuang.
    """
    words = _WORD_RE.findall(text.lower())
    if not words:
        return True

    stopword_ratio = sum(1 for word in words if word in INDONESIAN_STOPWORDS) / len(words)
    if stopword_ratio > NOISE_MAX_STOPWORD_RATIO:
        return False

    # Tabel dikenali dari kepadatan angkanya.
    digit_ratio = sum(1 for char in text if char.isdigit()) / len(text)
    if digit_ratio >= NOISE_MAX_DIGIT_RATIO:
        return False

    # Kalimat sungguhan berakhir dengan tanda baca.
    if any(mark in text for mark in (".", ":", ";", "?", "!")):
        return False

    return True


# ──────────────────────────────────────────────
# API publik
# ──────────────────────────────────────────────

def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE,
    max_pages_per_chunk: int = MAX_PAGES_PER_CHUNK,
    drop_noise: bool = True,
    verbose: bool = True,
) -> list[Chunk]:
    """
    Bagi halaman-halaman hasil ekstraksi menjadi chunk yang siap diembed.

    Args:
        pages: Daftar ``(nomor_halaman, teks)`` dari `extract_text`.
        chunk_size: Batas karakter per chunk.
        chunk_overlap: Karakter yang dibawa ke chunk berikutnya, dibulatkan
            ke batas paragraf.
        min_chunk_size: Chunk lebih kecil dari ini dibuang.
        max_pages_per_chunk: Batas halaman yang boleh diklaim satu chunk.
        drop_noise: Buang chunk yang isinya sisa teks rusak.
        verbose: Cetak statistik hasil chunking.

    Returns:
        Daftar `Chunk` dengan `chunk_index` berurutan.
    """
    lines = _to_lines(pages)
    if not lines:
        return []

    blocks = _build_blocks(lines)
    chunks = _pack_blocks(blocks, chunk_size, chunk_overlap, max_pages_per_chunk)

    total_packed = len(chunks)
    chunks = [c for c in chunks if c.char_count >= min_chunk_size]
    dropped_small = total_packed - len(chunks)

    dropped_noise = 0
    if drop_noise:
        before = len(chunks)
        chunks = [c for c in chunks if not looks_like_extraction_noise(c.text)]
        dropped_noise = before - len(chunks)

    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i

    if verbose:
        print(f"✅ Dihasilkan {len(chunks)} chunk dari {len(pages)} halaman "
              f"({len(blocks)} blok paragraf)")
        if dropped_small or dropped_noise:
            print(f"   🗑️  Dibuang: {dropped_small} chunk terlalu kecil, "
                  f"{dropped_noise} chunk teks rusak")
        print_chunk_stats(chunks)

    return chunks


def print_chunk_stats(chunks: list[Chunk]) -> None:
    """Cetak statistik chunk, termasuk kualitas presisi halaman."""
    if not chunks:
        print("⚠️  Tidak ada chunk yang dihasilkan!")
        return

    sizes = [c.char_count for c in chunks]
    tokens = [c.estimated_tokens for c in chunks]
    spans = [len(c.page_numbers) for c in chunks]
    single_page = sum(1 for s in spans if s == 1)
    with_heading = sum(1 for c in chunks if c.heading_path)

    print("   📊 Statistik chunk:")
    print(f"      Jumlah        : {len(chunks)}")
    print(f"      Karakter      : min={min(sizes)}, max={max(sizes)}, "
          f"avg={sum(sizes) // len(sizes)}")
    print(f"      Token est.    : min={min(tokens)}, max={max(tokens)}, "
          f"avg={sum(tokens) // len(tokens)}")
    print("   🎯 Presisi halaman:")
    print(f"      Halaman/chunk : max={max(spans)}, "
          f"avg={sum(spans) / len(spans):.2f}")
    print(f"      Satu halaman  : {single_page}/{len(chunks)} "
          f"({single_page / len(chunks):.0%})")
    print(f"      Ada heading   : {with_heading}/{len(chunks)} "
          f"({with_heading / len(chunks):.0%})")


if __name__ == "__main__":
    test_pages = [
        (1, "BAB I\nPENDAHULUAN\n"
            "Buku pedoman akademik ini disusun sebagai acuan resmi bagi seluruh\n"
            "mahasiswa Fakultas Teknologi Industri dalam menjalani proses\n"
            "pembelajaran.\n"
            "Pedoman ini mencakup ketentuan umum, kurikulum, serta prosedur\n"
            "administrasi akademik."),
        (2, "BAB II\nKETENTUAN UMUM\n"
            "Pasal 1\nPedoman ini berlaku untuk seluruh mahasiswa aktif.\n"
            "Pasal 2\nMahasiswa wajib mengikuti seluruh peraturan yang berlaku di\n"
            "lingkungan universitas."),
    ]

    for chunk in chunk_pages(test_pages, chunk_size=400, chunk_overlap=80, min_chunk_size=20):
        print(f"\n--- Chunk {chunk.chunk_index} (halaman {chunk.page_label}) ---")
        print(f"Heading : {' > '.join(chunk.heading_path) or '-'}")
        print(f"Teks ({chunk.char_count} chars): {chunk.text[:220]}")
