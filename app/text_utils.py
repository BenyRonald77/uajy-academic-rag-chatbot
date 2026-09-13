"""
text_utils.py — Normalisasi dan tokenisasi teks Bahasa Indonesia.

Dipakai oleh BM25 lexical index dan gate relevansi leksikal. Sengaja tanpa
dependency eksternal (tanpa NLTK/Sastrawi) agar tetap ringan dan CPU-only.

Keputusan desain:

- **Tanpa stemming agresif.** Stemmer Bahasa Indonesia gaya Nazief-Adriani
  sering over-stem istilah akademik (mis. "penilaian" → "nilai" masih aman,
  tapi "kelulusan" → "lulus" vs "pengulangan" → "ulang" tidak konsisten).
  Untuk korpus sekecil satu buku pedoman, presisi lebih penting daripada
  recall, jadi kita hanya membuang klitik yang murni gramatikal.
- **Alias domain.** Dokumen akademik penuh singkatan (SKS, IPK, KRS, TA).
  Query pengguna dan teks dokumen sering memakai bentuk berbeda, jadi kedua
  bentuk dinormalisasi ke kanonikal yang sama.
"""

from __future__ import annotations

import re
import unicodedata

# ──────────────────────────────────────────────
# Stopwords
# ──────────────────────────────────────────────

#: Stopwords Bahasa Indonesia (kata fungsi + kata tanya) plus beberapa kata
#: Inggris umum. Kata tanya ikut dibuang karena hampir setiap pertanyaan
#: pengguna memuatnya, sehingga nilai diskriminatifnya nol.
INDONESIAN_STOPWORDS: frozenset[str] = frozenset("""
ada adalah adanya agar akan aku akhir akhirnya antar antara apa apabila apakah
atas atau ataupun bagaimana bagi bahkan bahwa banyak baru begitu belum benar
berada berapa berikut bersama beserta bila bisa boleh buat bukan cukup dahulu
dalam dan dapat dari daripada dekat demi dengan di dia dll dsb dua entah guna
hal hampir hanya harus hendak hingga ia ialah ini itu jadi jangan jika juga
kalau kami kamu kapan karena ke kecuali kemudian kepada kita ku lagi lain
lalu maka makin mana manakala masih maupun melainkan melalui memang mempunyai
mengapa mengenai menjadi menurut mereka merupakan meski meskipun mungkin nah
namun nya oleh pada padahal paling para pasti per pernah pula pun punya saat
saja sama sambil sampai sana sangat saya se sebab sebagai sebagaimana sebelum
sebuah secara sedang sedangkan segera sehingga sejak sekali sekitar selain
selama seluruh semua semua sendiri seorang seperti sering serta sesuai sesudah
setelah setiap siapa sini situ suatu sudah supaya tadi tak tanpa tapi telah
tentang tentu terhadap termasuk tersebut tetap tetapi tiap tidak turut untuk
walau walaupun yaitu yakni yang
a an and are as at be but by for from has have how in is it its of on or that
the this to was were what when where which who will with
""".split())

#: Alias domain akademik → bentuk kanonikal.
#: Alias multi-kata ("mata kuliah") dinormalisasi sebelum tokenisasi;
#: alias satu kata dipetakan setelah tokenisasi.
_PHRASE_ALIASES: dict[str, str] = {
    "mata kuliah": "matakuliah",
    "tugas akhir": "tugasakhir",
    "kerja praktek": "kerjapraktek",
    "kerja praktik": "kerjapraktek",
    "kartu rencana studi": "krs",
    "kartu hasil studi": "khs",
    "indeks prestasi kumulatif": "ipk",
    "indeks prestasi": "ip",
    "satuan kredit semester": "sks",
    "cum laude": "cumlaude",
    "cumlaude": "cumlaude",
    "uang kuliah tunggal": "ukt",
    "program studi": "programstudi",
    "semester antara": "semesterantara",
    "semester pendek": "semesterantara",
    "ujian tengah semester": "uts",
    "ujian akhir semester": "uas",
    "drop out": "dropout",
}

_TOKEN_ALIASES: dict[str, str] = {
    "matkul": "matakuliah",
    "mk": "matakuliah",
    "ta": "tugasakhir",
    "skripsi": "tugasakhir",
    "kp": "kerjapraktek",
    "prodi": "programstudi",
    "jurusan": "programstudi",
    "spp": "spp",
    "mhs": "mahasiswa",
    "univ": "universitas",
    "fti": "fti",
    "uajy": "uajy",
    "do": "dropout",
}

#: Klitik yang murni gramatikal dan aman dibuang.
_CLITIC_SUFFIXES = ("nya", "lah", "kah", "pun", "ku", "mu")

#: Panjang minimum kata sebelum klitik boleh dibuang, supaya "punya" tidak
#: terpotong jadi "pu" dan "kupu" tidak jadi "kup".
_MIN_STEM_LENGTH = 5

_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def _build_roman_numerals(limit: int = 30) -> frozenset[str]:
    """
    Angka romawi 1..limit dalam huruf kecil, sebagai daftar eksplisit.

    Regex romawi generik tidak bisa dipakai di sini: pola seperti
    ``^[ivxlcdm]+$`` juga cocok dengan kata Indonesia yang kebetulan hanya
    tersusun dari huruf tersebut ("di", "civil", "mil"), sehingga kata itu
    lolos dari filter stopword. Nomor BAB tidak pernah melebihi puluhan,
    jadi daftar eksplisit lebih aman.
    """
    numerals = [
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
    ]
    result = set()
    for number in range(1, limit + 1):
        remaining, out = number, ""
        for value, symbol in numerals:
            while remaining >= value:
                out += symbol
                remaining -= value
        result.add(out)
    return frozenset(result)


#: Angka romawi yang dianggap penanda nomor BAB/Bagian.
ROMAN_NUMERALS = _build_roman_numerals()


def normalize_text(text: str) -> str:
    """
    Lowercase, buang aksen, dan normalisasi frasa alias multi-kata.

    Dipanggil sebelum tokenisasi agar "mata kuliah" jadi satu token.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()

    for phrase, canonical in _PHRASE_ALIASES.items():
        if phrase in text:
            text = text.replace(phrase, canonical)

    return text


def strip_clitic(token: str) -> str:
    """Buang satu klitik gramatikal di akhir token, jika aman."""
    for suffix in _CLITIC_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM_LENGTH - 1:
            return token[: -len(suffix)]
    return token


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """
    Ubah teks bebas menjadi daftar token yang siap diindeks BM25.

    Angka dipertahankan karena dokumen akademik penuh rujukan numerik
    ("Pasal 12", "IPK 3.50", "144 SKS"). Angka romawi juga dipertahankan
    karena menandai nomor BAB.

    Args:
        text: Teks mentah.
        remove_stopwords: Buang stopwords Bahasa Indonesia.

    Returns:
        Daftar token ternormalisasi.
    """
    normalized = normalize_text(text)
    raw_tokens = [t for t in _NON_WORD_RE.split(normalized) if t]

    tokens: list[str] = []
    for token in raw_tokens:
        # Angka dipertahankan apa adanya.
        if token.isdigit():
            tokens.append(token)
            continue

        # Stopword dibuang lebih dulu supaya kata seperti "di" tidak lolos
        # lewat jalur angka romawi.
        if remove_stopwords and token in INDONESIAN_STOPWORDS:
            continue

        if token in ROMAN_NUMERALS:
            tokens.append(token)
            continue

        token = strip_clitic(token)
        token = _TOKEN_ALIASES.get(token, token)

        if len(token) < 2:
            continue
        if remove_stopwords and token in INDONESIAN_STOPWORDS:
            continue

        tokens.append(token)

    return tokens


def content_tokens(text: str) -> list[str]:
    """
    Token bermakna dari sebuah query, duplikat dibuang tapi urutan dijaga.

    Dipakai gate relevansi leksikal untuk menghitung seberapa banyak istilah
    query yang benar-benar muncul di sebuah chunk.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for token in tokenize(text):
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


if __name__ == "__main__":
    samples = [
        "Berapa IPK minimum untuk lulus cum laude?",
        "Apa saja syarat mata kuliah prasyarat di BAB IV Pasal 12?",
        "Bagaimana cara memasak nasi goreng?",
        "Berapa jumlah SKS untuk mengajukan skripsi / tugas akhir?",
    ]
    for s in samples:
        print(f"{s}\n  → {tokenize(s)}\n")
