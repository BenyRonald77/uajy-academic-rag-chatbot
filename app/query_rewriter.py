"""
query_rewriter.py — Ubah pertanyaan lanjutan menjadi query mandiri sebelum
retrieval.

Masalah yang diselesaikan:
    Retrieval bekerja per query, tanpa memori. Ketika pengguna bertanya
    "berapa syaratnya?" setelah sebelumnya membahas cuti kuliah, embedding
    dari kalimat itu sendiri hampir tidak membawa informasi — sistem akan
    mengambil chunk acak yang menyebut kata "syarat". Riwayat chat memang
    dikirim ke LLM penjawab, tetapi itu terlambat: konteks yang salah sudah
    terambil.

    Solusinya menulis ulang pertanyaan menjadi bentuk mandiri
    ("berapa syarat pengajuan cuti kuliah?") SEBELUM retrieval berjalan.

Hemat kuota:
    Menulis ulang setiap pertanyaan berarti satu panggilan LLM tambahan per
    query. Padahal sebagian besar pertanyaan sudah mandiri. Karena itu ada
    penyaring heuristik lebih dulu; LLM hanya dipanggil ketika pertanyaan
    benar-benar terlihat bergantung pada konteks sebelumnya.

Modul ini **fail-safe**: kegagalan apa pun mengembalikan query asli.
"""

from __future__ import annotations

import re

from app.llm_client import call_utility_llm

#: Penanda bahwa pertanyaan merujuk sesuatu yang disebut sebelumnya.
_REFERENTIAL_MARKERS = (
    "itu", "tersebut", "tadi", "sebelumnya", "yang mana", "kalau",
    "bagaimana dengan", "gimana dengan", "terus", "lalu", "kemudian",
    "dia", "mereka", "nya", "ini", "begitu", "sama seperti",
    "berapa lama", "kenapa", "mengapa", "apakah begitu", "selain itu",
    "contohnya", "misalnya", "detailnya", "lengkapnya", "bedanya",
)

#: Pertanyaan dengan kata sebanyak ini atau kurang hampir pasti butuh konteks
#: ("berapa syaratnya?", "apa itu?", "berapa lama maksimal?").
#: Dihitung dari kata mentah, bukan token hasil tokenisasi: tokenizer membuang
#: stopwords sehingga pertanyaan mandiri seperti "Berapa IPK minimum untuk
#: lulus cum laude?" pun menyusut jadi empat token.
_SHORT_QUERY_WORD_LIMIT = 4

#: Berapa pesan terakhir yang dipakai sebagai konteks penulisan ulang.
_HISTORY_WINDOW = 6

#: Batas panjang tiap pesan riwayat dalam prompt, agar prompt tetap ringkas.
_HISTORY_MESSAGE_CHAR_LIMIT = 400

REWRITE_SYSTEM_PROMPT = """Kamu adalah komponen penulis ulang query pada sistem pencarian dokumen akademik.

Tugasmu: mengubah pertanyaan terakhir pengguna menjadi pertanyaan MANDIRI yang bisa dipahami tanpa membaca riwayat percakapan.

Aturan:
1. Ganti kata rujukan ("itu", "tersebut", "-nya", "dia") dengan entitas yang dimaksud dari riwayat percakapan.
2. Pertahankan maksud asli. JANGAN menambah syarat, batasan, atau topik baru.
3. Pertahankan semua istilah dan angka spesifik dari pertanyaan asli.
4. Jika pertanyaan terakhir sudah mandiri, kembalikan APA ADANYA tanpa perubahan.
5. Gunakan Bahasa Indonesia.
6. Balas HANYA dengan satu kalimat pertanyaan hasil penulisan ulang. Tanpa penjelasan, tanpa tanda kutip, tanpa awalan apa pun."""


def needs_rewrite(query: str, chat_history: list[dict] | None) -> bool:
    """
    Tentukan apakah query perlu ditulis ulang, tanpa memanggil LLM.

    True bila ada riwayat percakapan DAN pertanyaannya terlihat bergantung
    konteks: sangat pendek, atau memuat kata rujukan.

    Args:
        query: Pertanyaan terakhir pengguna.
        chat_history: Riwayat percakapan sebelumnya.
    """
    if not chat_history:
        return False

    # Butuh minimal satu putaran percakapan sebelumnya untuk dirujuk.
    if not any(msg.get("role") == "assistant" for msg in chat_history):
        return False

    stripped = query.strip()
    if not stripped:
        return False

    if len(stripped.split()) <= _SHORT_QUERY_WORD_LIMIT:
        return True

    lowered = stripped.lower()

    # Sufiks posesif "-nya" adalah rujukan yang paling sering muncul
    # ("berapa biayanya", "apa syaratnya") dan tidak tertangkap pencocokan
    # kata utuh karena menempel pada kata dasarnya.
    if re.search(r"\w{3,}nya\b", lowered):
        return True

    return any(
        re.search(rf"\b{re.escape(marker)}\b", lowered)
        for marker in _REFERENTIAL_MARKERS
    )


def build_rewrite_prompt(query: str, chat_history: list[dict]) -> str:
    """Susun prompt penulisan ulang dari jendela riwayat terakhir."""
    recent = chat_history[-_HISTORY_WINDOW:]

    parts = ["RIWAYAT PERCAKAPAN:"]
    for msg in recent:
        role = "Pengguna" if msg.get("role") == "user" else "Asisten"
        content = (msg.get("content") or "").strip()
        if len(content) > _HISTORY_MESSAGE_CHAR_LIMIT:
            content = content[:_HISTORY_MESSAGE_CHAR_LIMIT].rsplit(" ", 1)[0] + " …"
        parts.append(f"{role}: {content}")

    parts.append(f"\nPERTANYAAN TERAKHIR PENGGUNA:\n{query}")
    parts.append("\nPertanyaan mandiri hasil penulisan ulang:")
    return "\n".join(parts)


def _sanitize(raw: str, fallback: str) -> str:
    """
    Bersihkan balasan model dan tolak hasil yang jelas menyimpang.

    Model bantu sesekali menambahkan awalan, tanda kutip, atau malah
    menjawab pertanyaannya. Hasil yang terlalu panjang atau multi-baris
    dianggap gagal dan query asli dipakai kembali.
    """
    if not raw:
        return fallback

    text = raw.strip()

    # Ambil baris tidak kosong pertama.
    for line in text.splitlines():
        if line.strip():
            text = line.strip()
            break

    text = re.sub(r'^["\'`]+|["\'`]+$', "", text).strip()
    text = re.sub(
        r"^(pertanyaan mandiri|hasil penulisan ulang|query|jawaban)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    if not text:
        return fallback

    # Penulisan ulang yang wajar tidak akan jauh lebih panjang dari aslinya.
    if len(text) > max(240, len(fallback) * 6):
        return fallback

    return text


def rewrite_query(
    query: str,
    chat_history: list[dict] | None,
    api_key: str | None = None,
    force: bool = False,
) -> str:
    """
    Kembalikan bentuk mandiri dari `query`, atau query asli bila tak perlu.

    Args:
        query: Pertanyaan terakhir pengguna.
        chat_history: Riwayat percakapan (format ``{"role", "content"}``).
        api_key: Override API key Gemini.
        force: Lewati penyaring heuristik dan selalu panggil LLM.

    Returns:
        Query yang akan dipakai untuk retrieval. Selalu berupa string yang
        bisa dipakai — kegagalan apa pun jatuh kembali ke query asli.
    """
    if not force and not needs_rewrite(query, chat_history):
        return query

    if not chat_history:
        return query

    try:
        raw = call_utility_llm(
            build_rewrite_prompt(query, chat_history),
            system_instruction=REWRITE_SYSTEM_PROMPT,
            max_tokens=256,
            api_key=api_key,
        )
    except Exception:
        return query

    return _sanitize(raw, fallback=query)
