"""
prompt_builder.py — Menyusun prompt untuk LLM.

Menggabungkan system instruction, konteks dari retrieved chunks,
dan pertanyaan pengguna menjadi prompt yang terstruktur.
"""

from app.retrieval import RetrievalResult


# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """Kamu adalah asisten chatbot resmi Universitas Atma Jaya Yogyakarta (UAJY) yang menjawab pertanyaan berdasarkan dokumen akademik kampus.

ATURAN KETAT yang WAJIB diikuti:

1. **HANYA jawab berdasarkan konteks yang diberikan.** Jangan pernah menambahkan informasi dari pengetahuan umummu. Jika informasi tidak ada di konteks, katakan dengan jujur bahwa informasi tersebut tidak ditemukan.

2. **Selalu sebutkan sumber.** Di akhir jawaban, sebutkan nomor halaman dan/atau bagian dokumen tempat informasi ditemukan, dengan format:
   📄 Sumber: Halaman X — [Nama Bagian]

3. **Jawab dalam Bahasa Indonesia** yang sopan, jelas, dan profesional. Gunakan format yang mudah dibaca (bullet points, numbering) jika jawabannya berisi beberapa poin.

4. **Jika pertanyaan di luar topik** dokumen kampus (misalnya tentang berita umum, hal pribadi, dll), tolak dengan sopan:
   "Maaf, saya hanya dapat menjawab pertanyaan seputar dokumen akademik Universitas Atma Jaya Yogyakarta. Silakan ajukan pertanyaan terkait peraturan akademik, prosedur kampus, atau informasi yang ada di dokumen pedoman."

5. **Jika informasi tidak ditemukan** dalam konteks yang diberikan, jawab:
   "Mohon maaf, informasi mengenai hal tersebut tidak ditemukan dalam dokumen yang tersedia. Silakan hubungi bagian akademik kampus untuk informasi lebih lanjut."

6. **Jangan mengarang atau berasumsi.** Lebih baik bilang tidak tahu daripada memberikan informasi yang salah.

7. **Pertahankan konteks percakapan.** Jika ada riwayat chat, gunakan untuk memahami konteks pertanyaan lanjutan."""


# ──────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────

def build_prompt(
    question: str,
    retrieved_chunks: list[RetrievalResult],
    chat_history: list[dict] | None = None,
) -> tuple[str, str]:
    """
    Bangun prompt lengkap untuk LLM.

    Args:
        question: Pertanyaan pengguna.
        retrieved_chunks: Chunk dokumen yang relevan dari retrieval.
        chat_history: Riwayat chat sebelumnya (opsional).
            Format: [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        Tuple of (system_instruction, user_prompt).
    """
    parts = []

    # Tambahkan konteks dari retrieved chunks
    if retrieved_chunks:
        parts.append("=" * 50)
        parts.append("KONTEKS DOKUMEN (gunakan HANYA informasi di bawah ini untuk menjawab):")
        parts.append("=" * 50)

        for i, chunk in enumerate(retrieved_chunks, 1):
            # `page_label` memberi rentang halaman yang ringkas ("51" atau
            # "51-52") dan `heading_label` memberi jalur bagian berjenjang,
            # sehingga LLM dapat mengutip sumber setepat mungkin.
            section = f" — {chunk.heading_label}" if chunk.heading_label else ""
            parts.append(f"\n--- Sumber #{i} (Halaman {chunk.page_label}{section}) ---")
            parts.append(chunk.text)

        parts.append("\n" + "=" * 50)

    # Tambahkan riwayat percakapan jika ada (ambil 6 pesan terakhir saja)
    if chat_history:
        recent_history = chat_history[-6:]
        parts.append("\nRIWAYAT PERCAKAPAN TERKINI:")
        for msg in recent_history:
            role = "Pengguna" if msg["role"] == "user" else "Asisten"
            parts.append(f"{role}: {msg['content']}")
        parts.append("")

    # Tambahkan pertanyaan
    parts.append(f"PERTANYAAN PENGGUNA: {question}")

    # Instruksi akhir
    if retrieved_chunks:
        parts.append("\nJawab pertanyaan di atas HANYA berdasarkan konteks dokumen yang diberikan. "
                      "Sertakan sumber (halaman/bagian) di akhir jawaban.")
    else:
        parts.append("\nTidak ada konteks dokumen yang relevan ditemukan untuk pertanyaan ini. "
                      "Jawab sesuai aturan untuk kasus 'informasi tidak ditemukan'.")

    user_prompt = "\n".join(parts)
    return SYSTEM_PROMPT, user_prompt


def build_no_context_response() -> str:
    """
    Response ketika tidak ada chunk relevan yang ditemukan (FR-5).
    Dipanggil TANPA memanggil LLM untuk hemat biaya API.
    """
    return (
        "Mohon maaf, informasi mengenai hal tersebut **tidak ditemukan** "
        "dalam dokumen yang tersedia saat ini.\n\n"
        "Beberapa saran:\n"
        "- Coba formulasikan pertanyaan dengan kata kunci yang berbeda.\n"
        "- Pastikan pertanyaan berkaitan dengan isi dokumen pedoman akademik.\n"
        "- Hubungi bagian akademik kampus untuk informasi yang tidak tercakup "
        "dalam dokumen ini."
    )
