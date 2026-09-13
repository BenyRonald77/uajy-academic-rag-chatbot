"""
reranker.py — Reranking kandidat retrieval memakai LLM sebagai cross-encoder.

Kenapa perlu reranking?
    Dense retrieval dan BM25 sama-sama menilai query dan dokumen secara
    terpisah (bi-encoder / bag-of-words), jadi keduanya menilai *kemiripan*
    bukan *kemampuan menjawab*. Sebuah chunk bisa sangat mirip topiknya tapi
    tidak memuat jawabannya. Reranker melihat query dan chunk bersamaan,
    sehingga jauh lebih tepat memutuskan mana yang benar-benar menjawab.

Kenapa LLM, bukan cross-encoder lokal?
    Cross-encoder multilingual (mis. varian mMiniLM) berukuran ratusan MB dan
    menarik dependency PyTorch — bertabrakan dengan tujuan proyek ini yang
    ringan dan CPU-only. Satu panggilan `gemini-2.5-flash-lite` menilai
    seluruh kandidat sekaligus dengan latensi ratusan milidetik.

Pendekatannya *listwise*: semua kandidat dikirim dalam satu panggilan, model
memberi skor 0–10 per kandidat. Ini lebih murah daripada pointwise (satu
panggilan per kandidat) dan memberi model konteks pembanding.

Reranker didesain **fail-safe**: kegagalan apa pun (error API, JSON rusak)
mengembalikan urutan hasil fusi apa adanya, jadi chatbot tetap menjawab.
"""

from __future__ import annotations

import json
import re

from app.llm_client import call_utility_llm

#: Batas karakter tiap kandidat yang dikirim ke reranker. Cukup untuk menilai
#: relevansi tanpa membengkakkan prompt.
SNIPPET_CHAR_LIMIT = 900

RERANK_SYSTEM_PROMPT = """Kamu adalah komponen pemeringkat relevansi (reranker) pada sistem pencarian dokumen akademik.

Tugasmu: menilai seberapa baik setiap potongan dokumen dapat MENJAWAB pertanyaan pengguna.

Panduan skor (0-10):
- 9-10 : Memuat jawaban langsung dan lengkap atas pertanyaan.
- 6-8  : Memuat sebagian jawaban atau informasi pendukung yang jelas relevan.
- 3-5  : Topiknya berkaitan, tetapi tidak menjawab pertanyaan.
- 0-2  : Tidak relevan, hanya berbagi kata umum dengan pertanyaan.

Aturan penting:
- Nilai kemampuan MENJAWAB, bukan kemiripan kata. Potongan yang memuat kata yang sama tetapi membahas hal lain harus diberi skor rendah.
- Jika pertanyaan sama sekali di luar cakupan dokumen akademik kampus, beri skor 0-2 pada SEMUA potongan.
- Jangan mengarang. Nilai hanya berdasarkan teks yang diberikan.

Balas HANYA dengan JSON array, tanpa penjelasan:
[{"id": 1, "score": 8}, {"id": 2, "score": 3}]"""


def build_rerank_prompt(query: str, snippets: list[str]) -> str:
    """Susun prompt listwise untuk reranker."""
    parts = [f"PERTANYAAN PENGGUNA:\n{query}\n", "POTONGAN DOKUMEN:"]
    for i, snippet in enumerate(snippets, start=1):
        parts.append(f"\n[{i}]\n{snippet}")
    parts.append(
        f"\nBeri skor relevansi 0-10 untuk setiap potongan [1]-[{len(snippets)}]. "
        "Balas hanya JSON array."
    )
    return "\n".join(parts)


def _snippet_for(candidate) -> str:
    """Cuplikan kandidat, diawali heading agar model tahu konteks bagiannya."""
    heading = candidate.heading_label
    text = candidate.text.strip()
    if len(text) > SNIPPET_CHAR_LIMIT:
        text = text[:SNIPPET_CHAR_LIMIT].rsplit(" ", 1)[0] + " …"
    return f"(Halaman {candidate.page_label} — {heading})\n{text}" if heading else text


def parse_rerank_response(raw: str, expected: int) -> dict[int, float]:
    """
    Ubah balasan model menjadi peta ``id kandidat (1-based) → skor``.

    Toleran terhadap output yang dibungkus code fence atau diberi teks
    tambahan. ID di luar rentang dan skor non-numerik diabaikan.

    Returns:
        Peta id → skor. Kosong jika tidak ada yang bisa diurai.
    """
    if not raw:
        return {}

    payload = raw.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", payload).strip()

    data = None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", payload, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

    if isinstance(data, dict):
        # Beberapa model membungkus array dalam sebuah objek.
        for value in data.values():
            if isinstance(value, list):
                data = value
                break

    if not isinstance(data, list):
        return {}

    scores: dict[int, float] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            item_id = int(item.get("id"))
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        if 1 <= item_id <= expected:
            scores[item_id] = max(0.0, min(10.0, score))

    return scores


def rerank_candidates(
    query: str,
    candidates: list,
    api_key: str | None = None,
) -> list:
    """
    Nilai ulang dan urutkan kandidat berdasarkan kemampuannya menjawab query.

    Mengisi atribut ``rerank_score`` pada setiap kandidat lalu mengurutkan
    dari skor tertinggi. Kandidat yang tidak dinilai model (jarang terjadi)
    mempertahankan urutan fusinya di bagian bawah.

    Args:
        query: Pertanyaan (sudah melalui rewriting jika aktif).
        candidates: Daftar `RetrievalCandidate` hasil fusi.
        api_key: Override API key Gemini.

    Returns:
        Daftar kandidat yang sudah diurutkan ulang. Jika reranking gagal,
        daftar asli dikembalikan tanpa perubahan (``rerank_score`` tetap
        ``None``), sehingga gate reranker melewatkannya.
    """
    if len(candidates) <= 1:
        return candidates

    snippets = [_snippet_for(c) for c in candidates]
    prompt = build_rerank_prompt(query, snippets)

    try:
        raw = call_utility_llm(
            prompt,
            system_instruction=RERANK_SYSTEM_PROMPT,
            max_tokens=512,
            api_key=api_key,
            json_mode=True,
        )
    except Exception:
        # Fail-safe: pertahankan urutan fusi agar chatbot tetap menjawab.
        return candidates

    scores = parse_rerank_response(raw, expected=len(candidates))
    if not scores:
        return candidates

    for position, candidate in enumerate(candidates, start=1):
        candidate.rerank_score = scores.get(position)

    # Kandidat tanpa skor diletakkan setelah yang berskor, urutan fusi dijaga.
    def sort_key(item: tuple[int, object]) -> tuple:
        position, candidate = item
        score = getattr(candidate, "rerank_score", None)
        return (0 if score is None else 1, score or 0.0, -position)

    ordered = sorted(enumerate(candidates), key=sort_key, reverse=True)
    return [candidate for _, candidate in ordered]
