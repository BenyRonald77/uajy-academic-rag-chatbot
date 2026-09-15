"""
llm_client.py — Wrapper untuk Google Gemini API (LLM generation + embedding).

Modul ini sengaja TIDAK bergantung pada Streamlit sehingga bisa dipakai ulang
oleh script ingestion, runner evaluasi, dan (nanti) API/bot lain. UI Streamlit
menangkap `LLMConfigError` dan menampilkan pesannya sendiri.

Desain API-agnostic: cukup ganti isi fungsi di bawah jika ingin pindah
provider (OpenAI, Groq, dll).
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

from google import genai
from google.genai import types as genai_types

from app.config import (
    ANSWER_MAX_TOKENS,
    ANSWER_TEMPERATURE,
    EMBED_BATCH_DELAY,
    EMBED_BATCH_SIZE,
    EMBED_MAX_RETRIES,
    EMBEDDING_MODEL,
    LLM_MODEL,
    MODEL_FALLBACKS,
    PROJECT_ROOT,
    UTILITY_MODEL,
    UTILITY_TEMPERATURE,
)

_PLACEHOLDER_KEYS = {
    "",
    "MASUKKAN_API_KEY_GEMINI_ANDA_DI_SINI",
    "your_actual_gemini_api_key_here",
}

#: Task type embedding. Memberi tahu model apakah teks berperan sebagai
#: dokumen yang diindeks atau sebagai query pencarian — meningkatkan kualitas
#: retrieval dibanding embedding generik.
TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_TYPE_QUERY = "RETRIEVAL_QUERY"


class LLMConfigError(RuntimeError):
    """API key belum diatur atau tidak valid."""


class LLMCallError(RuntimeError):
    """Panggilan ke Gemini API gagal setelah semua percobaan."""


# ──────────────────────────────────────────────
# API Key & Client
# ──────────────────────────────────────────────

def resolve_api_key(explicit_key: str | None = None) -> str:
    """
    Cari API key Gemini dari beberapa sumber, berurutan:

    1. Argumen eksplisit (mis. flag ``--api-key``).
    2. Environment variable ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
    3. File ``.streamlit/secrets.toml``.
    4. ``st.secrets`` (hanya jika Streamlit sedang berjalan).

    Raises:
        LLMConfigError: jika tidak ada key valid yang ditemukan.
    """
    candidates: list[str | None] = [explicit_key]

    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        candidates.append(os.environ.get(env_name))

    candidates.append(_read_key_from_secrets_file())
    candidates.append(_read_key_from_streamlit())

    for candidate in candidates:
        if candidate and candidate.strip() not in _PLACEHOLDER_KEYS:
            return candidate.strip()

    raise LLMConfigError(
        "API key Gemini belum diatur. Pilih salah satu cara:\n"
        "  1. Set environment variable GEMINI_API_KEY\n"
        "  2. Isi GEMINI_API_KEY di .streamlit/secrets.toml\n"
        "  3. Jalankan script dengan flag --api-key YOUR_KEY"
    )


def _read_key_from_secrets_file() -> str | None:
    """Baca API key langsung dari .streamlit/secrets.toml tanpa Streamlit."""
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None

    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        return None

    try:
        with open(secrets_path, "rb") as f:
            return tomllib.load(f).get("GEMINI_API_KEY")
    except Exception:
        return None


def _read_key_from_streamlit() -> str | None:
    """Ambil dari st.secrets bila modul Streamlit tersedia dan punya key-nya."""
    try:
        import streamlit as st

        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


@lru_cache(maxsize=4)
def get_client(api_key: str | None = None) -> genai.Client:
    """Client Gemini yang di-cache per API key."""
    return genai.Client(api_key=resolve_api_key(api_key))


# ──────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────

def call_llm(
    prompt: str,
    system_instruction: str = "",
    temperature: float = ANSWER_TEMPERATURE,
    max_tokens: int = ANSWER_MAX_TOKENS,
    model: str = LLM_MODEL,
    api_key: str | None = None,
    raise_on_error: bool = False,
) -> str:
    """
    Panggil Gemini untuk menghasilkan jawaban akhir.

    Args:
        raise_on_error: Bila ``True``, kegagalan dilempar sebagai exception
            alih-alih dikembalikan sebagai pesan. UI memerlukan pesan yang
            ramah untuk ditampilkan di chat, tetapi runner evaluasi
            memerlukan kegagalan yang bisa dibedakan: pesan error yang
            diperlakukan sebagai jawaban akan merusak metriknya — penilai
            groundedness bahkan memberi nilai sempurna pada pesan error,
            sebab pesan itu memang tidak memuat klaim faktual apa pun.
    """
    if raise_on_error:
        return _generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            api_key=api_key,
        )

    try:
        text = _generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            api_key=api_key,
        )
        return text or "Maaf, saya tidak dapat menghasilkan jawaban untuk pertanyaan ini."
    except LLMConfigError as e:
        return f"⚠️ {e}"
    except Exception as e:
        error_msg = str(e)
        lowered = error_msg.lower()
        if "quota" in lowered or "rate" in lowered or "429" in error_msg:
            return "⚠️ Batas penggunaan API tercapai. Silakan coba lagi dalam beberapa menit."
        if "api_key" in lowered or "api key" in lowered or "authentication" in lowered:
            return "⚠️ API key tidak valid. Periksa konfigurasi di `.streamlit/secrets.toml`."
        return f"⚠️ Terjadi kesalahan saat memproses: {error_msg}"


def call_utility_llm(
    prompt: str,
    system_instruction: str = "",
    max_tokens: int = 1024,
    api_key: str | None = None,
    json_mode: bool = False,
) -> str:
    """
    Panggil model bantu yang murah & deterministik (rerank, query rewriting).

    Args:
        json_mode: Minta model mengembalikan JSON valid. Menghilangkan
            kebutuhan mengurai teks bebas pada output reranker.

    Raises:
        Exception: dibiarkan naik agar pemanggil bisa fallback dengan aman.
    """
    return _generate(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=UTILITY_TEMPERATURE,
        max_tokens=max_tokens,
        model=UTILITY_MODEL,
        api_key=api_key,
        response_mime_type="application/json" if json_mode else None,
    )


#: Galat yang menandakan model itu sendiri tidak bisa dipakai: 404 berarti
#: model sudah dihapus, 503 berarti sedang kelebihan beban. Keduanya khas
#: per-model, sehingga berpindah ke model lain masuk akal.
#:
#: Galat kuota (429) sengaja TIDAK disertakan. Batas kuota berlaku pada akun,
#: bukan pada model, jadi berpindah model tidak menolong dan justru
#: menghabiskan sisa kuota lebih cepat.
_MODEL_UNAVAILABLE_MARKERS = ("404", "NOT_FOUND", "503", "UNAVAILABLE")

#: Model yang terbukti bisa dipanggil, dipetakan dari model utama yang diminta.
#: Tanpa ingatan ini, setiap permintaan akan menabrak model yang sudah mati
#: lebih dulu dan menambah satu perjalanan jaringan yang pasti gagal.
_WORKING_MODEL: dict[str, str] = {}


def _is_model_unavailable(error: Exception) -> bool:
    """True bila galat menandakan modelnya yang bermasalah, bukan permintaannya."""
    message = str(error)
    return any(marker in message for marker in _MODEL_UNAVAILABLE_MARKERS)


def model_chain(primary: str) -> list[str]:
    """
    Urutan model yang dicoba untuk sebuah model utama.

    Model yang sebelumnya terbukti bekerja diletakkan paling depan, tetapi
    sisanya tetap dipertahankan sebagai cadangan agar sistem bisa pulih bila
    model itu pun kemudian ikut dihapus.
    """
    configured = [primary, *MODEL_FALLBACKS.get(primary, ())]

    working = _WORKING_MODEL.get(primary)
    if working and working in configured:
        return [working] + [m for m in configured if m != working]
    return configured


def effective_model(primary: str) -> str:
    """
    Model yang sedang benar-benar dipakai untuk `primary`.

    Dipakai UI agar yang ditampilkan adalah kenyataan, bukan konfigurasi yang
    barangkali sudah digantikan oleh fallback.
    """
    return _WORKING_MODEL.get(primary, primary)


def reset_model_cache() -> None:
    """Lupakan model yang tercatat bekerja. Terutama berguna untuk pengujian."""
    _WORKING_MODEL.clear()


def _generate_once(
    prompt: str,
    system_instruction: str,
    temperature: float,
    max_tokens: int,
    model: str,
    api_key: str | None,
    response_mime_type: str | None,
) -> str:
    client = get_client(api_key)
    config = genai_types.GenerateContentConfig(
        system_instruction=system_instruction or None,
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type=response_mime_type,
    )
    response = client.models.generate_content(model=model, contents=prompt, config=config)
    return (response.text or "").strip()


def _generate(
    prompt: str,
    system_instruction: str,
    temperature: float,
    max_tokens: int,
    model: str,
    api_key: str | None,
    response_mime_type: str | None = None,
) -> str:
    """
    Hasilkan teks, berpindah ke model cadangan bila modelnya tidak tersedia.

    Raises:
        Exception: galat terakhir, bila seluruh model pada rantai gagal.
    """
    chain = model_chain(model)
    last_error: Exception | None = None

    for candidate in chain:
        try:
            text = _generate_once(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=temperature,
                max_tokens=max_tokens,
                model=candidate,
                api_key=api_key,
                response_mime_type=response_mime_type,
            )
        except Exception as e:
            last_error = e
            # Galat non-model (kuota, autentikasi, permintaan salah) tidak akan
            # membaik dengan berpindah model, jadi diteruskan apa adanya.
            if not _is_model_unavailable(e):
                raise
            continue

        if candidate != model:
            _WORKING_MODEL[model] = candidate
        return text

    raise last_error if last_error else LLMCallError("Tidak ada model yang bisa dipanggil.")


# ──────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    task_type: str | None = TASK_TYPE_DOCUMENT,
    api_key: str | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
    batch_delay: float = EMBED_BATCH_DELAY,
    max_retries: int = EMBED_MAX_RETRIES,
    progress: bool = False,
) -> list[list[float]]:
    """
    Embed banyak teks sekaligus, dengan batching dan exponential backoff.

    Args:
        texts: Daftar teks yang akan di-embed.
        task_type: ``RETRIEVAL_DOCUMENT`` untuk chunk dokumen,
            ``RETRIEVAL_QUERY`` untuk pertanyaan, atau ``None`` untuk
            embedding generik (dipakai index versi lama).
        progress: Cetak progres batch ke stdout.

    Returns:
        Daftar embedding vector, urutannya sama dengan ``texts``.
    """
    if not texts:
        return []

    client = get_client(api_key)
    config = genai_types.EmbedContentConfig(task_type=task_type) if task_type else None

    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) - 1) // batch_size + 1

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1

        if progress:
            print(f"   Embedding batch {batch_num}/{total_batches} ({len(batch)} teks)...")

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=config,
                )
                all_embeddings.extend(e.values for e in result.embeddings)
                last_error = None
                break
            except Exception as e:
                last_error = e
                error_msg = str(e)
                is_rate_limit = "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg
                if not is_rate_limit or attempt == max_retries - 1:
                    break
                wait = batch_delay * (2 ** attempt)
                if progress:
                    print(f"   >> Rate limit, menunggu {wait:.0f}s "
                          f"(percobaan {attempt + 1}/{max_retries})...")
                time.sleep(wait)

        if last_error is not None:
            raise LLMCallError(
                f"Gagal embed batch {batch_num}/{total_batches}: {last_error}"
            ) from last_error

        if batch_num < total_batches and batch_delay > 0:
            time.sleep(batch_delay)

    return all_embeddings


#: Cache embedding query dalam proses.
#: Pertanyaan yang sama sering diulang: pengguna mencoba ulang, contoh
#: pertanyaan diklik berkali-kali, dan runner evaluasi menjalankan query yang
#: identik untuk beberapa konfigurasi. Menyimpannya menghemat kuota API
#: sekaligus menghilangkan satu perjalanan jaringan.
_QUERY_EMBEDDING_CACHE: dict[tuple[str, str | None], list[float]] = {}
_QUERY_EMBEDDING_CACHE_LIMIT = 512


def get_query_embedding(
    query: str,
    task_type: str | None = TASK_TYPE_QUERY,
    api_key: str | None = None,
    use_cache: bool = True,
) -> list[float]:
    """
    Embedding untuk satu query pencarian.

    ``task_type`` harus cocok dengan yang dipakai saat membangun index —
    `DocumentRetriever` membacanya dari ``index/index_info.json``.

    Args:
        query: Pertanyaan yang akan di-embed.
        task_type: Task type embedding, harus cocok dengan index.
        api_key: Override API key Gemini.
        use_cache: Pakai cache dalam proses untuk query yang identik.
    """
    cache_key = (query, task_type)
    if use_cache and cache_key in _QUERY_EMBEDDING_CACHE:
        return _QUERY_EMBEDDING_CACHE[cache_key]

    embeddings = embed_texts(
        [query],
        task_type=task_type,
        api_key=api_key,
        batch_delay=0.0,
    )
    if not embeddings:
        raise LLMCallError("API embedding tidak mengembalikan vector apa pun.")

    if use_cache:
        if len(_QUERY_EMBEDDING_CACHE) >= _QUERY_EMBEDDING_CACHE_LIMIT:
            _QUERY_EMBEDDING_CACHE.clear()
        _QUERY_EMBEDDING_CACHE[cache_key] = embeddings[0]

    return embeddings[0]


def clear_query_embedding_cache() -> None:
    """Kosongkan cache embedding query."""
    _QUERY_EMBEDDING_CACHE.clear()


def test_connection(api_key: str | None = None) -> bool:
    """
    Cek koneksi ke Gemini API. True jika berhasil.

    ``max_tokens`` sengaja dilonggarkan: pada model Gemini 2.5/3.x, token
    *thinking* ikut dihitung ke dalam ``max_output_tokens``. Dengan batas
    sangat kecil, model menghabiskan seluruh kuota untuk berpikir lalu
    berhenti dengan ``finish_reason=MAX_TOKENS`` tanpa menghasilkan teks —
    membuat koneksi yang sehat terlaporkan gagal.
    """
    try:
        return bool(call_utility_llm(
            "Balas hanya dengan kata: OK",
            max_tokens=256,
            api_key=api_key,
        ))
    except Exception:
        return False
