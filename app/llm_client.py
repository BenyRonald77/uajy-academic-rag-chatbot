"""
llm_client.py — Adapter provider untuk pipeline RAG.

Pembagian provider jangka pendek:

- **OpenAI-compatible endpoint** (`LLM_BASE_URL`) untuk jawaban, reranker,
  dan query rewriting. Default-nya `https://bandelbanget.xyz/v1`.
- **Google Gemini** hanya untuk embedding, karena endpoint OpenAI-compatible
  baru yang dipakai belum menyediakan `/embeddings`.

API internal tetap sama agar modul lain tidak berubah:

- `call_llm()`
- `call_utility_llm()`
- `embed_texts()`
- `get_query_embedding()`

Dengan begitu perpindahan provider tidak merusak retrieval, evaluasi, atau UI.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from google import genai

from app.config import (
    ANSWER_MAX_TOKENS,
    ANSWER_TEMPERATURE,
    EMBED_BATCH_DELAY,
    EMBED_BATCH_SIZE,
    EMBED_MAX_RETRIES,
    EMBEDDING_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
    MODEL_FALLBACKS,
    PROJECT_ROOT,
    TASK_TYPE_DOCUMENT,
    TASK_TYPE_QUERY,
    UTILITY_MODEL,
    UTILITY_TEMPERATURE,
)

_PLACEHOLDER_KEYS = {
    "",
    "MASUKKAN_API_KEY_GEMINI_ANDA_DI_SINI",
    "your_actual_gemini_api_key_here",
    "your_bandel_api_key_here",
    "your_api_key_here",
}

# Provider OpenAI-compatible mengembalikan error model/kuota per model.
# Karena itu model cadangan boleh dicoba untuk 404, 429, dan 503.
_MODEL_UNAVAILABLE_MARKERS = (
    "404", "NOT_FOUND",
    "429", "RESOURCE_EXHAUSTED", "RATE_LIMIT",
    "500", "502", "503", "504", "UNAVAILABLE",
)

# Ingatan proses: setelah `auto` terbukti bekerja, request berikutnya tidak
# perlu menguji model mati yang sama terlebih dahulu.
_WORKING_MODEL: dict[str, str] = {}

# Cache query embedding tetap memakai key task type agar embedding dokumen dan
# query tidak pernah tertukar.
_QUERY_EMBEDDING_CACHE: dict[tuple[str, str | None], list[float]] = {}
_QUERY_EMBEDDING_CACHE_LIMIT = 512

# Endpoint baru bisa membutuhkan waktu lebih lama saat gateway memilih model.
HTTP_TIMEOUT_SECONDS = 120

# Alias kompatibilitas: modul lama mengimpor konstanta ini dari llm_client.
# Sumber aslinya sekarang app.config.


class LLMConfigError(RuntimeError):
    """Provider/key belum dikonfigurasi."""


class LLMCallError(RuntimeError):
    """Panggilan provider gagal setelah semua percobaan."""


# ──────────────────────────────────────────────
# Secrets & provider config
# ──────────────────────────────────────────────

def _valid_key(value: Any) -> bool:
    return bool(value and str(value).strip() not in _PLACEHOLDER_KEYS)


def _looks_like_gateway_key(value: str | None) -> bool:
    """
    Heuristik untuk mencegah key Bandel dipakai ke Gemini embedding.

    Key gateway yang terlihat pada konfigurasi user berbentuk `sk-…`,
    sedangkan key Google AI Studio umumnya berbentuk `AIza…`. Ini bukan
    validasi autentikasi — hanya pencegah salah provider sebelum request.
    """
    return bool(value and str(value).strip().lower().startswith(("sk-", "sk_")))


def _read_secrets_file(*names: str) -> str | None:
    """Baca key generik dari secrets.toml tanpa mencetak nilainya."""
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return None

    try:
        import tomllib
        with open(secrets_path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, ValueError, ModuleNotFoundError):
        return None

    for name in names:
        value = data.get(name)
        if _valid_key(value):
            return str(value).strip()
    return None


def _read_streamlit_secret(*names: str) -> str | None:
    """Baca secret dari Streamlit bila aplikasi sedang berjalan."""
    try:
        import streamlit as st
        for name in names:
            value = st.secrets.get(name)
            if _valid_key(value):
                return str(value).strip()
    except Exception:
        pass
    return None


def resolve_llm_api_key(explicit_key: str | None = None) -> str:
    """
    Cari key untuk endpoint OpenAI-compatible.

    Prioritas:

    1. argumen eksplisit
    2. `LLM_API_KEY`, `BANDEL_API_KEY`, `OPENAI_API_KEY`
    3. key dengan nama sama di secrets.toml
    4. `GEMINI_API_KEY` lama sebagai fallback transisi

    Fallback terakhir membantu instalasi lama tetap hidup, tetapi konfigurasi
    yang dianjurkan adalah `LLM_API_KEY` agar jelas key itu bukan key Google.
    """
    candidates: list[str | None] = [explicit_key]
    for env_name in ("LLM_API_KEY", "BANDEL_API_KEY", "OPENAI_API_KEY"):
        candidates.append(os.environ.get(env_name))
    candidates.append(_read_secrets_file("LLM_API_KEY", "BANDEL_API_KEY", "OPENAI_API_KEY"))
    candidates.append(_read_streamlit_secret("LLM_API_KEY", "BANDEL_API_KEY", "OPENAI_API_KEY"))

    # Transisi: sebelumnya project hanya punya GEMINI_API_KEY. Ini boleh
    # dipakai untuk chat bila base URL-nya sudah diarahkan ke provider baru.
    candidates.append(os.environ.get("GEMINI_API_KEY"))
    candidates.append(_read_secrets_file("GEMINI_API_KEY"))
    candidates.append(_read_streamlit_secret("GEMINI_API_KEY"))

    for candidate in candidates:
        if _valid_key(candidate):
            return str(candidate).strip()

    raise LLMConfigError(
        "API key provider chat belum diatur. Isi salah satu:\n"
        "  1. LLM_API_KEY di .streamlit/secrets.toml\n"
        "  2. environment variable LLM_API_KEY\n"
        "  3. flag --api-key YOUR_KEY"
    )


def resolve_api_key(explicit_key: str | None = None) -> str:
    """
    Alias kompatibilitas untuk kode lama.

    Sebelumnya semua provider memakai nama `resolve_api_key`. Sekarang chat
    dan embedding dipisah, tetapi pemanggil lama tetap diarahkan ke resolver
    key chat agar tidak pecah saat migrasi.
    """
    return resolve_llm_api_key(explicit_key)


def resolve_embedding_api_key(explicit_key: str | None = None) -> str:
    """
    Cari key Google untuk embedding.

    Konfigurasi yang dianjurkan:

        GEMINI_EMBEDDING_API_KEY = "key_google"

    `GEMINI_API_KEY` tetap diterima sebagai kompatibilitas instalasi lama,
    tetapi key Bandel tidak boleh ditempatkan di sana bila embedding Gemini
    masih digunakan.
    """
    candidates: list[str | None] = [explicit_key]
    for env_name in (
        "GEMINI_EMBEDDING_API_KEY",
        "GOOGLE_API_KEY",
        "EMBEDDING_API_KEY",
    ):
        candidates.append(os.environ.get(env_name))

    candidates.append(_read_secrets_file(
        "GEMINI_EMBEDDING_API_KEY",
        "GOOGLE_API_KEY",
        "EMBEDDING_API_KEY",
    ))
    candidates.append(_read_streamlit_secret(
        "GEMINI_EMBEDDING_API_KEY",
        "GOOGLE_API_KEY",
        "EMBEDDING_API_KEY",
    ))

    # Kompatibilitas dengan instalasi lama yang menyimpan key Google di
    # GEMINI_API_KEY. Key gateway `sk-…` sengaja dilewati agar error-nya tidak
    # baru muncul sebagai 401 jauh di dalam request embedding.
    legacy_candidates = [
        os.environ.get("GEMINI_API_KEY"),
        _read_secrets_file("GEMINI_API_KEY"),
        _read_streamlit_secret("GEMINI_API_KEY"),
    ]
    candidates.extend(
        value for value in legacy_candidates
        if not _looks_like_gateway_key(value)
    )

    for candidate in candidates:
        if _valid_key(candidate):
            return str(candidate).strip()

    raise LLMConfigError(
        "API key Gemini embedding belum diatur. Isi "
        "GEMINI_EMBEDDING_API_KEY dengan key Google AI Studio. "
        "Key Bandel tidak bisa dipakai untuk embedding Gemini."
    )


def resolve_llm_base_url(explicit_url: str | None = None) -> str:
    """Cari base URL OpenAI-compatible dari argumen, env, secrets, default."""
    candidates: list[str | None] = [explicit_url]
    candidates.extend([
        os.environ.get("LLM_BASE_URL"),
        os.environ.get("OPENAI_BASE_URL"),
        _read_secrets_file("LLM_BASE_URL", "OPENAI_BASE_URL"),
        _read_streamlit_secret("LLM_BASE_URL", "OPENAI_BASE_URL"),
        LLM_BASE_URL,
    ])

    for candidate in candidates:
        if candidate and str(candidate).strip():
            return str(candidate).strip().rstrip("/")

    raise LLMConfigError("Base URL provider chat belum diatur.")


@lru_cache(maxsize=4)
def get_client(api_key: str | None = None) -> genai.Client:
    """Gemini client yang di-cache khusus untuk embedding."""
    return genai.Client(api_key=resolve_embedding_api_key(api_key))


# ──────────────────────────────────────────────
# OpenAI-compatible chat
# ──────────────────────────────────────────────

def _is_model_unavailable(error: Exception) -> bool:
    message = str(error).upper()
    return any(marker in message for marker in _MODEL_UNAVAILABLE_MARKERS)


def model_chain(primary: str) -> list[str]:
    """Urutan model utama dan cadangannya."""
    configured = [primary, *MODEL_FALLBACKS.get(primary, ())]
    working = _WORKING_MODEL.get(primary)
    if working and working in configured:
        return [working] + [model for model in configured if model != working]
    return configured


def effective_model(primary: str) -> str:
    """Model yang terakhir terbukti berhasil untuk model konfigurasi tertentu."""
    return _WORKING_MODEL.get(primary, primary)


def reset_model_cache() -> None:
    """Reset ingatan model aktif, terutama untuk test/operator."""
    _WORKING_MODEL.clear()


def _openai_error(status: int | None, body: str) -> LLMCallError:
    # Body provider boleh panjang dan kadang memuat detail internal. Tetap
    # simpan kode + cuplikan agar UI tidak dipenuhi dump respons mentah.
    compact = " ".join(body.split())[:700]
    return LLMCallError(f"{status or 'HTTP'} {compact}")


def _post_json(
    path: str,
    payload: dict,
    api_key: str,
    base_url: str,
) -> dict:
    """POST JSON dengan stdlib, tanpa dependency SDK tambahan."""
    request = Request(
        f"{base_url}/{path.lstrip('/')}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise _openai_error(error.code, body) from error
    except (URLError, TimeoutError, OSError) as error:
        raise LLMCallError(f"Provider tidak dapat dihubungi: {error}") from error

    try:
        data = json.loads(body)
    except json.JSONDecodeError as error:
        raise _openai_error(status, body) from error

    if not isinstance(data, dict):
        raise LLMCallError("Provider mengembalikan format JSON yang tidak valid.")
    if data.get("error"):
        raise _openai_error(status, json.dumps(data["error"], ensure_ascii=False))
    return data


def _extract_chat_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # OpenAI-compatible multimodal format kadang mengembalikan list part.
        return "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content).strip() if content else ""


def _generate_once(
    prompt: str,
    system_instruction: str,
    temperature: float,
    max_tokens: int,
    model: str,
    api_key: str | None,
    response_mime_type: str | None,
) -> str:
    """Satu panggilan chat completions ke endpoint OpenAI-compatible."""
    key = resolve_llm_api_key(api_key)
    base_url = resolve_llm_base_url()

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_mime_type == "application/json":
        payload["response_format"] = {"type": "json_object"}

    return _extract_chat_text(_post_json("chat/completions", payload, key, base_url))


def _generate(
    prompt: str,
    system_instruction: str,
    temperature: float,
    max_tokens: int,
    model: str,
    api_key: str | None,
    response_mime_type: str | None = None,
) -> str:
    """Generate dengan fallback model bila provider mengembalikan error model."""
    last_error: Exception | None = None

    for candidate in model_chain(model):
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
        except Exception as error:
            last_error = error
            if not _is_model_unavailable(error):
                raise
            continue

        if candidate != model:
            _WORKING_MODEL[model] = candidate
        return text

    raise last_error or LLMCallError("Tidak ada model chat yang bisa dipanggil.")


# ──────────────────────────────────────────────
# Public generation API
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
    """Panggil provider chat untuk jawaban akhir."""
    if raise_on_error:
        return _generate(
            prompt, system_instruction, temperature, max_tokens, model, api_key
        )

    try:
        text = _generate(
            prompt, system_instruction, temperature, max_tokens, model, api_key
        )
        return text or "Maaf, saya tidak dapat menghasilkan jawaban untuk pertanyaan ini."
    except LLMConfigError as error:
        return f"⚠️ {error}"
    except Exception as error:
        message = str(error)
        lowered = message.lower()
        if "quota" in lowered or "rate" in lowered or "429" in message:
            return "⚠️ Batas penggunaan API tercapai. Silakan coba lagi dalam beberapa menit."
        if any(token in lowered for token in ("api key", "api_key", "authentication", "401")):
            return "⚠️ API key provider tidak valid. Periksa konfigurasi secrets."
        return f"⚠️ Terjadi kesalahan saat memproses: {message[:500]}"


def call_utility_llm(
    prompt: str,
    system_instruction: str = "",
    max_tokens: int = 1024,
    api_key: str | None = None,
    json_mode: bool = False,
) -> str:
    """Panggil provider chat untuk reranker/query rewriting."""
    return _generate(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=UTILITY_TEMPERATURE,
        max_tokens=max_tokens,
        model=UTILITY_MODEL,
        api_key=api_key,
        response_mime_type="application/json" if json_mode else None,
    )


def test_connection(api_key: str | None = None) -> bool:
    """
    Tes koneksi DAN kepatuhan provider terhadap prompt.

    HTTP 200 saja tidak cukup. Endpoint Bandel yang diuji mengembalikan 200
    tetapi mengabaikan prompt dan mengirim kalimat motivasi tetap. Probe unik
    ini memastikan model benar-benar membaca instruksi sebelum UI/operator
    menganggap provider siap.
    """
    probe = "LLM_HEALTH_PROBE_7F3C"
    try:
        answer = call_utility_llm(
            f"Balas tepat dengan token {probe}. Jangan tambahkan kata lain.",
            max_tokens=32,
            api_key=api_key,
        )
        return probe in answer
    except Exception:
        return False


# ──────────────────────────────────────────────
# Gemini embedding API
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
    """Embed teks memakai Gemini khusus embedding."""
    if not texts:
        return []

    client = get_client(api_key)
    from google.genai import types as genai_types

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
            except Exception as error:
                last_error = error
                message = str(error)
                rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message
                if not rate_limited or attempt == max_retries - 1:
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


def get_query_embedding(
    query: str,
    task_type: str | None = TASK_TYPE_QUERY,
    api_key: str | None = None,
    use_cache: bool = True,
) -> list[float]:
    """Embedding query Gemini dengan cache proses."""
    cache_key = (query, task_type)
    if use_cache and cache_key in _QUERY_EMBEDDING_CACHE:
        return _QUERY_EMBEDDING_CACHE[cache_key]

    embeddings = embed_texts(
        [query], task_type=task_type, api_key=api_key, batch_delay=0.0
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
