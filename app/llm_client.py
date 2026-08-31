"""
llm_client.py — Wrapper untuk Google Gemini API (LLM generation + embedding).

Desain API-agnostic: cukup ganti fungsi call_llm() dan get_query_embedding()
jika ingin pindah ke provider lain (OpenAI, Groq, dll).
"""

import streamlit as st
from google import genai


# ──────────────────────────────────────────────
# Konfigurasi Model
# ──────────────────────────────────────────────

LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# Generation parameters
DEFAULT_TEMPERATURE = 0.2  # rendah untuk jawaban faktual
DEFAULT_MAX_TOKENS = 2048


def _get_client() -> genai.Client:
    """Dapatkan Gemini client dengan API key dari Streamlit secrets."""
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "MASUKKAN_API_KEY_GEMINI_ANDA_DI_SINI":
        st.error("⚠️ API key Gemini belum diatur! Edit file `.streamlit/secrets.toml`.")
        st.stop()

    return genai.Client(api_key=api_key)


def call_llm(
    prompt: str,
    system_instruction: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """
    Panggil Gemini LLM untuk generate jawaban.

    Args:
        prompt: Prompt lengkap (konteks + pertanyaan user).
        system_instruction: System instruction untuk model.
        temperature: Kreativitas jawaban (0.0-1.0). Default rendah untuk faktual.
        max_tokens: Maksimum token jawaban.

    Returns:
        Teks jawaban dari LLM.
    """
    client = _get_client()

    try:
        config = genai.types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=config,
        )

        if response.text:
            return response.text.strip()
        else:
            return "Maaf, saya tidak dapat menghasilkan jawaban untuk pertanyaan ini."

    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "rate" in error_msg.lower():
            return ("⚠️ Batas penggunaan API tercapai. Silakan coba lagi dalam beberapa menit.")
        elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return "⚠️ API key tidak valid. Periksa konfigurasi di `.streamlit/secrets.toml`."
        else:
            return f"⚠️ Terjadi kesalahan saat memproses: {error_msg}"


def get_query_embedding(query: str) -> list[float]:
    """
    Dapatkan embedding vector untuk query pengguna.

    Args:
        query: Pertanyaan pengguna.

    Returns:
        Embedding vector (list of floats).
    """
    client = _get_client()

    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
        )
        return result.embeddings[0].values

    except Exception as e:
        st.error(f"⚠️ Gagal menggenerate embedding: {e}")
        st.stop()


def test_connection() -> bool:
    """Test koneksi ke Gemini API. Returns True jika berhasil."""
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents="Jawab hanya dengan 'OK': apakah koneksi berhasil?",
            config=genai.types.GenerateContentConfig(
                max_output_tokens=10,
            ),
        )
        return bool(response.text)
    except Exception:
        return False
