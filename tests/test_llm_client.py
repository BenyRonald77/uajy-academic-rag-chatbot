"""test_llm_client.py — provider chat/embedding config dan cache query."""

from __future__ import annotations

import pytest

from app import llm_client
from app.llm_client import (
    LLMConfigError,
    clear_query_embedding_cache,
    get_query_embedding,
    resolve_api_key,
    resolve_embedding_api_key,
    resolve_llm_base_url,
)


@pytest.fixture(autouse=True)
def isolasi_lingkungan(monkeypatch):
    """Jauhkan test dari secrets sungguhan milik mesin developer."""
    for name in (
        "LLM_API_KEY", "BANDEL_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
        "GEMINI_EMBEDDING_API_KEY", "GOOGLE_API_KEY", "EMBEDDING_API_KEY",
        "LLM_BASE_URL", "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_client, "_read_secrets_file", lambda *names: None)
    monkeypatch.setattr(llm_client, "_read_streamlit_secret", lambda *names: None)
    clear_query_embedding_cache()
    llm_client.reset_model_cache()
    yield
    clear_query_embedding_cache()
    llm_client.reset_model_cache()


class TestResolveChatApiKey:
    def test_argumen_eksplisit_diprioritaskan(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "dari-env")
        assert resolve_api_key("dari-argumen") == "dari-argumen"

    def test_dari_llm_environment_variable(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "kunci-llm")
        assert resolve_api_key() == "kunci-llm"

    def test_bandel_api_key_sebagai_alternatif(self, monkeypatch):
        monkeypatch.setenv("BANDEL_API_KEY", "kunci-bandel")
        assert resolve_api_key() == "kunci-bandel"

    def test_openai_api_key_sebagai_alternatif(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "kunci-openai")
        assert resolve_api_key() == "kunci-openai"

    def test_legacy_gemini_key_diterima_sebagai_transisi(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "legacy-key")
        assert resolve_api_key() == "legacy-key"

    def test_llm_didahulukan_atas_legacy(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "utama")
        monkeypatch.setenv("GEMINI_API_KEY", "legacy")
        assert resolve_api_key() == "utama"

    def test_dari_berkas_secrets(self, monkeypatch):
        monkeypatch.setattr(
            llm_client, "_read_secrets_file",
            lambda *names: "kunci-secrets" if "LLM_API_KEY" in names else None,
        )
        assert resolve_api_key() == "kunci-secrets"

    @pytest.mark.parametrize("placeholder", [
        "",
        "MASUKKAN_API_KEY_GEMINI_ANDA_DI_SINI",
        "your_actual_gemini_api_key_here",
        "your_bandel_api_key_here",
    ])
    def test_placeholder_ditolak(self, monkeypatch, placeholder):
        monkeypatch.setenv("LLM_API_KEY", placeholder)
        with pytest.raises(LLMConfigError):
            resolve_api_key()

    def test_pesan_galat_menyebut_konfigurasi_baru(self):
        with pytest.raises(LLMConfigError) as info:
            resolve_api_key()
        pesan = str(info.value)
        assert "LLM_API_KEY" in pesan
        assert "secrets.toml" in pesan


class TestResolveEmbeddingKey:
    def test_key_embedding_diprioritaskan(self, monkeypatch):
        monkeypatch.setenv("GEMINI_EMBEDDING_API_KEY", "embedding-utama")
        monkeypatch.setenv("GEMINI_API_KEY", "legacy")
        assert resolve_embedding_api_key() == "embedding-utama"

    def test_google_api_key_diterima(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        assert resolve_embedding_api_key() == "google-key"

    def test_tanpa_key_gagal_dengan_pesan_jelas(self):
        with pytest.raises(LLMConfigError, match="embedding"):
            resolve_embedding_api_key()


class TestResolveBaseUrl:
    def test_default_bandel(self):
        assert resolve_llm_base_url() == "https://bandelbanget.xyz/v1"

    def test_environment_mengoverride_default(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1/")
        assert resolve_llm_base_url() == "https://example.test/v1"

    def test_secrets_mengoverride_default(self, monkeypatch):
        monkeypatch.setattr(
            llm_client, "_read_secrets_file",
            lambda *names: "https://secret.test/v1" if "LLM_BASE_URL" in names else None,
        )
        assert resolve_llm_base_url() == "https://secret.test/v1"


class TestQueryEmbeddingCache:
    def test_query_identik_hanya_sekali_memanggil_api(self, monkeypatch):
        panggilan = {"count": 0}

        def palsu(texts, **kwargs):
            panggilan["count"] += 1
            return [[0.1, 0.2, 0.3]]

        monkeypatch.setattr(llm_client, "embed_texts", palsu)
        pertama = get_query_embedding("pertanyaan sama")
        kedua = get_query_embedding("pertanyaan sama")
        assert pertama == kedua
        assert panggilan["count"] == 1

    def test_query_berbeda_memanggil_terpisah(self, monkeypatch):
        panggilan = {"count": 0}

        def palsu(texts, **kwargs):
            panggilan["count"] += 1
            return [[float(panggilan["count"])]]

        monkeypatch.setattr(llm_client, "embed_texts", palsu)
        get_query_embedding("pertanyaan A")
        get_query_embedding("pertanyaan B")
        assert panggilan["count"] == 2

    def test_task_type_berbeda_tidak_saling_tumpang(self, monkeypatch):
        panggilan = {"count": 0}

        def palsu(texts, **kwargs):
            panggilan["count"] += 1
            return [[float(panggilan["count"])]]

        monkeypatch.setattr(llm_client, "embed_texts", palsu)
        get_query_embedding("pertanyaan", task_type="RETRIEVAL_QUERY")
        get_query_embedding("pertanyaan", task_type=None)
        assert panggilan["count"] == 2

    def test_cache_bisa_dimatikan(self, monkeypatch):
        panggilan = {"count": 0}

        def palsu(texts, **kwargs):
            panggilan["count"] += 1
            return [[0.5]]

        monkeypatch.setattr(llm_client, "embed_texts", palsu)
        get_query_embedding("pertanyaan", use_cache=False)
        get_query_embedding("pertanyaan", use_cache=False)
        assert panggilan["count"] == 2

    def test_clear_cache_bekerja(self, monkeypatch):
        panggilan = {"count": 0}

        def palsu(texts, **kwargs):
            panggilan["count"] += 1
            return [[0.5]]

        monkeypatch.setattr(llm_client, "embed_texts", palsu)
        get_query_embedding("pertanyaan")
        clear_query_embedding_cache()
        get_query_embedding("pertanyaan")
        assert panggilan["count"] == 2

    def test_balasan_kosong_menghasilkan_galat(self, monkeypatch):
        monkeypatch.setattr(llm_client, "embed_texts", lambda texts, **kwargs: [])
        with pytest.raises(llm_client.LLMCallError):
            get_query_embedding("pertanyaan")


class TestEmbedTexts:
    def test_daftar_kosong_tidak_memanggil_api(self, monkeypatch):
        def jangan_dipanggil(*args, **kwargs):
            raise AssertionError("client tidak boleh dibuat untuk input kosong")

        monkeypatch.setattr(llm_client, "get_client", jangan_dipanggil)
        assert llm_client.embed_texts([]) == []
