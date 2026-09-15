"""
test_llm_client.py — Resolusi API key dan cache embedding.

Tidak ada panggilan jaringan di sini: fungsi embedding di-monkeypatch, dan
resolusi API key diuji lewat environment variable tiruan.
"""

from __future__ import annotations

import pytest

from app import llm_client
from app.llm_client import (
    LLMConfigError,
    clear_query_embedding_cache,
    get_query_embedding,
    resolve_api_key,
)


@pytest.fixture(autouse=True)
def isolasi_lingkungan(monkeypatch):
    """
    Jauhkan test dari kunci sungguhan milik mesin ini.

    Tanpa isolasi ini, test bisa "lolos" hanya karena `.streamlit/secrets.toml`
    pengembang kebetulan terisi — dan gagal di CI.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(llm_client, "_read_key_from_secrets_file", lambda: None)
    monkeypatch.setattr(llm_client, "_read_key_from_streamlit", lambda: None)
    clear_query_embedding_cache()
    yield
    clear_query_embedding_cache()


class TestResolveApiKey:
    def test_argumen_eksplisit_diprioritaskan(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dari-env")
        assert resolve_api_key("dari-argumen") == "dari-argumen"

    def test_dari_environment_variable(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "kunci-env")
        assert resolve_api_key() == "kunci-env"

    def test_google_api_key_sebagai_alternatif(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "kunci-google")
        assert resolve_api_key() == "kunci-google"

    def test_gemini_didahulukan_atas_google(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "utama")
        monkeypatch.setenv("GOOGLE_API_KEY", "cadangan")
        assert resolve_api_key() == "utama"

    def test_dari_berkas_secrets(self, monkeypatch):
        monkeypatch.setattr(
            llm_client, "_read_key_from_secrets_file", lambda: "kunci-secrets"
        )
        assert resolve_api_key() == "kunci-secrets"

    def test_env_didahulukan_atas_secrets(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "kunci-env")
        monkeypatch.setattr(
            llm_client, "_read_key_from_secrets_file", lambda: "kunci-secrets"
        )
        assert resolve_api_key() == "kunci-env"

    @pytest.mark.parametrize("placeholder", [
        "",
        "MASUKKAN_API_KEY_GEMINI_ANDA_DI_SINI",
        "your_actual_gemini_api_key_here",
    ])
    def test_placeholder_ditolak(self, monkeypatch, placeholder):
        """
        Placeholder harus diperlakukan sebagai "belum diatur".

        Kalau diloloskan, kegagalannya muncul jauh di dalam sebagai galat
        autentikasi yang membingungkan, bukan sebagai pesan konfigurasi.
        """
        monkeypatch.setenv("GEMINI_API_KEY", placeholder)
        with pytest.raises(LLMConfigError):
            resolve_api_key()

    def test_spasi_dibersihkan(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "  kunci-berspasi  ")
        assert resolve_api_key() == "kunci-berspasi"

    def test_pesan_galat_menyebut_semua_cara(self):
        with pytest.raises(LLMConfigError) as info:
            resolve_api_key()
        pesan = str(info.value)
        assert "GEMINI_API_KEY" in pesan
        assert "secrets.toml" in pesan
        assert "--api-key" in pesan


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
        assert panggilan["count"] == 1, "panggilan kedua seharusnya dari cache"

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
        """
        Task type ikut menjadi kunci cache.

        Embedding untuk task type berbeda tidak boleh dipertukarkan, sebab
        vektornya memang berbeda ruang.
        """
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
