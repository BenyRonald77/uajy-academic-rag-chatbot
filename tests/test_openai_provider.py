"""Test adapter OpenAI-compatible tanpa memanggil jaringan."""

from __future__ import annotations

from app import llm_client


class TestOpenAICompatiblePayload:
    def test_system_instruction_dan_user_prompt_dikirim(self, monkeypatch):
        captured = {}

        def fake_post(path, payload, api_key, base_url):
            captured.update({"path": path, "payload": payload, "base_url": base_url})
            return {
                "choices": [{"message": {"role": "assistant", "content": "jawaban"}}]
            }

        monkeypatch.setattr(llm_client, "_post_json", fake_post)
        result = llm_client._generate_once(
            prompt="pertanyaan",
            system_instruction="aturan sistem",
            temperature=0.2,
            max_tokens=100,
            model="auto",
            api_key="key-test",
            response_mime_type=None,
        )

        assert result == "jawaban"
        assert captured["path"] == "chat/completions"
        assert captured["payload"]["model"] == "auto"
        assert captured["payload"]["messages"] == [
            {"role": "system", "content": "aturan sistem"},
            {"role": "user", "content": "pertanyaan"},
        ]
        assert captured["payload"]["max_tokens"] == 100

    def test_json_mode_meminta_response_format(self, monkeypatch):
        captured = {}

        def fake_post(path, payload, api_key, base_url):
            captured["payload"] = payload
            return {
                "choices": [{"message": {"content": '{"score": 8}'}}]
            }

        monkeypatch.setattr(llm_client, "_post_json", fake_post)
        result = llm_client._generate_once(
            prompt="nilai",
            system_instruction="JSON only",
            temperature=0.0,
            max_tokens=64,
            model="auto",
            api_key="key-test",
            response_mime_type="application/json",
        )

        assert result == '{"score": 8}'
        assert captured["payload"]["response_format"] == {"type": "json_object"}

    def test_content_part_list_digabung(self, monkeypatch):
        monkeypatch.setattr(
            llm_client,
            "_post_json",
            lambda *args: {
                "choices": [{
                    "message": {"content": [
                        {"type": "text", "text": "bagian "},
                        {"type": "text", "text": "satu"},
                    ]}
                }]
            },
        )
        result = llm_client._generate_once(
            prompt="q", system_instruction="", temperature=0,
            max_tokens=10, model="auto", api_key="k", response_mime_type=None,
        )
        assert result == "bagian satu"

    def test_choices_kosong_mengembalikan_string_kosong(self, monkeypatch):
        monkeypatch.setattr(llm_client, "_post_json", lambda *args: {"choices": []})
        result = llm_client._generate_once(
            prompt="q", system_instruction="", temperature=0,
            max_tokens=10, model="auto", api_key="k", response_mime_type=None,
        )
        assert result == ""

    def test_provider_error_diubah_menjadi_llm_call_error(self, monkeypatch):
        monkeypatch.setattr(
            llm_client,
            "urlopen",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                llm_client.URLError("server down")
            ),
        )
        try:
            llm_client._post_json("chat/completions", {}, "key", "https://example.test/v1")
        except llm_client.LLMCallError as error:
            assert "tidak dapat dihubungi" in str(error)
        else:
            raise AssertionError("provider error seharusnya menjadi LLMCallError")
