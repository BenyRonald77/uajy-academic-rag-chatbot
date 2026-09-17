"""Model provider dapat diatur tanpa mengekspos API key."""

from app import config


def test_model_override_dari_environment_mengalahkan_secrets(monkeypatch, tmp_path):
    (tmp_path / ".streamlit").mkdir()
    (tmp_path / ".streamlit" / "secrets.toml").write_text(
        'LLM_MODEL = "auto"\n', encoding="utf-8"
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("LLM_MODEL", "glm-5.3-flash")

    assert config._read_model_setting("LLM_MODEL", "deepseek-v4-flash") == "glm-5.3-flash"


def test_model_override_dibaca_dari_secrets_toml(monkeypatch, tmp_path):
    streamlit_dir = tmp_path / ".streamlit"
    streamlit_dir.mkdir()
    (streamlit_dir / "secrets.toml").write_text(
        'UTILITY_MODEL = "glm-5.3-flash"\n', encoding="utf-8"
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("UTILITY_MODEL", raising=False)

    assert config._read_model_setting("UTILITY_MODEL", "deepseek-v4-flash") == "glm-5.3-flash"
