"""Test mode public/operator dan rate limiting per sesi."""

from __future__ import annotations

from app.public_config import app_mode, is_operator_mode
from app.rate_limit import SessionRateLimiter


class TestPublicConfig:
    def test_default_adalah_public(self, monkeypatch):
        monkeypatch.delenv("APP_MODE", raising=False)
        assert app_mode() == "public"
        assert is_operator_mode() is False

    def test_operator_hanya_dengan_nilai_eksplisit(self, monkeypatch):
        monkeypatch.setenv("APP_MODE", "operator")
        assert app_mode() == "operator"
        assert is_operator_mode() is True

    def test_nilai_lain_jatuh_ke_public(self, monkeypatch):
        monkeypatch.setenv("APP_MODE", "admin-yang-salah")
        assert app_mode() == "public"
        assert is_operator_mode() is False


class TestSessionRateLimiter:
    def test_mengizinkan_sampai_batas(self):
        limiter = SessionRateLimiter(limit=2, window_seconds=3600)
        state = {}

        assert limiter.check_and_consume(state, now=100).allowed is True
        assert limiter.check_and_consume(state, now=101).allowed is True
        result = limiter.check_and_consume(state, now=102)

        assert result.allowed is False
        assert result.used == 2
        assert result.limit == 2
        assert result.retry_after_seconds > 0

    def test_window_baru_mereset_penggunaan(self):
        limiter = SessionRateLimiter(limit=1, window_seconds=60)
        state = {}

        assert limiter.check_and_consume(state, now=100).allowed is True
        assert limiter.check_and_consume(state, now=120).allowed is False
        assert limiter.check_and_consume(state, now=160).allowed is True
        assert state["rate_window_used"] == 1

    def test_state_dapat_disimpan_di_session_state(self):
        limiter = SessionRateLimiter(limit=3, window_seconds=3600)
        state = {}
        result = limiter.check_and_consume(state, now=100)

        assert result.allowed is True
        assert "rate_window_started_at" in state
        assert "rate_window_used" in state

    def test_limit_minimum_satu(self):
        limiter = SessionRateLimiter(limit=0, window_seconds=0)
        state = {}
        assert limiter.limit == 1
        assert limiter.window_seconds == 60
        assert limiter.check_and_consume(state, now=100).allowed is True
        assert limiter.check_and_consume(state, now=101).allowed is False
