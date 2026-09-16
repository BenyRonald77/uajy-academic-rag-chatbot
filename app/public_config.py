"""Konfigurasi runtime publik/operator dan rate limit public mode."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, PUBLIC_APP_MODE, OPERATOR_APP_MODE


def _read_config_value(*names: str) -> Any:
    """Baca nilai dari environment lalu secrets.toml tanpa mencetak isinya."""
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value

    path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if path.exists():
        try:
            import tomllib
            with open(path, "rb") as f:
                data = tomllib.load(f)
            for name in names:
                value = data.get(name)
                if value not in (None, ""):
                    return value
        except Exception:
            pass
    return None


def app_mode() -> str:
    """
    Mode aplikasi.

    Default selalu `public` agar deploy VPS aman walau APP_MODE lupa diatur.
    Hanya nilai eksplisit `operator` yang membuka panel internal.
    """
    value = str(_read_config_value("APP_MODE") or PUBLIC_APP_MODE).strip().lower()
    return OPERATOR_APP_MODE if value == OPERATOR_APP_MODE else PUBLIC_APP_MODE


def is_operator_mode() -> bool:
    return app_mode() == OPERATOR_APP_MODE


def public_rate_limit() -> int:
    """Maksimum pertanyaan public per sesi dalam satu window."""
    try:
        return max(1, int(_read_config_value("PUBLIC_RATE_LIMIT") or 20))
    except (TypeError, ValueError):
        return 20


def public_rate_window_seconds() -> int:
    """Panjang window rate limit public dalam detik."""
    try:
        return max(60, int(_read_config_value("PUBLIC_RATE_WINDOW_SECONDS") or 3600))
    except (TypeError, ValueError):
        return 3600
