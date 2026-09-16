"""Rate limit sederhana per sesi Streamlit untuk public mode."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    used: int
    limit: int
    retry_after_seconds: int = 0


class SessionRateLimiter:
    """
    Fixed-window limiter ringan untuk satu sesi aplikasi.

    Ini bukan pengganti limiter terpusat Redis/Nginx untuk deployment besar.
    Tujuannya menjaga public mode dari satu sesi yang menghabiskan seluruh
    kuota provider, tanpa menambah dependency atau database.
    """

    def __init__(self, limit: int, window_seconds: int):
        self.limit = max(1, limit)
        self.window_seconds = max(60, window_seconds)

    def check_and_consume(self, state: dict, now: float | None = None) -> RateLimitResult:
        now = time.time() if now is None else now
        started_at = float(state.get("rate_window_started_at", 0.0) or 0.0)
        used = int(state.get("rate_window_used", 0) or 0)

        if not started_at or now - started_at >= self.window_seconds:
            started_at = now
            used = 0

        if used >= self.limit:
            retry = max(1, int(self.window_seconds - (now - started_at)))
            state["rate_window_started_at"] = started_at
            state["rate_window_used"] = used
            return RateLimitResult(False, used, self.limit, retry)

        used += 1
        state["rate_window_started_at"] = started_at
        state["rate_window_used"] = used
        return RateLimitResult(True, used, self.limit)
