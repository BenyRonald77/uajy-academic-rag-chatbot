"""
cli_utils.py — Utilitas kecil untuk script command line.
"""

from __future__ import annotations

import sys


def enable_utf8_stdout() -> None:
    """
    Paksa stdout/stderr memakai UTF-8.

    Di Windows, stdout Python memakai code page lokal (cp1252) ketika
    outputnya diarahkan ke pipe atau file. Emoji pada laporan progres
    membuat script gagal dengan `UnicodeEncodeError` — bukan karena logikanya
    salah, hanya karena encoding terminal. Pemanggilan ini menghilangkan
    kelas kegagalan tersebut.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
