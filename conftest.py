"""
conftest.py — Konfigurasi pytest tingkat root.

Keberadaan file ini di root membuat pytest menambahkan direktori proyek ke
``sys.path``, sehingga ``import app.…`` dan ``import ingestion.…`` bekerja
tanpa perlu memasang paketnya lebih dulu.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
