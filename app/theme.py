"""
theme.py — Tema visual dan komponen tampilan, mengikuti bahasa desain portal
akademik kampus (SIATMA UAJY).

Ciri desain yang diadopsi:

- Latar halaman biru sangat muda, kartu putih dengan garis tepi tipis.
- Judul bagian berupa *header strip* biru muda, bukan teks polos.
- Sidebar putih dengan label kelompok navigasi berwarna cyan.
- Aksen tunggal cyan, teks utama biru-slate gelap, teks pendukung abu-abu.
- Tabel bergaris dengan baris kepala abu-abu muda.
- Footer terpusat berukuran kecil.

CSS dipisahkan ke modul sendiri agar `main_streamlit.py` tetap berisi alur
aplikasi, bukan ratusan baris gaya. Seluruh warna didefinisikan sebagai
custom property di satu tempat sehingga penyesuaian merek cukup dilakukan
pada blok `:root`.
"""

from __future__ import annotations

import streamlit as st

# ──────────────────────────────────────────────
# Token warna
# ──────────────────────────────────────────────

#: Dipakai juga oleh `.streamlit/config.toml` agar tema bawaan Streamlit
#: (widget, fokus, scrollbar) sejalan dengan CSS di bawah.
PRIMARY = "#1B9DC4"
PAGE_BG = "#F2F9FC"
CARD_BG = "#FFFFFF"
INK = "#2F4858"

THEME_CSS = """
<style>
    :root {
        /* Aksen utama, mengikuti warna wordmark portal akademik. */
        --primary: #1B9DC4;
        --primary-dark: #14809F;
        --primary-soft: #E9F4F9;
        --primary-tint: #F5FBFD;

        --page-bg: #F2F9FC;
        --card-bg: #FFFFFF;
        --card-head: #E9F4F9;
        --sidebar-bg: #FFFFFF;

        --border: #D8E7EF;
        --border-strong: #C3DBE7;
        --table-head: #F1F4F6;

        --ink-strong: #1F3541;
        --ink: #2F4858;
        --muted: #7A8C99;

        --danger: #C8434F;
        --danger-soft: #FDEEF0;
        --warn: #B4761A;
        --warn-soft: #FDF5E7;
        --ok: #2E8B62;
        --ok-soft: #EAF6F0;

        --radius: 8px;
        --shadow: 0 1px 3px rgba(31, 53, 65, 0.06);
    }

    /* ── Dasar ──────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: var(--page-bg) !important;
        color: var(--ink) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     Helvetica, Arial, sans-serif;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    p, label, li, span, div {
        letter-spacing: 0 !important;
    }

    /* ── Sidebar ────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] .block-container,
    [data-testid="stSidebarUserContent"] {
        padding-top: 1.1rem;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5 {
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] hr {
        margin: 0.9rem 0 !important;
        border-color: var(--border) !important;
    }

    /* Wordmark + lambang pada kepala sidebar. */
    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 2px 2px 14px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 4px;
    }
    .brand-mark {
        flex: 0 0 auto;
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: linear-gradient(145deg, var(--primary) 0%, #0F6E8C 100%);
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.12rem;
        box-shadow: 0 2px 5px rgba(20, 128, 159, 0.28);
    }
    .brand-text { line-height: 1.15; min-width: 0; }
    .brand-name {
        font-size: 1.16rem;
        font-weight: 800;
        color: var(--primary) !important;
        letter-spacing: 0.2px !important;
    }
    .brand-sub {
        font-size: 0.72rem;
        color: var(--muted) !important;
        font-weight: 600;
    }

    /* Label kelompok navigasi, mengikuti pola "Navigasi / Nilai / Jadwal". */
    .nav-label {
        color: var(--primary) !important;
        font-size: 0.82rem;
        font-weight: 800;
        margin: 14px 0 6px;
    }

    /* Radio sidebar ditampilkan sebagai daftar menu, bukan tombol radio. */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 1px !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        width: 100%;
        padding: 8px 10px !important;
        border-radius: 6px;
        border-left: 3px solid transparent;
        cursor: pointer;
        transition: background-color 0.15s ease, color 0.15s ease;
    }
    /* Titik radio disembunyikan; status terpilih ditandai latar + garis kiri. */
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label div {
        font-size: 0.94rem !important;
        font-weight: 500 !important;
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: var(--primary-tint);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background-color: var(--primary-soft);
        border-left-color: var(--primary);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) div {
        color: var(--primary-dark) !important;
        font-weight: 700 !important;
    }
    /* Fokus keyboard tetap terlihat meski titik radio disembunyikan. */
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:focus-visible) {
        outline: 2px solid var(--primary);
        outline-offset: 1px;
    }

    /* ── Tipografi ──────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        color: var(--ink-strong) !important;
        font-weight: 700;
    }
    h1 { font-size: 1.9rem !important; }
    h3 { font-size: 1.02rem !important; margin-top: 1.1rem !important; }

    /* Judul bagian tampil sebagai header strip kartu. */
    h2 {
        font-size: 1.12rem !important;
        background: var(--card-head);
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: var(--radius);
        padding: 11px 16px !important;
        margin: 1.5rem 0 0 !important;
    }

    /* ── Kartu sambutan ─────────────────────── */
    .welcome {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 22px 26px;
        margin-bottom: 6px;
    }
    .welcome h1 {
        margin: 0 0 8px !important;
        font-size: 1.72rem !important;
        line-height: 1.2;
    }
    .welcome p {
        color: var(--ink) !important;
        margin: 0;
        font-size: 0.92rem;
        line-height: 1.6;
        max-width: 860px;
    }
    .welcome strong { color: var(--primary-dark) !important; }

    /* Kalimat pengantar di bawah judul bagian. */
    .section-lead {
        color: var(--muted) !important;
        max-width: 820px;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 12px 0 14px !important;
    }

    /* ── Panel & kotak ──────────────────────── */
    .panel {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 14px 18px;
        margin: 10px 0 16px;
        box-shadow: var(--shadow);
    }

    .status-badge {
        background: var(--ok-soft);
        border: 1px solid #BFE3D1;
        border-left: 4px solid var(--ok);
        border-radius: 6px;
        padding: 9px 13px;
        margin-bottom: 12px;
    }
    .status-badge.offline {
        background: var(--danger-soft);
        border-color: #F3CBD0;
        border-left-color: var(--danger);
    }
    .status-label {
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.4px !important;
    }
    .status-value {
        font-size: 0.92rem;
        font-weight: 700;
        color: var(--ok) !important;
    }
    .status-badge.offline .status-value { color: var(--danger) !important; }

    /* Kotak sumber rujukan di bawah setiap jawaban. */
    .source-box {
        background: var(--primary-tint);
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 14px;
        font-size: 0.86rem;
        line-height: 1.6;
        color: var(--ink) !important;
    }
    .source-box strong { color: var(--primary-dark) !important; }

    /* ── Metrik ─────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 13px 16px !important;
        box-shadow: var(--shadow);
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] div {
        color: var(--ink-strong) !important;
        font-size: 1.34rem !important;
        font-weight: 700 !important;
    }

    /* ── Percakapan ─────────────────────────── */
    [data-testid="stChatMessage"] {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 15px 19px !important;
        margin-bottom: 13px !important;
        box-shadow: var(--shadow);
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: var(--ink) !important;
        line-height: 1.65;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
        background: var(--primary-soft) !important;
        border: 1px solid var(--border-strong) !important;
    }

    [data-testid="stBottom"] { background-color: var(--page-bg) !important; }
    [data-testid="stChatInput"] {
        background: var(--card-bg) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stChatInput"] textarea { color: var(--ink) !important; }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(27, 157, 196, 0.12) !important;
    }

    /* ── Tombol ─────────────────────────────── */
    .stButton > button, .stDownloadButton > button {
        background-color: var(--card-bg) !important;
        color: var(--ink) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 6px !important;
        min-height: 2.5rem !important;
        font-weight: 600 !important;
        transition: all 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--primary) !important;
        color: var(--primary-dark) !important;
        background-color: var(--primary-tint) !important;
    }
    .stButton > button[kind="primary"] {
        background-color: var(--primary) !important;
        color: #FFFFFF !important;
        border-color: var(--primary) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--primary-dark) !important;
        color: #FFFFFF !important;
    }

    /* ── Input ──────────────────────────────── */
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stMultiSelect"] div[data-baseweb="select"] {
        background-color: var(--card-bg) !important;
        border-color: var(--border-strong) !important;
        border-radius: 6px !important;
    }
    [data-testid="stTextInput"] input {
        background-color: var(--card-bg) !important;
        border-color: var(--border-strong) !important;
        color: var(--ink) !important;
        border-radius: 6px !important;
    }
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background-color: var(--primary) !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stSlider"] label p,
    [data-testid="stWidgetLabel"] p {
        color: var(--ink) !important;
        font-weight: 600;
    }

    /* ── Tabel ──────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 6px;
        background: var(--card-bg);
        overflow: hidden;
    }
    [data-testid="stDataFrame"] thead tr th {
        background-color: var(--table-head) !important;
        color: var(--ink-strong) !important;
        font-weight: 700 !important;
    }

    /* ── Expander sebagai kartu ─────────────── */
    [data-testid="stExpander"] {
        background: var(--card-bg);
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow);
        margin-top: 12px;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: var(--ink) !important;
    }
    [data-testid="stExpander"] summary:hover { color: var(--primary-dark) !important; }

    /* ── Tab ────────────────────────────────── */
    [data-testid="stTabs"] button[role="tab"] {
        color: var(--muted) !important;
        font-weight: 600 !important;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--primary-dark) !important;
    }

    /* ── Notifikasi ─────────────────────────── */
    [data-testid="stAlert"] {
        border-radius: 6px !important;
        border-left-width: 4px !important;
    }
    [data-testid="stAlert"] p { color: var(--ink) !important; }

    /* ── Footer ─────────────────────────────── */
    .app-footer {
        text-align: center;
        color: var(--muted) !important;
        font-size: 0.82rem;
        line-height: 1.6;
        margin-top: 1.6rem;
    }

    hr { border-color: var(--border) !important; }
</style>
"""


# ──────────────────────────────────────────────
# Komponen tampilan
# ──────────────────────────────────────────────

def inject_theme() -> None:
    """Sisipkan CSS tema. Panggil sekali, sedini mungkin pada skrip."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_brand(name: str, subtitle: str, mark: str = "🎓") -> None:
    """
    Lambang bulat dan wordmark pada kepala sidebar.

    Args:
        name: Nama aplikasi, ditampilkan tebal berwarna aksen.
        subtitle: Keterangan singkat di bawahnya.
        mark: Ikon di dalam lambang bulat.
    """
    st.markdown(
        f"""
        <div class="brand">
            <div class="brand-mark">{mark}</div>
            <div class="brand-text">
                <div class="brand-name">{name}</div>
                <div class="brand-sub">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def nav_label(text: str) -> None:
    """Label kelompok navigasi pada sidebar."""
    st.markdown(f'<div class="nav-label">{text}</div>', unsafe_allow_html=True)


def render_welcome(heading: str, body: str) -> None:
    """Kartu sambutan di bagian atas halaman."""
    st.markdown(
        f'<div class="welcome"><h1>{heading}</h1><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def render_lead(text: str) -> None:
    """Kalimat pengantar di bawah judul bagian."""
    st.markdown(f'<p class="section-lead">{text}</p>', unsafe_allow_html=True)


def render_status_badge(label: str, value: str, online: bool = True) -> None:
    """Penanda status ringkas untuk sidebar."""
    css_class = "status-badge" if online else "status-badge offline"
    st.markdown(
        f'<div class="{css_class}">'
        f'<div class="status-label">{label}</div>'
        f'<div class="status-value">{value}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_footer(text: str) -> None:
    """Footer terpusat di dasar halaman."""
    st.markdown(f'<p class="app-footer">{text}</p>', unsafe_allow_html=True)
