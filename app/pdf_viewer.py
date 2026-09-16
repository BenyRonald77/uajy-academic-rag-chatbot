"""Komponen sumber interaktif dan renderer halaman PDF untuk mahasiswa."""

from __future__ import annotations

import base64
from pathlib import Path

import pymupdf
import streamlit as st

from app.config import DATA_DIR

DEFAULT_PDF_NAME = "Buku-Pedoman-Akademik-Fakultas-Teknologi-Industri-2025-2026.pdf"


@st.cache_data(show_spinner=False)
def _pdf_page_image(path_string: str, page_number: int, zoom: float = 1.5) -> bytes:
    """
    Render satu halaman PDF menjadi PNG.

    Browser PDF viewer/iframe tidak konsisten saat menerima data URL dari
    Streamlit. Dengan merender halaman di server menjadi PNG, isi halaman
    pasti terlihat di semua browser, termasuk VPS tanpa PDF plugin.
    """
    document = pymupdf.open(path_string)
    try:
        if page_number < 1 or page_number > len(document):
            raise ValueError(
                f"Halaman {page_number} di luar dokumen ({len(document)} halaman)."
            )
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(zoom, zoom),
            alpha=False,
        )
        return pixmap.tobytes("png")
    finally:
        document.close()


def _safe_pdf_path(source_document: str) -> Path | None:
    """Resolve dokumen hanya dari folder data, tanpa path traversal."""
    filename = source_document or DEFAULT_PDF_NAME
    candidate = (DATA_DIR / Path(filename).name).resolve()
    data_root = DATA_DIR.resolve()
    if candidate.parent != data_root or not candidate.exists() or candidate.suffix.lower() != ".pdf":
        return None
    return candidate


def _viewer_reference_key(reference: dict) -> str:
    document = reference.get("source_document") or DEFAULT_PDF_NAME
    pages = ",".join(str(page) for page in reference.get("pages", []))
    return f"{document}:{pages}"


def render_source_references(
    references: list[dict],
    key_prefix: str,
    public: bool = True,
) -> None:
    """Render sumber sebagai tombol pembuka viewer halaman."""
    if not references:
        return

    st.markdown(
        '<div class="source-box"><strong>📚 Sumber Dokumen:</strong>'
        '<div style="margin-top:4px;color:var(--muted);font-size:0.82rem;">'
        'Klik halaman untuk menampilkan isi PDF.</div></div>',
        unsafe_allow_html=True,
    )

    for index, reference in enumerate(references):
        label = f"📄 Halaman {reference.get('page_label', '-')}"
        section = reference.get("section_title")
        if section:
            label += f" · {section}"
        if not public and reference.get("score") is not None:
            label += f" · relevansi {reference['score']:.0%}"

        if st.button(
            label,
            key=f"{key_prefix}_source_{index}_{_viewer_reference_key(reference)}",
            type="secondary",
            width="stretch",
            help="Tampilkan halaman PDF yang dikutip",
        ):
            st.session_state["pdf_viewer_reference"] = reference
            st.session_state["pdf_viewer_open"] = True
            st.rerun()


def render_pdf_viewer() -> None:
    """Tampilkan isi halaman PDF terpilih sebagai preview yang selalu terlihat."""
    if not st.session_state.get("pdf_viewer_open"):
        return

    reference = st.session_state.get("pdf_viewer_reference") or {}
    path = _safe_pdf_path(reference.get("source_document", ""))
    if path is None:
        st.error("Dokumen PDF sumber tidak tersedia di server.")
        return

    pages = [int(page) for page in reference.get("pages", []) if str(page).isdigit()]
    if not pages:
        pages = [1]

    st.divider()
    with st.container(border=True):
        title = reference.get("section_title") or "Sumber Dokumen"
        st.subheader(f"📖 Lihat Dokumen — {title}")
        st.caption(
            f"{path.name} · halaman asli: {reference.get('page_label', '-')}. "
            "Preview halaman terpilih:"
        )

        selected_page = st.selectbox(
            "Halaman PDF",
            options=pages,
            index=0,
            key="pdf_viewer_selected_page",
            format_func=lambda page: f"Halaman {page}",
        )

        try:
            page_image = _pdf_page_image(str(path), selected_page)
            encoded = base64.b64encode(page_image).decode("ascii")
            st.markdown(
                f"""
                <div style="background:#ffffff;border:1px solid #d8e7ef;border-radius:8px;padding:18px;text-align:center;">
                  <img src="data:image/png;base64,{encoded}"
                       alt="Halaman {selected_page} dari {path.name}"
                       style="display:block;width:100%;height:auto;margin:0 auto;" />
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Halaman {selected_page} · {path.name}")
        except Exception:
            st.error("Halaman PDF tidak dapat dirender.")

        col_close, col_download = st.columns([1, 1])
        with col_close:
            if st.button("Tutup viewer", key="close_pdf_viewer", width="stretch"):
                st.session_state.pop("pdf_viewer_reference", None)
                st.session_state["pdf_viewer_open"] = False
                st.rerun()
        with col_download:
            st.download_button(
                "Unduh PDF",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/pdf",
                key="download_source_pdf",
                width="stretch",
            )
