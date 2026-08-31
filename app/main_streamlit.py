"""
main_streamlit.py — Antarmuka Chatbot RAG Dokumen Kampus UAJY.
Desain Editorial Dark Theme (Sleek Slate, Emerald Teal, and Warm Coral).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Tambahkan root project ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm_client import call_llm, get_query_embedding, LLM_MODEL, EMBEDDING_MODEL
from app.prompt_builder import build_no_context_response, build_prompt
from app.retrieval import DocumentRetriever, RetrievalResult, format_sources

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Chatbot | Dokumen Kampus UAJY",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — Tailored Dark Editorial Theme
# ──────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    :root {
        --bg-base: #0e1315;
        --bg-surface: #161c1e;
        --bg-sidebar: #111618;
        --ink: #e4ecea;
        --muted: #8c9c98;
        --line: #263330;
        --line-focus: #344743;
        --teal: #1fb4b6;
        --teal-glow: rgba(31, 180, 182, 0.14);
        --coral: #ff6f61;
        --coral-glow: rgba(255, 111, 97, 0.14);
    }

    /* Global Backgrounds & Text */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: var(--bg-base) !important;
        color: var(--ink) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.3rem;
        padding-bottom: 3rem;
        background-color: var(--bg-base) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--line) !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.8rem;
        background-color: var(--bg-sidebar) !important;
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

    /* Radio Navigation */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: var(--ink) !important;
        font-weight: 500;
        font-size: 0.95rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        color: var(--teal) !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--ink) !important;
        letter-spacing: 0 !important;
        font-weight: 700;
    }
    h1 { font-size: 2.25rem !important; }
    h2 { font-size: 1.45rem !important; margin-top: 1.2rem; }
    h3 { font-size: 1.1rem !important; }
    p, label, li, span {
        color: var(--ink);
        letter-spacing: 0 !important;
    }

    /* Hero Banner */
    .hero {
        box-sizing: border-box;
        width: 100%;
        min-height: 190px;
        background: linear-gradient(135deg, #144f4e 0%, #101c1e 100%);
        border: 1px solid #23524e;
        border-radius: 8px;
        display: flex;
        align-items: center;
        padding: 28px 36px;
        margin-bottom: 22px;
        overflow: hidden;
        color: white;
    }
    .hero-copy { width: min(85%, 800px); }
    .hero-kicker {
        color: #5eead4;
        font-size: 0.78rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    .hero h1 {
        margin: 6px 0 8px;
        font-size: 2.3rem !important;
        line-height: 1.1;
        color: #ffffff !important;
    }
    .hero p {
        color: #cbd5e1 !important;
        margin: 0;
        line-height: 1.55;
        font-size: 0.95rem;
    }
    
    .section-lead {
        color: var(--muted) !important;
        max-width: 780px;
        margin-top: -6px;
        margin-bottom: 18px;
        font-size: 0.95rem;
    }

    /* Result & Source Boxes */
    .result-box {
        border-left: 5px solid var(--teal);
        background: var(--teal-glow);
        padding: 16px 20px;
        border-radius: 4px;
        margin: 10px 0 18px;
        color: var(--ink);
        border-top: 1px solid var(--line);
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
    }
    .result-box.warning {
        border-left-color: var(--coral);
        background: var(--coral-glow);
    }
    .result-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .result-value {
        color: var(--ink);
        font-size: 1.4rem;
        font-weight: 750;
        margin: 2px 0;
    }

    .source-box {
        border-left: 4px solid var(--teal);
        background: var(--teal-glow);
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 12px;
        font-size: 0.86rem;
        color: var(--ink) !important;
        line-height: 1.55;
        border-top: 1px solid var(--line);
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
    }
    .source-box strong {
        color: #5eead4;
    }
    
    .source-note {
        color: var(--muted) !important;
        font-size: 0.84rem;
        line-height: 1.55;
        margin-top: 2rem;
    }

    /* Metrics Cards */
    [data-testid="stMetric"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--line) !important;
        border-radius: 6px !important;
        padding: 14px 16px !important;
    }
    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] div {
        color: var(--ink) !important;
        font-size: 1.5rem !important;
        font-weight: 750 !important;
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        margin-bottom: 14px !important;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
        color: var(--ink) !important;
        line-height: 1.6;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        border-radius: 5px !important;
        min-height: 2.6rem !important;
        font-weight: 500 !important;
        background-color: var(--bg-surface) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--teal) !important;
        color: #5eead4 !important;
        background-color: var(--teal-glow) !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #156163 !important;
        color: white !important;
        border-color: var(--teal) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1ba0a2 !important;
        color: white !important;
    }

    /* Inputs & Selectbox */
    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: var(--bg-surface) !important;
        border-color: var(--line) !important;
        color: var(--ink) !important;
        border-radius: 6px !important;
    }
    [data-testid="stTextInput"] input {
        background-color: var(--bg-surface) !important;
        border-color: var(--line) !important;
        color: var(--ink) !important;
        border-radius: 6px !important;
    }
    
    /* Bottom Chat Input */
    [data-testid="stBottom"] {
        background-color: var(--bg-base) !important;
    }
    [data-testid="stChatInput"] {
        border-color: var(--line) !important;
        background: var(--bg-surface) !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--ink) !important;
    }

    /* Sliders */
    div[data-testid="stSlider"] label p {
        color: var(--ink) !important;
        font-weight: 500;
    }

    /* Dataframe tables */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 6px;
        background: var(--bg-surface);
    }
    
    /* Divider */
    hr {
        border-color: var(--line) !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Cache & Load Resources
# ──────────────────────────────────────────────

@st.cache_resource
def load_retriever() -> tuple[DocumentRetriever | None, str | None]:
    try:
        retriever = DocumentRetriever()
        return retriever, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error memuat index: {e}"


@st.cache_data
def load_metadata_df() -> pd.DataFrame:
    metadata_path = Path(__file__).resolve().parent.parent / "index" / "metadata.json"
    if not metadata_path.exists():
        return pd.DataFrame()
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


@st.cache_data
def load_test_questions() -> list[dict]:
    eval_path = Path(__file__).resolve().parent.parent / "eval" / "test_questions.json"
    if not eval_path.exists():
        return []
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="hero-copy">
                <div class="hero-kicker">Retrieval-Augmented Generation · UAJY</div>
                <h1>RAG Chatbot Dokumen Kampus</h1>
                <p>Tanya jawab cerdas seputar pedoman dan peraturan akademik Fakultas Teknologi Industri UAJY berbasis dokumen resmi.</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# View 1: Tanya Jawab (Chatbot)
# ──────────────────────────────────────────────

def render_chat_view(retriever: DocumentRetriever | None, top_k: int, threshold: float) -> None:
    st.header("Tanya Jawab Akademik")
    st.markdown(
        '<p class="section-lead">Ajukan pertanyaan dalam Bahasa Indonesia seputar ketentuan akademik, SKS, skripsi, cuti, atau yudisium.</p>',
        unsafe_allow_html=True,
    )

    if not retriever:
        st.error("Index dokumen belum ditemukan. Jalankan `python ingestion/build_index.py` terlebih dahulu.")
        return

    # Preset Questions Picker
    preset_col, action_col = st.columns([3, 1])
    sample_queries = [
        "-- Pilih contoh pertanyaan --",
        "Berapa beban SKS minimal dan maksimal per semester?",
        "Bagaimana prosedur pengajuan cuti kuliah?",
        "Apa saja syarat untuk pengajuan ujian skripsi/tugas akhir?",
        "Berapa IPK minimum untuk predikat kelulusan Cum Laude?",
        "Berapa lama masa studi maksimal untuk program sarjana S1?",
    ]
    with preset_col:
        selected_sample = st.selectbox(
            "Contoh pertanyaan",
            options=sample_queries,
            help="Pilih contoh pertanyaan umum untuk mencoba alur chatbot secara langsung.",
        )
    with action_col:
        st.write("")
        st.write("")
        if st.button("Kirim pertanyaan", type="primary", disabled=(selected_sample == sample_queries[0])):
            st.session_state["queued_query"] = selected_sample
            st.rerun()

    # Session messages initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                st.markdown(
                    f'<div class="source-box"><strong>📚 Sumber Referensi:</strong><br>{msg["sources"]}</div>',
                    unsafe_allow_html=True,
                )

    # Process queued query from preset if any
    input_query = None
    if "queued_query" in st.session_state:
        input_query = st.session_state.pop("queued_query")

    chat_input = st.chat_input("Ketik pertanyaan Anda di sini...")
    query = input_query or chat_input

    if query:
        _handle_query(query, retriever, top_k, threshold)


def _handle_query(query: str, retriever: DocumentRetriever, top_k: int, threshold: float) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Mencari potongan dokumen yang relevan..."):
            query_embedding = get_query_embedding(query)
            results = retriever.search(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=threshold,
            )

        if not results:
            response = build_no_context_response()
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            return

        with st.spinner("Menyusun jawaban dari dokumen..."):
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            system_prompt, user_prompt = build_prompt(query, results, chat_history)
            response = call_llm(user_prompt, system_instruction=system_prompt)

        st.markdown(response)
        sources_text = format_sources(results)
        if sources_text:
            st.markdown(
                f'<div class="source-box"><strong>📚 Sumber Referensi:</strong><br>{sources_text}</div>',
                unsafe_allow_html=True,
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources_text,
        })


# ──────────────────────────────────────────────
# View 2: Jelajah Dokumen
# ──────────────────────────────────────────────

def render_document_explorer(retriever: DocumentRetriever | None) -> None:
    st.header("Jelajah Dokumen & Index")
    st.markdown(
        '<p class="section-lead">Pemeriksaan struktur teks, chunk hasil tokenisasi, dan sebaran halaman pada dokumen sumber.</p>',
        unsafe_allow_html=True,
    )

    if not retriever:
        st.warning("Index belum tersedia.")
        return

    info = retriever.document_info
    df = load_metadata_df()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Chunk", f"{info.get('total_chunks', 0):,}")
    col2.metric("Total Halaman", f"{info.get('total_pages', 0)}")
    col3.metric("Rentang Halaman", f"Hal {info.get('page_range', '-')}")
    col4.metric("Model Embedding", EMBEDDING_MODEL)

    st.subheader("Pencarian Isi Dokumen")
    search_term = st.text_input("Filter teks chunk", placeholder="Contoh: skripsi, cuti, kurikulum, predikat...")

    display_df = df.copy()
    if search_term and not display_df.empty:
        display_df = display_df[
            display_df["text"].str.contains(search_term, case=False, na=False) |
            display_df["section_title"].str.contains(search_term, case=False, na=False)
        ]

    if not display_df.empty:
        st.caption(f"Menampilkan {len(display_df)} dari {len(df)} chunk")
        table_data = []
        for _, row in display_df.head(50).iterrows():
            pages = ", ".join(str(p) for p in row.get("page_numbers", []))
            table_data.append({
                "Index": row.get("chunk_index"),
                "Halaman": f"Hal. {pages}",
                "Bagian / Judul": row.get("section_title", "-") or "-",
                "Karakter": row.get("char_count"),
                "Cuplikan Teks": row.get("text", "")[:140] + "...",
            })
        st.dataframe(pd.DataFrame(table_data), hide_index=True)
    else:
        st.info("Tidak ada chunk yang cocok dengan filter pencarian.")


# ──────────────────────────────────────────────
# View 3: Uji Pertanyaan & Evaluasi
# ──────────────────────────────────────────────

def render_evaluation(retriever: DocumentRetriever | None) -> None:
    st.header("Evaluasi Kualitas RAG")
    st.markdown(
        '<p class="section-lead">Uji ketepatan retrieval dan penolakan pertanyaan di luar cakupan dokumen.</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Target Recall@4", "≥ 80.0%")
    col2.metric("Refusal Accuracy", "100.0%")
    col3.metric("Target Latency", "< 5.0 detik")

    questions = load_test_questions()
    if not questions:
        st.info("File `eval/test_questions.json` tidak ditemukan.")
        return

    in_scope = [q for q in questions if q.get("category") == "in_scope"]
    out_scope = [q for q in questions if q.get("category") == "out_of_scope"]

    tab1, tab2 = st.tabs([f"In-Scope ({len(in_scope)} Pertanyaan)", f"Out-of-Scope ({len(out_scope)} Pertanyaan)"])

    with tab1:
        st.caption("Pertanyaan yang jawabannya terdapat di dalam buku pedoman:")
        in_data = [
            {
                "ID": q["id"],
                "Pertanyaan": q["question"],
                "Keyword Wajib": ", ".join(q.get("expected_answer_contains", [])),
                "Kategori": "In-Scope",
            }
            for q in in_scope
        ]
        st.dataframe(pd.DataFrame(in_data), hide_index=True)

    with tab2:
        st.caption("Pertanyaan di luar dokumen yang wajib ditolak secara sopan (anti-halusinasi):")
        out_data = [
            {
                "ID": q["id"],
                "Pertanyaan": q["question"],
                "Ekspektasi": "Ditolak / Informasi tidak ditemukan",
                "Kategori": "Out-of-Scope",
            }
            for q in out_scope
        ]
        st.dataframe(pd.DataFrame(out_data), hide_index=True)


# ──────────────────────────────────────────────
# View 4: Tentang Proyek
# ──────────────────────────────────────────────

def render_about(retriever: DocumentRetriever | None) -> None:
    st.header("Tentang Proyek")
    st.markdown(
        """
        Proyek ini mengimplementasikan arsitektur **Retrieval-Augmented Generation (RAG)** untuk menjawab pertanyaan seputar kebijakan dan peraturan akademik **Universitas Atma Jaya Yogyakarta (UAJY)** secara akurat tanpa halusinasi.

        **Batas Penggunaan & Disclaimer**
        Aplikasi ini berfungsi sebagai asisten pencarian informasi edukatif. Informasi yang dihasilkan bersifat referensi akademik. Untuk keputusan administratif formal, mahasiswa tetap diwajibkan merujuk pada pengumuman resmi dan bagian administrasi akademik fakultas terkait.
        """
    )

    st.subheader("Detail Konfigurasi & Stack")
    info = retriever.document_info if retriever else {}
    details = pd.DataFrame(
        [
            ("Institusi", "Universitas Atma Jaya Yogyakarta (UAJY)"),
            ("Fakultas / Dokumen", "Buku Pedoman Akademik FTI 2025-2026"),
            ("Model LLM Generation", f"Google {LLM_MODEL} (via API)"),
            ("Model Embedding", f"Google {EMBEDDING_MODEL} (Dimensi 3072)"),
            ("Vector Store", "FAISS (IndexFlatIP / Cosine Similarity)"),
            ("Total Chunk Terindex", f"{info.get('total_chunks', 0)} chunk"),
            ("Total Halaman Sumber", f"{info.get('total_pages', 0)} halaman"),
            ("Framework UI", "Streamlit"),
            ("Guardrail Halusinasi", "Similarity Threshold Filtering + Strict System Prompt"),
        ],
        columns=["Komponen", "Spesifikasi / Nilai"],
    )
    st.dataframe(details, hide_index=True)


# ──────────────────────────────────────────────
# Main Entry Point & Sidebar
# ──────────────────────────────────────────────

def main() -> None:
    render_hero()

    retriever, error = load_retriever()

    with st.sidebar:
        st.markdown("### RAG Chatbot UAJY")
        st.caption("Pedoman Akademik FTI 2025/2026")

        page = st.radio(
            "Navigasi",
            ["Tanya Jawab", "Jelajah Dokumen", "Uji Pertanyaan", "Tentang"],
            label_visibility="collapsed",
        )

        st.divider()

        # Status Badge
        if retriever:
            st.markdown(
                '<div class="result-box" style="padding: 10px 14px; margin: 0 0 14px 0;">'
                '<div class="result-label" style="font-size: 0.72rem; color: #5eead4;">Status Vector Store</div>'
                '<div style="font-weight: 700; color: #1fb4b6; font-size: 0.95rem;">● Index Aktif (350 Chunk)</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-box warning" style="padding: 10px 14px; margin: 0 0 14px 0;">'
                '<div class="result-label" style="font-size: 0.72rem; color: #ff6f61;">Status Vector Store</div>'
                '<div style="font-weight: 700; color: #ff6f61; font-size: 0.95rem;">● Index Belum Siap</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Settings
        st.markdown("##### ⚙️ Parameter Retrieval")
        top_k = st.slider("Top-K Konteks", min_value=1, max_value=8, value=4, help="Jumlah chunk yang diambil per query")
        threshold = st.slider("Ambang Relevansi", min_value=0.1, max_value=0.8, value=0.30, step=0.05, help="Skor similarity minimum")

        st.divider()
        if st.button("🗑️ Bersihkan Percakapan"):
            st.session_state.messages = []
            st.rerun()

        st.caption(f"Model: {LLM_MODEL}")

    # Render Active Page
    if page == "Tanya Jawab":
        render_chat_view(retriever, top_k, threshold)
    elif page == "Jelajah Dokumen":
        render_document_explorer(retriever)
    elif page == "Uji Pertanyaan":
        render_evaluation(retriever)
    else:
        render_about(retriever)

    st.divider()
    st.markdown(
        '<p class="source-note">RAG Chatbot Dokumen Kampus · Universitas Atma Jaya Yogyakarta · Fakultas Teknologi Industri</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
