"""
main_streamlit.py — Antarmuka Chatbot RAG Dokumen Kampus UAJY.

Tampilannya mengikuti bahasa desain portal akademik kampus: latar biru sangat
muda, kartu putih bergaris tipis, judul bagian berupa header strip, sidebar
putih dengan label kelompok navigasi, dan aksen tunggal cyan. Definisi
gayanya ada di `app/theme.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Tambahkan root project ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.answer_guard import is_safe_answer
from app.citations import (
    citation_pages_are_in_context,
    extract_cited_pages,
    strip_inline_source_lines,
)
from app.config import (
    DEFAULT_RETRIEVAL_CONFIG,
    EMBEDDING_MODEL,
    LLM_MODEL,
    UTILITY_MODEL,
    RetrievalConfig,
)
from app.llm_client import LLMConfigError, call_llm, effective_model
from app.pdf_viewer import render_pdf_viewer, render_source_references
from app.prompt_builder import build_no_context_response, build_prompt
from app.public_config import (
    app_mode,
    is_operator_mode,
    public_rate_limit,
    public_rate_window_seconds,
)
from app.rate_limit import SessionRateLimiter
from app.retrieval import (
    DocumentRetriever,
    RetrievalOutcome,
    build_source_references,
    format_sources,
)
from app.theme import (
    inject_theme,
    nav_label,
    render_brand,
    render_footer,
    render_lead,
    render_status_badge,
    render_welcome,
)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Chatbot | Dokumen Kampus UAJY",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tema visual mengikuti bahasa desain portal akademik kampus.
# Definisinya ada di app/theme.py agar berkas ini tetap berisi alur
# aplikasi, bukan ratusan baris CSS.
inject_theme()


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


# ──────────────────────────────────────────────
# Panel Debug Retrieval
# ──────────────────────────────────────────────

def outcome_to_debug(outcome: RetrievalOutcome) -> dict:
    """
    Ringkas hasil retrieval menjadi data yang bisa disimpan di riwayat chat.

    `RetrievalOutcome` memuat objek yang tidak ramah untuk disimpan di
    session state, jadi hanya angka dan teks yang diambil. Dengan begini
    panel debug tetap bisa dirender ulang setiap kali Streamlit menjalankan
    skrip dari awal.
    """
    used = {candidate.chunk_index for candidate in outcome.contexts}

    banyak_dokumen = len({
        c.source_document for c in outcome.candidates if c.source_document
    }) > 1

    rows = []
    for candidate in outcome.candidates:
        rows.append({
            "Dipakai": "✅" if candidate.chunk_index in used else "",
            "Chunk": candidate.chunk_index,
            **({"Dokumen": candidate.document_label[:24]} if banyak_dokumen else {}),
            "Halaman": candidate.page_label,
            # Panel debug menampilkan jalur lengkap agar hierarki bisa diaudit;
            # sitasi ke pengguna hanya memakai heading terdalam.
            "Bagian": candidate.heading_path_label or "-",
            "Ditemukan oleh": candidate.retrieved_by,
            "Dense (cosine)": round(candidate.dense_score, 3),
            "BM25": round(candidate.lexical_score, 2),
            "RRF": round(candidate.fusion_score, 5),
            "Rerank (0-10)": (
                round(candidate.rerank_score, 1)
                if candidate.rerank_score is not None else None
            ),
            "Cakupan istilah": f"{candidate.lexical_coverage:.0%}",
            "Istilah cocok": ", ".join(candidate.matched_terms) or "-",
        })

    return {
        "rows": rows,
        "gate": dict(outcome.gate),
        "timings_ms": dict(outcome.timings_ms),
        "effective_query": outcome.effective_query,
        "was_rewritten": outcome.was_rewritten,
        "refused": outcome.refused,
        "refusal_reason": outcome.refusal_reason,
        "rerank_failed": outcome.rerank_failed,
    }


def render_retrieval_debug(debug: dict) -> None:
    """Tampilkan rincian skor tiap tahap pipeline retrieval."""
    if not debug:
        return

    rows = debug.get("rows") or []
    label = f"🔬 Rincian retrieval ({len(rows)} kandidat)"

    with st.expander(label, expanded=False):
        timings = debug.get("timings_ms") or {}
        if timings:
            stage_names = {
                "rewrite_ms": "Query rewriting",
                "dense_ms": "Dense (embed + FAISS)",
                "lexical_ms": "BM25",
                "fusion_ms": "Fusi RRF",
                "rerank_ms": "Reranker LLM",
            }
            cols = st.columns(len(timings) + 1)
            for col, (key, value) in zip(cols, timings.items()):
                col.metric(stage_names.get(key, key), f"{value:.0f} ms")
            cols[-1].metric("Total", f"{sum(timings.values()):.0f} ms")

        gate = debug.get("gate") or {}
        if gate:
            st.caption(
                "**Gate relevansi** — "
                f"kemiripan tertinggi {gate.get('max_dense_score', 0):.3f} "
                f"(ambang {gate.get('dense_threshold', 0):.2f}) · "
                f"cakupan istilah tertinggi {gate.get('max_lexical_coverage', 0):.0%} "
                f"(ambang {gate.get('lexical_coverage_threshold', 0):.0%}) · "
                f"jalur dense {'lolos' if gate.get('dense_pass') else 'gagal'}, "
                f"jalur leksikal {'lolos' if gate.get('lexical_pass') else 'gagal'}"
                + (
                    f", reranker {'lolos' if gate.get('rerank_pass') else 'menolak'}"
                    if "rerank_pass" in gate else ""
                )
            )

        if debug.get("rerank_failed"):
            st.error(
                "Reranker gagal dipanggil, jadi urutan hasil fusi dipakai apa "
                "adanya dan gate presisi tidak berjalan untuk pertanyaan ini. "
                "Biasanya karena API sedang sibuk — coba ulangi.",
                icon="🔥",
            )

        if debug.get("refused"):
            st.warning(f"Ditolak: {debug.get('refusal_reason', '-')}")

        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("Tidak ada kandidat yang ditemukan pada kedua jalur pencarian.")


def render_index_freshness(retriever: DocumentRetriever | None) -> None:
    """
    Peringatkan bila index sudah tidak sesuai dengan dokumen sumbernya.

    Ini kegagalan yang paling sulit disadari pengguna: chatbot tetap menjawab
    dengan penuh keyakinan dari dokumen versi lama, lengkap dengan sitasi
    halaman yang tampak sah. Pemeriksaannya memakai hash yang dicatat saat
    index dibangun.
    """
    if not retriever:
        return

    freshness = retriever.freshness()
    if not freshness.has_info or not freshness.is_stale:
        return

    pesan = "\n\n".join(f"- {m}" for m in freshness.messages())
    st.warning(
        f"**Index mungkin sudah usang.**\n\n{pesan}\n\n"
        "Jalankan `python ingestion/build_index.py --yes` untuk membangun ulang.",
        icon="🕒",
    )


def render_hero() -> None:
    """Kartu sambutan, mengikuti pola halaman muka portal akademik kampus."""
    render_welcome(
        "Selamat Datang di Chatbot Akademik FTI",
        "Ajukan pertanyaan seputar <strong>Buku Pedoman Akademik Fakultas Teknologi "
        "Industri Universitas Atma Jaya Yogyakarta</strong>. Setiap jawaban disusun "
        "hanya dari isi dokumen resmi dan selalu menyertakan nomor halaman serta "
        "bagian sumbernya, sehingga dapat Anda periksa sendiri.",
    )


# ──────────────────────────────────────────────
# View 1: Tanya Jawab (Chatbot)
# ──────────────────────────────────────────────

def render_chat_view(
    retriever: DocumentRetriever | None,
    config: RetrievalConfig,
    operator_mode: bool = False,
    rate_limiter: SessionRateLimiter | None = None,
) -> None:
    st.header("Tanya Jawab Akademik")
    render_lead(
        "Ajukan pertanyaan dalam Bahasa Indonesia seputar ketentuan akademik, "
        "SKS, skripsi, cuti, atau yudisium."
    )

    if not retriever:
        if operator_mode:
            st.error(
                "Index dokumen belum ditemukan. Jalankan "
                "`python ingestion/build_index.py` terlebih dahulu."
            )
        else:
            st.error(
                "Layanan chatbot sedang dipersiapkan. Silakan coba lagi nanti "
                "atau hubungi administrator."
            )
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
            if msg["role"] == "assistant":
                if msg.get("source_refs"):
                    render_source_references(
                        msg["source_refs"],
                        key_prefix=f"history_{msg.get('message_id', id(msg))}",
                        public=not operator_mode,
                    )
                elif msg.get("sources"):
                    # Kompatibilitas dengan riwayat lama sebelum source_refs
                    # disimpan secara terstruktur.
                    st.markdown(
                        f'<div class="source-box"><strong>📚 Sumber Dokumen:</strong><br>{msg["sources"]}</div>',
                        unsafe_allow_html=True,
                    )
                if operator_mode and msg.get("debug"):
                    render_retrieval_debug(msg["debug"])

    # Process queued query from preset if any
    input_query = None
    if "queued_query" in st.session_state:
        input_query = st.session_state.pop("queued_query")

    chat_input = st.chat_input("Ketik pertanyaan Anda di sini...")
    query = input_query or chat_input

    if query:
        _handle_query(
            query,
            retriever,
            config,
            operator_mode=operator_mode,
            rate_limiter=rate_limiter,
        )

    render_pdf_viewer()


def _handle_query(
    query: str,
    retriever: DocumentRetriever,
    config: RetrievalConfig,
    operator_mode: bool = False,
    rate_limiter: SessionRateLimiter | None = None,
) -> None:
    if rate_limiter is not None:
        limit_result = rate_limiter.check_and_consume(st.session_state)
        if not limit_result.allowed:
            menit = max(1, (limit_result.retry_after_seconds + 59) // 60)
            st.warning(
                f"Batas pertanyaan sementara tercapai "
                f"({limit_result.limit} pertanyaan per jam). "
                f"Silakan coba lagi sekitar {menit} menit lagi.",
                icon="⏳",
            )
            return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(query)

    # Riwayat tanpa pertanyaan yang baru saja ditambahkan.
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant", avatar="🎓"):
        spinner_text = (
            "Menulis ulang pertanyaan lalu mencari dokumen..."
            if config.use_query_rewrite and chat_history
            else "Mencari potongan dokumen yang relevan..."
        )

        try:
            with st.spinner(spinner_text):
                outcome = retriever.retrieve(
                    query,
                    config=config,
                    chat_history=chat_history,
                )
        except LLMConfigError as error:
            if operator_mode:
                st.error(f"⚠️ {error}")
            else:
                st.error(
                    "Layanan pencarian belum siap. Silakan hubungi administrator.",
                    icon="⚠️",
                )
            st.session_state.messages.pop()
            return
        except Exception as error:
            if operator_mode:
                st.error(f"⚠️ Gagal melakukan pencarian: {error}")
            else:
                st.error(
                    "Layanan pencarian sedang mengalami gangguan. "
                    "Silakan coba lagi nanti.",
                    icon="⚠️",
                )
            st.session_state.messages.pop()
            return

        debug = outcome_to_debug(outcome)

        if outcome.was_rewritten and operator_mode:
            st.caption(
                f"🔁 Pertanyaan ditulis ulang untuk pencarian: "
                f"*{outcome.effective_query}*"
            )

        if outcome.refused:
            response = build_no_context_response()
            st.markdown(response)
            if operator_mode:
                render_retrieval_debug(debug)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "debug": debug,
            })
            return

        with st.spinner("Menyusun jawaban dari dokumen..."):
            system_prompt, user_prompt = build_prompt(
                outcome.original_query, outcome.contexts, chat_history
            )
            response = call_llm(user_prompt, system_instruction=system_prompt)

        if not is_safe_answer(response) or not citation_pages_are_in_context(
            response, outcome.contexts
        ):
            # Circuit breaker public: provider yang mengabaikan prompt atau
            # mengembalikan sitasi di luar konteks tidak boleh ditampilkan
            # sebagai jawaban akademik.
            if operator_mode:
                st.error(
                    "Provider chat tidak mengembalikan jawaban dengan format "
                    "akademik yang valid (sitasi halaman tidak ditemukan atau "
                    "tidak cocok dengan konteks retrieval). Periksa provider."
                )
            else:
                st.warning(
                    "Layanan AI belum mengembalikan jawaban dengan format yang "
                    "valid. Silakan coba lagi nanti atau hubungi administrator.",
                    icon="⚠️",
                )
            response = (
                "Maaf, layanan AI belum dapat menyusun jawaban yang dapat "
                "diverifikasi. Silakan coba lagi nanti."
            )

        cited_pages = set(extract_cited_pages(response))
        source_refs = build_source_references(
            outcome.contexts,
            cited_pages=cited_pages,
        )
        display_response = strip_inline_source_lines(response)
        st.markdown(display_response)

        sources_text = format_sources(
            outcome.contexts,
            cited_pages=cited_pages,
            include_scores=operator_mode,
        )
        if source_refs:
            render_source_references(
                source_refs,
                key_prefix=f"current_{len(st.session_state.messages)}",
                public=not operator_mode,
            )

        if operator_mode:
            render_retrieval_debug(debug)

        st.session_state.messages.append({
            "role": "assistant",
            "content": display_response,
            "sources": sources_text,
            "source_refs": source_refs,
            "message_id": len(st.session_state.messages),
            "debug": debug,
        })


# ──────────────────────────────────────────────
# View 2: Jelajah Dokumen
# ──────────────────────────────────────────────

def render_document_explorer(retriever: DocumentRetriever | None) -> None:
    st.header("Jelajah Dokumen & Index")
    render_lead(
        "Pemeriksaan struktur teks, chunk hasil tokenisasi, dan sebaran halaman "
        "pada dokumen sumber."
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
    col4.metric(
        "Halaman per Chunk",
        f"{info.get('avg_pages_per_chunk', 0)} rata-rata",
        help="Semakin dekat ke 1, semakin presisi sitasi halaman yang bisa "
             "diberikan chatbot. Nilai maksimum saat ini: "
             f"{info.get('max_pages_per_chunk', 0)} halaman.",
    )

    documents = info.get("documents") or []
    if len(documents) > 1:
        st.subheader("Dokumen dalam Index")
        st.dataframe(
            pd.DataFrame([
                {
                    "Dokumen": doc["name"],
                    "Chunk": doc["chunks"],
                    "Halaman": doc["pages"],
                }
                for doc in documents
            ]),
            hide_index=True,
            width="stretch",
        )

    lexical = info.get("lexical") or {}
    if lexical:
        st.caption(
            f"**Index leksikal BM25** — {lexical.get('documents', 0)} dokumen · "
            f"{lexical.get('vocabulary', 0):,} istilah unik · "
            f"rata-rata {lexical.get('avg_tokens_per_doc', 0)} token per chunk · "
            f"k1={lexical.get('k1')}, b={lexical.get('b')}. "
            "Dibangun di memori saat aplikasi dimuat, jadi tidak ada artefak "
            "tambahan yang perlu disimpan."
        )

    st.subheader("Pencarian Isi Dokumen")
    search_term = st.text_input(
        "Filter teks chunk",
        placeholder="Contoh: skripsi, cuti, kurikulum, predikat...",
    )

    display_df = df.copy()
    if search_term and not display_df.empty:
        haystack = display_df["text"].fillna("")
        if "section_title" in display_df.columns:
            haystack = haystack + " " + display_df["section_title"].fillna("")
        display_df = display_df[haystack.str.contains(search_term, case=False, na=False)]

    if not display_df.empty:
        st.caption(f"Menampilkan {min(len(display_df), 50)} dari {len(display_df)} chunk yang cocok")
        table_data = []
        for _, row in display_df.head(50).iterrows():
            pages = row.get("page_numbers") or []
            page_label = "-"
            if len(pages):
                first, last = min(pages), max(pages)
                page_label = str(first) if first == last else f"{first}-{last}"

            heading_path = row.get("heading_path")
            if isinstance(heading_path, list) and heading_path:
                heading = " › ".join(heading_path)
            else:
                heading = row.get("section_title") or "-"

            entri = {
                "Index": row.get("chunk_index"),
                "Halaman": page_label,
                "Bagian / Judul": heading,
                "Karakter": row.get("char_count"),
                "Cuplikan Teks": (row.get("text") or "")[:140] + "...",
            }
            if len(documents) > 1:
                nama = row.get("source_document") or "-"
                entri = {"Dokumen": nama.rsplit(".", 1)[0][:28], **entri}
            table_data.append(entri)
        st.dataframe(pd.DataFrame(table_data), hide_index=True, width="stretch")
    else:
        st.info("Tidak ada chunk yang cocok dengan filter pencarian.")


# ──────────────────────────────────────────────
# View 3: Uji Pertanyaan & Evaluasi
# ──────────────────────────────────────────────

def render_evaluation(retriever: DocumentRetriever | None) -> None:
    st.header("Evaluasi Kualitas RAG")
    render_lead(
        "Uji ketepatan retrieval dan penolakan pertanyaan di luar cakupan dokumen."
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

    st.subheader("Pipeline Retrieval")
    st.markdown(
        """
        Setiap pertanyaan melewati tahapan berikut:

        1. **Query rewriting** — pertanyaan lanjutan seperti *"berapa syaratnya?"* diubah
           menjadi pertanyaan mandiri berdasarkan riwayat percakapan, **sebelum** pencarian
           berjalan. Tanpa langkah ini, konteks yang salah sudah terambil sejak awal.
        2. **Pencarian dua jalur** — *dense* (embedding Gemini + FAISS cosine) menangkap
           kesamaan makna, *BM25 Okapi* menangkap kecocokan istilah eksak seperti
           "IPK 3,50", "144 SKS", atau "Pasal 12".
        3. **Reciprocal Rank Fusion** — kedua peringkat digabung memakai peringkat, bukan
           skor mentah, sehingga cosine (0–1) dan BM25 (tak terbatas) bisa disatukan tanpa
           normalisasi.
        4. **Reranking** — model menilai ulang setiap kandidat dengan melihat pertanyaan dan
           potongan dokumen bersamaan, menilai *kemampuan menjawab* alih-alih sekadar
           kemiripan.
        5. **Gate relevansi berlapis** — ambang kemiripan, cakupan istilah, skor reranker,
           lalu instruksi sistem yang ketat. Pertanyaan di luar cakupan ditolak, bukan
           dijawab dengan karangan.
        """
    )

    st.subheader("Detail Konfigurasi & Stack")
    info = retriever.document_info if retriever else {}
    index_info = info.get("index_info") or {}
    embedding_info = index_info.get("embedding") or {}
    lexical = info.get("lexical") or {}

    # Menangani dua format: source_documents (multi-dokumen) dan source_pdf
    # (dokumen tunggal, index versi lama).
    from app.index_status import indexed_documents

    sumber = indexed_documents(index_info)
    if len(sumber) > 1:
        label_dokumen = f"{len(sumber)} dokumen: " + ", ".join(
            d.get("name", "?").rsplit(".", 1)[0][:32] for d in sumber
        )
    elif sumber:
        label_dokumen = sumber[0].get("name", "-")
    else:
        label_dokumen = "Buku Pedoman Akademik FTI 2025-2026"

    dimension = embedding_info.get("dimension")
    embedding_label = f"Google {EMBEDDING_MODEL}"
    if dimension:
        embedding_label += f" (dimensi {dimension})"

    rows = [
        ("Institusi", "Universitas Atma Jaya Yogyakarta (UAJY)"),
        ("Dokumen Sumber", label_dokumen),
        ("Model LLM Generation", f"Bandel AI · {effective_model(LLM_MODEL)} (via API)"),
        ("Model Rerank & Rewrite", f"Bandel AI · {effective_model(UTILITY_MODEL)} (via API)"),
        ("Model Embedding", embedding_label),
        ("Task Type Embedding", embedding_info.get("task_type") or "generik (index lama)"),
        ("Vector Store", "FAISS IndexFlatIP (cosine similarity)"),
        ("Index Leksikal", f"BM25 Okapi in-memory, {lexical.get('vocabulary', 0):,} istilah unik"),
        ("Total Chunk Terindex", f"{info.get('total_chunks', 0)} chunk"),
        ("Total Halaman Sumber", f"{info.get('total_pages', 0)} halaman"),
        (
            "Presisi Sitasi Halaman",
            f"rata-rata {info.get('avg_pages_per_chunk', 0)} halaman per chunk "
            f"(maksimum {info.get('max_pages_per_chunk', 0)})",
        ),
        ("Framework UI", "Streamlit"),
        (
            "Guardrail Halusinasi",
            "Ambang kemiripan + cakupan istilah leksikal + gate reranker + system prompt ketat",
        ),
    ]

    if index_info.get("built_at"):
        rows.append(("Index Dibangun", index_info["built_at"]))

    st.dataframe(
        pd.DataFrame(rows, columns=["Komponen", "Spesifikasi / Nilai"]),
        hide_index=True,
        width="stretch",
    )

    if not index_info:
        st.info(
            "Index ini dibangun sebelum pencatatan `index_info.json` ada. "
            "Jalankan `python ingestion/build_index.py --yes` untuk membangun ulang "
            "dengan chunking presisi halaman dan embedding yang menyertakan judul bagian."
        )


# ──────────────────────────────────────────────
# Main Entry Point & Sidebar
# ──────────────────────────────────────────────

def _operator_main_legacy() -> None:
    render_hero()

    retriever, error = load_retriever()
    render_index_freshness(retriever)

    with st.sidebar:
        render_brand("Chatbot Akademik", "Pedoman FTI UAJY 2025/2026")

        nav_label("Navigasi")
        page = st.radio(
            "Navigasi",
            ["Tanya Jawab", "Jelajah Dokumen", "Uji Pertanyaan", "Tentang"],
            label_visibility="collapsed",
        )

        st.divider()

        nav_label("Status Sistem")
        if retriever:
            render_status_badge(
                "Vector Store",
                f"● Index Aktif · {retriever.total_chunks} Chunk",
                online=True,
            )
        else:
            render_status_badge("Vector Store", "● Index Belum Siap", online=False)

        # Settings
        # Pemilih dokumen hanya muncul bila index memang memuat lebih dari
        # satu dokumen, agar tidak menambah kendali yang tak berguna.
        selected_documents: tuple[str, ...] = ()
        if retriever:
            available = retriever.source_documents
            if len(available) > 1:
                nav_label("Cakupan Dokumen")
                chosen = st.multiselect(
                    "Cari hanya di dokumen berikut",
                    options=available,
                    default=available,
                    format_func=lambda name: name.rsplit(".", 1)[0][:42],
                    help="Kosongkan untuk mencari di seluruh dokumen.",
                )
                # Memilih semuanya sama artinya dengan tanpa filter, jadi
                # jangan bebani pipeline dengan penyaringan yang sia-sia.
                if chosen and len(chosen) < len(available):
                    selected_documents = tuple(chosen)

        nav_label("Pipeline Retrieval")
        use_hybrid = st.toggle(
            "Hybrid search (BM25 + dense)",
            value=DEFAULT_RETRIEVAL_CONFIG.use_hybrid,
            help="Menggabungkan pencarian makna (embedding) dengan pencocokan "
                 "istilah eksak (BM25) memakai Reciprocal Rank Fusion. Membantu "
                 "pertanyaan yang memuat angka atau nomor pasal.",
        )
        use_rerank = st.toggle(
            "Reranker LLM",
            value=DEFAULT_RETRIEVAL_CONFIG.use_rerank,
            help="Menilai ulang kandidat dengan melihat pertanyaan dan potongan "
                 "dokumen bersamaan. Menambah satu panggilan API per pertanyaan, "
                 "tetapi menaikkan presisi dan menjadi penjaga utama terhadap "
                 "pertanyaan di luar cakupan.",
        )
        if not use_rerank:
            # Pengukuran menunjukkan reranker adalah satu-satunya tahap yang
            # mampu menolak pertanyaan di luar cakupan pada index ini, karena
            # cosine similarity in-scope dan out-of-scope saling tumpang tindih.
            st.warning(
                "Tanpa reranker, penolakan pertanyaan di luar cakupan turun ke "
                "**0%** di tingkat retrieval. Jawaban masih dijaga oleh system "
                "prompt, tetapi konteks yang tidak relevan bisa ikut terkirim.",
                icon="⚠️",
            )
        use_rewrite = st.toggle(
            "Query rewriting",
            value=DEFAULT_RETRIEVAL_CONFIG.use_query_rewrite,
            help="Mengubah pertanyaan lanjutan menjadi pertanyaan mandiri sebelum "
                 "pencarian. Hanya aktif bila pertanyaan terdeteksi bergantung pada "
                 "riwayat percakapan.",
        )

        nav_label("Parameter Retrieval")
        top_k = st.slider(
            "Top-K Konteks", min_value=1, max_value=8,
            value=DEFAULT_RETRIEVAL_CONFIG.top_k,
            help="Jumlah chunk yang dikirim ke LLM sebagai konteks",
        )
        threshold = st.slider(
            "Ambang Relevansi", min_value=0.1, max_value=0.8,
            value=DEFAULT_RETRIEVAL_CONFIG.dense_threshold, step=0.05,
            help="Cosine similarity minimum agar jalur dense dianggap menemukan konteks",
        )

        config = DEFAULT_RETRIEVAL_CONFIG.with_overrides(
            top_k=top_k,
            dense_threshold=threshold,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            use_query_rewrite=use_rewrite,
            document_filter=selected_documents,
        )

        st.divider()
        if st.button("🗑️ Bersihkan Percakapan"):
            st.session_state.messages = []
            st.rerun()

        # Tampilkan model yang benar-benar dipakai. Bila model utama sudah
        # dihapus penyedianya, rantai fallback berpindah diam-diam dan caption
        # ini yang memberi tahu bahwa yang berjalan bukan lagi konfigurasi awal.
        st.caption(
            f"Jawaban: {effective_model(LLM_MODEL)} · "
            f"Rerank/rewrite: {effective_model(UTILITY_MODEL)}"
        )

    if page == "Tanya Jawab":
        render_chat_view(retriever, config, operator_mode=True)
    elif page == "Jelajah Dokumen":
        render_document_explorer(retriever)
    elif page == "Uji Pertanyaan":
        render_evaluation(retriever)
    else:
        render_about(retriever)

    st.divider()
    render_footer(
        "Chatbot Akademik · Fakultas Teknologi Industri "
        "Universitas Atma Jaya Yogyakarta"
    )


# Entry point publik didefinisikan di bawah fungsi operator legacy. Mode public
# adalah default agar deploy VPS aman walaupun APP_MODE lupa dikonfigurasi.



def main() -> None:
    """Jalankan aplikasi dalam mode public atau operator."""
    operator_mode = is_operator_mode()

    if operator_mode:
        # Panel lama lengkap untuk admin/pengembang. Mode ini hanya dibuka
        # dengan APP_MODE=operator di server internal.
        _operator_main_legacy()
        return

    render_hero()
    retriever, error = load_retriever()

    # Public mode: mahasiswa hanya melihat layanan Tanya Jawab. Tidak ada
    # slider, toggle pipeline, explorer, evaluasi, jumlah chunk, skor internal,
    # atau nama model provider.
    limiter = SessionRateLimiter(
        limit=public_rate_limit(),
        window_seconds=public_rate_window_seconds(),
    )
    config = DEFAULT_RETRIEVAL_CONFIG

    with st.sidebar:
        render_brand("Chatbot Akademik", "Layanan Informasi FTI UAJY")
        nav_label("Layanan Mahasiswa")
        st.caption(
            "Tanyakan informasi akademik berdasarkan dokumen resmi Fakultas "
            "Teknologi Industri UAJY."
        )
        st.divider()
        if st.button("🗑️ Bersihkan Percakapan", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption("Gunakan informasi ini sebagai panduan. Untuk keputusan resmi, hubungi bagian akademik.")

    render_chat_view(
        retriever,
        config,
        operator_mode=False,
        rate_limiter=limiter,
    )

    st.divider()
    render_footer(
        "Chatbot Akademik · Fakultas Teknologi Industri "
        "Universitas Atma Jaya Yogyakarta"
    )


if __name__ == "__main__":
    main()
