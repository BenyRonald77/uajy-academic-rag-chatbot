"""
build_index.py — Jalankan ingestion pipeline: PDF → chunks → embeddings → FAISS index.

Usage:
    python ingestion/build_index.py
    python ingestion/build_index.py --pdf path/to/custom.pdf

Script ini dijalankan SEKALI saat setup atau saat dokumen diperbarui.
Hasilnya disimpan di folder index/ dan di-deploy bersama app.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Tambahkan root project ke path agar bisa import module lain
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.extract_text import extract_text_from_pdf
from ingestion.chunking import chunk_pages, Chunk

# ──────────────────────────────────────────────
# Konfigurasi
# ──────────────────────────────────────────────

DEFAULT_PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "Buku-Pedoman-Akademik-Fakultas-Teknologi-Industri-2025-2026.pdf"
INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768  # dimensi output gemini-embedding-001
BATCH_SIZE = 10  # jumlah chunk per batch embedding request (kecil untuk hindari rate limit)
BATCH_DELAY = 2  # detik delay antar batch


def get_embeddings(texts: list[str], api_key: str) -> list[list[float]]:
    """
    Dapatkan embeddings dari Gemini Embedding API.

    Args:
        texts: List teks yang akan di-embed.
        api_key: Google Gemini API key.

    Returns:
        List of embedding vectors.
    """
    import time
    from google import genai

    client = genai.Client(api_key=api_key)
    all_embeddings = []
    total_batches = (len(texts) - 1) // BATCH_SIZE + 1

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"   Embedding batch {batch_num}/{total_batches} "
              f"({len(batch)} chunks)...")

        # Retry logic untuk handle rate limit
        max_retries = 5
        for attempt in range(max_retries):
            try:
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                )
                for embedding in result.embeddings:
                    all_embeddings.append(embedding.values)
                break  # berhasil, lanjut ke batch berikutnya
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    wait_time = BATCH_DELAY * (2 ** attempt)  # exponential backoff
                    print(f"   >> Rate limit, menunggu {wait_time}s (percobaan {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise  # error lain, langsung raise

        # Delay antar batch untuk hindari rate limit
        if batch_num < total_batches:
            time.sleep(BATCH_DELAY)

    return all_embeddings


def build_faiss_index(embeddings: list[list[float]]) -> "faiss.IndexFlatIP":
    """
    Bangun FAISS index dari embedding vectors.
    Menggunakan Inner Product (cosine similarity setelah normalisasi).
    """
    import faiss

    dimension = len(embeddings[0])
    vectors = np.array(embeddings, dtype=np.float32)

    # Normalisasi untuk cosine similarity
    faiss.normalize_L2(vectors)

    # Buat index
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    print(f"✅ FAISS index dibangun: {index.ntotal} vectors, dimensi {dimension}")
    return index


def save_index(index, metadata: list[dict], index_dir: Path) -> None:
    """Simpan FAISS index dan metadata ke disk."""
    import faiss

    index_dir.mkdir(parents=True, exist_ok=True)

    # Simpan FAISS index
    index_path = index_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    print(f"✅ FAISS index disimpan: {index_path}")

    # Simpan metadata
    metadata_path = index_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✅ Metadata disimpan: {metadata_path} ({len(metadata)} entries)")


def run_ingestion(pdf_path: str, api_key: str) -> None:
    """
    Jalankan full ingestion pipeline.

    Args:
        pdf_path: Path ke file PDF sumber.
        api_key: Google Gemini API key.
    """
    pdf_path = Path(pdf_path)

    print("=" * 60)
    print("🚀 INGESTION PIPELINE — RAG Chatbot Kampus UAJY")
    print("=" * 60)

    # Step 1: Ekstrak teks
    print(f"\n📖 Step 1: Mengekstrak teks dari '{pdf_path.name}'...")
    pages = extract_text_from_pdf(str(pdf_path))

    if not pages:
        print("❌ Tidak ada teks yang bisa diekstrak dari PDF!")
        sys.exit(1)

    # Step 2: Chunking
    print(f"\n✂️  Step 2: Membagi teks menjadi chunks...")
    chunks = chunk_pages(pages)

    if not chunks:
        print("❌ Tidak ada chunk yang dihasilkan!")
        sys.exit(1)

    # Step 3: Generate embeddings
    print(f"\n🧮 Step 3: Menggenerate embeddings via Gemini API...")
    texts = [chunk.text for chunk in chunks]
    embeddings = get_embeddings(texts, api_key)

    # Step 4: Build & save FAISS index
    print(f"\n💾 Step 4: Membangun dan menyimpan FAISS index...")
    index = build_faiss_index(embeddings)

    # Siapkan metadata
    metadata = []
    for chunk in chunks:
        metadata.append({
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "page_numbers": chunk.page_numbers,
            "section_title": chunk.section_title,
            "char_count": chunk.char_count,
            "estimated_tokens": chunk.estimated_tokens,
        })

    save_index(index, metadata, INDEX_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("✅ INGESTION SELESAI!")
    print(f"   📄 PDF         : {pdf_path.name}")
    print(f"   📝 Halaman     : {len(pages)}")
    print(f"   ✂️  Chunks      : {len(chunks)}")
    print(f"   🧮 Embeddings  : {len(embeddings)}")
    print(f"   💾 Index dir   : {INDEX_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index dari dokumen PDF kampus")
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(DEFAULT_PDF_PATH),
        help=f"Path ke file PDF (default: {DEFAULT_PDF_PATH})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google Gemini API key (atau set di .streamlit/secrets.toml)",
    )
    args = parser.parse_args()

    # Coba baca API key dari berbagai sumber
    api_key = args.api_key

    if not api_key:
        # Coba baca dari .streamlit/secrets.toml
        secrets_path = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            import tomllib
            with open(secrets_path, "rb") as f:
                secrets = tomllib.load(f)
            api_key = secrets.get("GEMINI_API_KEY")

    if not api_key or api_key == "MASUKKAN_API_KEY_GEMINI_ANDA_DI_SINI":
        print("❌ API key belum diatur!")
        print("   Cara 1: python build_index.py --api-key YOUR_KEY")
        print("   Cara 2: Edit .streamlit/secrets.toml")
        sys.exit(1)

    # Jalankan pipeline
    run_ingestion(args.pdf, api_key)
