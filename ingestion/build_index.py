"""
build_index.py — Pipeline ingestion: PDF → teks bersih → chunk → embedding →
FAISS index.

Usage:
    # Lihat hasil chunking tanpa memanggil API sama sekali (gratis)
    python ingestion/build_index.py --dry-run

    # Bangun index (menimpa index yang ada, memakai kuota embedding)
    python ingestion/build_index.py --yes

    # Dokumen lain
    python ingestion/build_index.py --pdf path/to/dokumen.pdf --yes

Selain `faiss.index` dan `metadata.json`, pipeline ini juga menulis
`index_info.json` yang mencatat asal-usul index: hash PDF sumber, model
embedding, task type, dan parameter chunking. Berkat catatan itu aplikasi
tahu index sudah usang bila PDF berubah, dan retriever tahu task type mana
yang harus dipakai saat meng-embed pertanyaan pengguna.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Tambahkan root project ke path agar modul lain bisa diimpor saat script
# dijalankan langsung.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cli_utils import enable_utf8_stdout
from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    INDEX_DIR,
    MIN_CHUNK_SIZE,
)
from ingestion.chunking import MAX_PAGES_PER_CHUNK, Chunk, chunk_pages
from ingestion.extract_text import ExtractionReport, extract_text_with_report

DEFAULT_PDF_PATH = (
    DATA_DIR / "Buku-Pedoman-Akademik-Fakultas-Teknologi-Industri-2025-2026.pdf"
)


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def file_sha256(path: Path) -> str:
    """Hash SHA-256 sebuah file, dibaca bertahap agar hemat memori."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_faiss_index(embeddings: list[list[float]]):
    """
    Bangun FAISS index dari embedding vector.

    Memakai `IndexFlatIP` dengan vector yang dinormalisasi L2, sehingga inner
    product setara cosine similarity. Index flat juga memungkinkan
    `reconstruct()`, yang dipakai retriever untuk menghitung skor dense
    kandidat yang hanya ditemukan BM25.
    """
    import faiss

    vectors = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    print(f"✅ FAISS index dibangun: {index.ntotal} vector, dimensi {vectors.shape[1]}")
    return index


def chunks_to_metadata(chunks: list[Chunk]) -> list[dict]:
    """
    Susun metadata yang disimpan berdampingan dengan index.

    `heading_path` disimpan agar retriever dapat merekonstruksi teks yang
    diembed (untuk BM25) dan menampilkan sitasi berjenjang tanpa perlu
    menyimpan teks embedding dua kali.
    """
    return [
        {
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "page_numbers": chunk.page_numbers,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_title": chunk.section_title,
            "heading_path": chunk.heading_path,
            "char_count": chunk.char_count,
            "estimated_tokens": chunk.estimated_tokens,
        }
        for chunk in chunks
    ]


def build_index_info(
    pdf_path: Path,
    chunks: list[Chunk],
    pages: list[tuple[int, str]],
    report: ExtractionReport,
    embedding_dimension: int,
    task_type: str,
) -> dict:
    """Catatan asal-usul index, disimpan sebagai `index_info.json`."""
    spans = [len(chunk.page_numbers) for chunk in chunks]
    all_pages = sorted({page for chunk in chunks for page in chunk.page_numbers})

    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_pdf": {
            "name": pdf_path.name,
            "sha256": file_sha256(pdf_path),
            "size_bytes": pdf_path.stat().st_size,
            "pages_with_text": len(pages),
        },
        "embedding": {
            "model": EMBEDDING_MODEL,
            "task_type": task_type,
            "dimension": embedding_dimension,
            "includes_heading_path": True,
        },
        # Retriever membaca kunci ini untuk memilih task type saat meng-embed
        # pertanyaan. Query dan dokumen wajib memakai task type yang cocok.
        "embedding_task_type": task_type,
        "chunking": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "min_chunk_size": MIN_CHUNK_SIZE,
            "max_pages_per_chunk": MAX_PAGES_PER_CHUNK,
        },
        "stats": {
            "total_chunks": len(chunks),
            "total_pages": len(all_pages),
            "page_range": f"{all_pages[0]}-{all_pages[-1]}" if all_pages else "N/A",
            "max_pages_per_chunk": max(spans) if spans else 0,
            "avg_pages_per_chunk": round(sum(spans) / len(spans), 2) if spans else 0,
            "single_page_chunks": sum(1 for s in spans if s == 1),
            "avg_chars_per_chunk": (
                round(sum(c.char_count for c in chunks) / len(chunks)) if chunks else 0
            ),
        },
        "extraction": {
            "lines_raw": report.lines_raw,
            "lines_kept": report.lines_kept,
            "dropped": dict(report.dropped),
            "pages_empty_after_cleaning": report.pages_empty_after_cleaning,
            "noisy_pages": [
                {"page": page, "noise_ratio": round(ratio, 3)}
                for page, ratio in report.noisy_pages
            ],
        },
    }


def save_index(index, metadata: list[dict], info: dict, index_dir: Path) -> None:
    """Tulis FAISS index, metadata, dan catatan build ke disk."""
    import faiss

    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    print(f"✅ FAISS index disimpan: {index_path}")

    metadata_path = index_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✅ Metadata disimpan: {metadata_path} ({len(metadata)} entri)")

    info_path = index_dir / "index_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"✅ Catatan build disimpan: {info_path}")


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

def prepare_chunks(pdf_path: Path) -> tuple[list[Chunk], list[tuple[int, str]], ExtractionReport]:
    """
    Jalankan tahap yang tidak memerlukan API: ekstraksi teks dan chunking.

    Dipisah agar `--dry-run` bisa memakai jalur kode yang persis sama dengan
    build sungguhan, tanpa mengeluarkan biaya sepeser pun.
    """
    print(f"\n📖 Step 1: Mengekstrak teks dari '{pdf_path.name}'...")
    pages, report = extract_text_with_report(pdf_path)
    if not pages:
        print("❌ Tidak ada teks yang bisa diekstrak dari PDF!")
        sys.exit(1)

    print("\n✂️  Step 2: Membagi teks menjadi chunk...")
    chunks = chunk_pages(pages)
    if not chunks:
        print("❌ Tidak ada chunk yang dihasilkan!")
        sys.exit(1)

    return chunks, pages, report


def run_ingestion(
    pdf_path: str | Path,
    api_key: str | None = None,
    dry_run: bool = False,
    index_dir: Path = INDEX_DIR,
) -> None:
    """
    Jalankan pipeline ingestion lengkap.

    Args:
        pdf_path: Path ke PDF sumber.
        api_key: Override API key Gemini.
        dry_run: Berhenti setelah chunking, tanpa memanggil API.
        index_dir: Folder tujuan penyimpanan index.
    """
    pdf_path = Path(pdf_path)

    print("=" * 62)
    print("🚀 INGESTION PIPELINE — RAG Chatbot Kampus UAJY")
    if dry_run:
        print("   MODE: dry-run (tanpa panggilan API, tanpa menulis index)")
    print("=" * 62)

    chunks, pages, report = prepare_chunks(pdf_path)

    if dry_run:
        print("\n🔍 Step 3: Contoh chunk hasil pemotongan")
        for chunk in chunks[:3]:
            heading = " > ".join(chunk.heading_path) or "-"
            print(f"\n   --- chunk {chunk.chunk_index} | halaman {chunk.page_label} "
                  f"| {chunk.char_count} karakter ---")
            print(f"   heading : {heading}")
            preview = chunk.text[:220].replace("\n", " ⏎ ")
            print(f"   teks    : {preview}")

        print("\n" + "=" * 62)
        print("✅ DRY-RUN SELESAI — tidak ada biaya API dan index tidak diubah.")
        print(f"   Jalankan tanpa --dry-run untuk meng-embed {len(chunks)} chunk.")
        print("=" * 62)
        return

    # Teks yang diembed menyertakan heading path, bukan hanya isi paragraf.
    print(f"\n🧮 Step 3: Meng-embed {len(chunks)} chunk via Gemini API...")
    from app.llm_client import TASK_TYPE_DOCUMENT, embed_texts

    texts = [chunk.embed_text for chunk in chunks]
    embeddings = embed_texts(
        texts,
        task_type=TASK_TYPE_DOCUMENT,
        api_key=api_key,
        progress=True,
    )

    if len(embeddings) != len(chunks):
        print(f"❌ Jumlah embedding ({len(embeddings)}) tidak sama dengan "
              f"jumlah chunk ({len(chunks)}). Index dibatalkan.")
        sys.exit(1)

    print("\n💾 Step 4: Membangun dan menyimpan index...")
    index = build_faiss_index(embeddings)
    metadata = chunks_to_metadata(chunks)
    info = build_index_info(
        pdf_path=pdf_path,
        chunks=chunks,
        pages=pages,
        report=report,
        embedding_dimension=len(embeddings[0]),
        task_type=TASK_TYPE_DOCUMENT,
    )
    save_index(index, metadata, info, index_dir)

    stats = info["stats"]
    print("\n" + "=" * 62)
    print("✅ INGESTION SELESAI")
    print(f"   📄 PDF            : {pdf_path.name}")
    print(f"   📝 Halaman        : {len(pages)}")
    print(f"   ✂️  Chunk          : {len(chunks)}")
    print(f"   🎯 Halaman/chunk  : maks {stats['max_pages_per_chunk']}, "
          f"rata-rata {stats['avg_pages_per_chunk']}")
    print(f"   🧮 Dimensi vector : {info['embedding']['dimension']}")
    print(f"   💾 Folder index   : {index_dir}")
    print("=" * 62)


def _confirm_overwrite(index_dir: Path, assume_yes: bool) -> None:
    """
    Cegah penimpaan index secara tidak sengaja.

    Membangun ulang index memakan kuota embedding API dan menghapus index
    yang sudah ada, jadi butuh persetujuan eksplisit.
    """
    existing = [
        path for path in (
            index_dir / "faiss.index",
            index_dir / "metadata.json",
        ) if path.exists()
    ]
    if not existing or assume_yes:
        return

    print("⚠️  Index yang sudah ada akan DITIMPA:")
    for path in existing:
        print(f"      {path}")
    print("\n   Proses ini juga memakai kuota embedding Gemini API.")
    print("   Tambahkan flag --yes untuk melanjutkan, atau --dry-run untuk")
    print("   melihat hasil chunking tanpa biaya dan tanpa mengubah apa pun.")
    sys.exit(1)


def main() -> None:
    enable_utf8_stdout()

    parser = argparse.ArgumentParser(
        description="Bangun FAISS index dari dokumen PDF kampus",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=str(DEFAULT_PDF_PATH),
        help=f"Path ke file PDF (default: {DEFAULT_PDF_PATH.name})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google Gemini API key (default: env GEMINI_API_KEY atau secrets.toml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hanya ekstraksi + chunking, tanpa panggilan API dan tanpa menulis index",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Setujui penimpaan index yang sudah ada",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ File PDF tidak ditemukan: {pdf_path}")
        sys.exit(1)

    if not args.dry_run:
        _confirm_overwrite(INDEX_DIR, assume_yes=args.yes)

        # Gagal lebih awal bila API key belum ada, sebelum memproses PDF.
        from app.llm_client import LLMConfigError, resolve_api_key

        try:
            resolve_api_key(args.api_key)
        except LLMConfigError as e:
            print(f"❌ {e}")
            sys.exit(1)

    run_ingestion(pdf_path, api_key=args.api_key, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
