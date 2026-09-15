"""
build_index.py — Pipeline ingestion: PDF → teks bersih → chunk → embedding →
FAISS index.

Usage:
    # Lihat hasil chunking tanpa memanggil API sama sekali (gratis)
    python ingestion/build_index.py --dry-run

    # Bangun index dari seluruh PDF di folder data/
    python ingestion/build_index.py --yes

    # Dokumen tertentu saja (boleh diulang)
    python ingestion/build_index.py --pdf data/pedoman.pdf --pdf data/kalender.pdf --yes

Satu index memuat banyak dokumen. Setiap chunk mencatat dokumen asalnya,
sebab nomor halaman hanya bermakna dalam konteks dokumennya — "halaman 48"
pada pedoman akademik dan pada SK Rektor adalah dua hal berbeda.

Selain `faiss.index` dan `metadata.json`, pipeline ini menulis
`index_info.json` yang mencatat hash SHA-256 setiap dokumen sumber. Berkat
catatan itu aplikasi bisa memberi tahu bahwa index sudah usang ketika dokumen
sumbernya diperbarui.
"""

from __future__ import annotations

import argparse
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
from app.index_status import file_sha256
from ingestion.chunking import MAX_PAGES_PER_CHUNK, Chunk, chunk_pages, print_chunk_stats
from ingestion.extract_text import ExtractionReport, extract_text_with_report


# ──────────────────────────────────────────────
# Penemuan dokumen
# ──────────────────────────────────────────────

def discover_pdfs(paths: list[str] | None, data_dir: Path = DATA_DIR) -> list[Path]:
    """
    Tentukan daftar PDF yang akan diproses.

    Args:
        paths: Path eksplisit dari CLI. Boleh berupa berkas maupun folder.
            Bila ``None``, seluruh PDF di ``data_dir`` dipakai.
        data_dir: Folder dokumen bawaan.

    Returns:
        Daftar path PDF, terurut dan tanpa duplikat.
    """
    if not paths:
        return sorted(data_dir.glob("*.pdf"))

    hasil: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            hasil.extend(sorted(path.glob("*.pdf")))
        else:
            hasil.append(path)

    # Buang duplikat tetapi jaga urutannya.
    unik: list[Path] = []
    terlihat: set[Path] = set()
    for path in hasil:
        resolved = path.resolve()
        if resolved not in terlihat:
            terlihat.add(resolved)
            unik.append(path)
    return unik


# ──────────────────────────────────────────────
# Index & metadata
# ──────────────────────────────────────────────

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
            "source_document": chunk.source_document,
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
    documents: list[dict],
    chunks: list[Chunk],
    embedding_dimension: int,
    task_type: str,
) -> dict:
    """Catatan asal-usul index, disimpan sebagai `index_info.json`."""
    spans = [len(chunk.page_numbers) for chunk in chunks]
    pages_per_doc: dict[str, set[int]] = {}
    for chunk in chunks:
        pages_per_doc.setdefault(chunk.source_document, set()).update(chunk.page_numbers)

    total_pages = sum(len(pages) for pages in pages_per_doc.values())

    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_documents": documents,
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
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "total_pages": total_pages,
            "max_pages_per_chunk": max(spans) if spans else 0,
            "avg_pages_per_chunk": round(sum(spans) / len(spans), 2) if spans else 0,
            "single_page_chunks": sum(1 for s in spans if s == 1),
            "avg_chars_per_chunk": (
                round(sum(c.char_count for c in chunks) / len(chunks)) if chunks else 0
            ),
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

def prepare_chunks(
    pdf_paths: list[Path],
) -> tuple[list[Chunk], list[dict]]:
    """
    Jalankan tahap yang tidak memerlukan API: ekstraksi teks dan chunking.

    Dipisah agar `--dry-run` memakai jalur kode yang persis sama dengan build
    sungguhan, tanpa mengeluarkan biaya sepeser pun.

    Returns:
        Tuple ``(chunks, catatan dokumen)``. ``chunk_index`` bersifat unik
        global sehingga tetap sejajar dengan posisi vector di FAISS.
    """
    semua_chunk: list[Chunk] = []
    catatan: list[dict] = []

    for nomor, pdf_path in enumerate(pdf_paths, start=1):
        print(f"\n{'─' * 62}")
        print(f"📖 Dokumen {nomor}/{len(pdf_paths)}: {pdf_path.name}")
        print(f"{'─' * 62}")

        pages, report = extract_text_with_report(pdf_path)
        if not pages:
            print(f"⚠️  Dilewati: tidak ada teks yang bisa diekstrak.")
            continue

        chunks = chunk_pages(pages)
        if not chunks:
            print(f"⚠️  Dilewati: tidak ada chunk yang dihasilkan.")
            continue

        for chunk in chunks:
            chunk.source_document = pdf_path.name

        semua_chunk.extend(chunks)
        catatan.append({
            "name": pdf_path.name,
            "sha256": file_sha256(pdf_path),
            "size_bytes": pdf_path.stat().st_size,
            "pages_with_text": len(pages),
            "chunks": len(chunks),
            "extraction": {
                "lines_raw": report.lines_raw,
                "lines_kept": report.lines_kept,
                "dropped": dict(report.dropped),
                "pages_empty_after_cleaning": report.pages_empty_after_cleaning,
            },
        })

    # Nomor ulang secara global supaya sejajar dengan urutan vector FAISS.
    for i, chunk in enumerate(semua_chunk):
        chunk.chunk_index = i

    return semua_chunk, catatan


def run_ingestion(
    pdf_paths: list[Path],
    api_key: str | None = None,
    dry_run: bool = False,
    index_dir: Path = INDEX_DIR,
) -> None:
    """
    Jalankan pipeline ingestion lengkap.

    Args:
        pdf_paths: Daftar PDF sumber.
        api_key: Override API key Gemini.
        dry_run: Berhenti setelah chunking, tanpa memanggil API.
        index_dir: Folder tujuan penyimpanan index.
    """
    print("=" * 62)
    print("🚀 INGESTION PIPELINE — RAG Chatbot Kampus UAJY")
    print(f"   Dokumen: {len(pdf_paths)}")
    if dry_run:
        print("   MODE: dry-run (tanpa panggilan API, tanpa menulis index)")
    print("=" * 62)

    chunks, documents = prepare_chunks(pdf_paths)
    if not chunks:
        print("\n❌ Tidak ada chunk yang dihasilkan dari dokumen mana pun!")
        sys.exit(1)

    if len(documents) > 1:
        print(f"\n{'─' * 62}")
        print("📚 Gabungan seluruh dokumen")
        print(f"{'─' * 62}")
        for doc in documents:
            print(f"   {doc['name']}: {doc['chunks']} chunk dari "
                  f"{doc['pages_with_text']} halaman")
        print_chunk_stats(chunks)

    if dry_run:
        print("\n🔍 Contoh chunk hasil pemotongan")
        for chunk in chunks[:3]:
            heading = " > ".join(chunk.heading_path) or "-"
            print(f"\n   --- chunk {chunk.chunk_index} | {chunk.source_document} "
                  f"| halaman {chunk.page_label} | {chunk.char_count} karakter ---")
            print(f"   heading : {heading}")
            print(f"   teks    : {chunk.text[:200].replace(chr(10), ' ⏎ ')}")

        print("\n" + "=" * 62)
        print("✅ DRY-RUN SELESAI — tidak ada biaya API dan index tidak diubah.")
        print(f"   Jalankan tanpa --dry-run untuk meng-embed {len(chunks)} chunk.")
        print("=" * 62)
        return

    # Teks yang diembed menyertakan heading path, bukan hanya isi paragraf.
    print(f"\n🧮 Meng-embed {len(chunks)} chunk via Gemini API...")
    from app.llm_client import TASK_TYPE_DOCUMENT, embed_texts

    embeddings = embed_texts(
        [chunk.embed_text for chunk in chunks],
        task_type=TASK_TYPE_DOCUMENT,
        api_key=api_key,
        progress=True,
    )

    if len(embeddings) != len(chunks):
        print(f"❌ Jumlah embedding ({len(embeddings)}) tidak sama dengan "
              f"jumlah chunk ({len(chunks)}). Index dibatalkan.")
        sys.exit(1)

    print("\n💾 Membangun dan menyimpan index...")
    index = build_faiss_index(embeddings)
    info = build_index_info(
        documents=documents,
        chunks=chunks,
        embedding_dimension=len(embeddings[0]),
        task_type=TASK_TYPE_DOCUMENT,
    )
    save_index(index, chunks_to_metadata(chunks), info, index_dir)

    stats = info["stats"]
    print("\n" + "=" * 62)
    print("✅ INGESTION SELESAI")
    print(f"   📚 Dokumen        : {stats['total_documents']}")
    print(f"   📝 Halaman        : {stats['total_pages']}")
    print(f"   ✂️  Chunk          : {stats['total_chunks']}")
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
        description="Bangun FAISS index dari satu atau beberapa dokumen PDF kampus",
    )
    parser.add_argument(
        "--pdf",
        action="append",
        default=None,
        metavar="PATH",
        help="Berkas PDF atau folder. Boleh diulang. "
             f"Default: seluruh PDF di {DATA_DIR.name}/",
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

    pdf_paths = discover_pdfs(args.pdf)
    if not pdf_paths:
        lokasi = ", ".join(args.pdf) if args.pdf else str(DATA_DIR)
        print(f"❌ Tidak ada berkas PDF yang ditemukan di: {lokasi}")
        sys.exit(1)

    hilang = [p for p in pdf_paths if not p.exists()]
    if hilang:
        for path in hilang:
            print(f"❌ File PDF tidak ditemukan: {path}")
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

    run_ingestion(pdf_paths, api_key=args.api_key, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
