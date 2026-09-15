"""
run_eval.py — Evaluasi kualitas retrieval RAG, dijalankan sebagai *ablation
study*.

Usage:
    # Bandingkan dense-only vs hybrid vs hybrid+rerank
    python eval/run_eval.py

    # Hanya satu konfigurasi
    python eval/run_eval.py --mode hybrid_rerank

    # Tanpa uji percakapan lanjutan (hemat panggilan API)
    python eval/run_eval.py --skip-conversational

Kenapa metriknya diperketat
--------------------------
Versi awal skrip ini menganggap sebuah query berhasil bila **salah satu**
kata kunci muncul di gabungan seluruh chunk yang terambil::

    keywords_found = any(kw in all_text for kw in expected)

Ukuran itu terlalu longgar sampai hampir tidak mungkin gagal. Pertanyaan
"Apa sanksi bagi mahasiswa yang melakukan plagiarisme?" dinyatakan lolos
hanya karena kata "sanksi" muncul di suatu tempat — padahal kata "plagiat"
sama sekali tidak ada di dokumen, sehingga pertanyaan itu memang tidak bisa
dijawab. Angka 100% yang dihasilkan mencerminkan kelonggaran metrik, bukan
kualitas sistem.

Versi ini memakai ukuran yang bisa dipertanggungjawabkan:

- **Page Recall@k** — apakah ada chunk terambil yang berasal dari halaman
  yang sudah diverifikasi memuat jawabannya. Relevansi tingkat halaman
  bersifat objektif dan tidak bisa "dicurangi" oleh kata umum.
- **MRR** — seberapa tinggi peringkat chunk relevan pertama. Membedakan
  sistem yang menempatkan jawaban di peringkat 1 dari yang menaruhnya di
  peringkat 4.
- **Cakupan kata kunci** — SELURUH kata kunci wajib muncul, bukan salah satu.
- **Akurasi penolakan** — untuk pertanyaan di luar cakupan.
- **Penolakan salah** — pertanyaan yang layak dijawab tetapi ikut ditolak.
  Tanpa metrik ini, sistem bisa memoles angka penolakan dengan cara menolak
  segalanya.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cli_utils import enable_utf8_stdout
from app.config import ABLATION_PRESETS, EVAL_DIR, RetrievalConfig
from app.retrieval import HybridRetriever, RetrievalOutcome

TEST_QUESTIONS_PATH = EVAL_DIR / "test_questions.json"
RESULTS_PATH = EVAL_DIR / "eval_results.json"

#: Urutan preset saat ditampilkan, dari paling sederhana ke paling lengkap.
PRESET_ORDER = ["dense", "hybrid", "hybrid_rerank"]

PRESET_LABELS = {
    "dense": "Dense only (baseline)",
    "hybrid": "Hybrid (BM25 + dense, RRF)",
    "hybrid_rerank": "Hybrid + reranker LLM",
}


# ──────────────────────────────────────────────
# Struktur hasil
# ──────────────────────────────────────────────

@dataclass
class QuestionResult:
    """Hasil satu pertanyaan pada satu konfigurasi."""

    id: int
    question: str
    category: str
    is_conversational: bool = False

    expected_pages: list[int] = field(default_factory=list)
    retrieved_pages: list[int] = field(default_factory=list)
    page_hit: bool = False
    first_relevant_rank: int | None = None

    expected_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    keyword_hit: bool = False

    refused: bool = False
    refusal_reason: str = ""
    effective_query: str = ""
    was_rewritten: bool = False
    rerank_failed: bool = False
    gate: dict = field(default_factory=dict)

    top_dense_score: float = 0.0
    latency_ms: float = 0.0
    timings_ms: dict = field(default_factory=dict)

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_relevant_rank if self.first_relevant_rank else 0.0

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        data["reciprocal_rank"] = round(self.reciprocal_rank, 4)
        return data


# ──────────────────────────────────────────────
# Pemuatan & penilaian
# ──────────────────────────────────────────────

def load_test_questions(path: Path = TEST_QUESTIONS_PATH) -> list[dict]:
    """Muat pertanyaan uji dari JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Berkas pertanyaan uji tidak ditemukan: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_question(question: dict, outcome: RetrievalOutcome, latency_ms: float) -> QuestionResult:
    """Nilai satu hasil retrieval terhadap kunci jawaban."""
    expected_pages = question.get("expected_pages") or []
    expected_keywords = question.get("expected_answer_contains") or []

    retrieved_pages: list[int] = []
    first_relevant_rank: int | None = None

    for rank, candidate in enumerate(outcome.contexts, start=1):
        retrieved_pages.extend(candidate.page_numbers)
        if first_relevant_rank is None and expected_pages:
            if set(candidate.page_numbers) & set(expected_pages):
                first_relevant_rank = rank

    combined_text = " ".join(c.text.lower() for c in outcome.contexts)
    missing = [kw for kw in expected_keywords if kw.lower() not in combined_text]

    return QuestionResult(
        id=question["id"],
        question=question["question"],
        category=question.get("category", "in_scope"),
        is_conversational=bool(question.get("chat_history")),
        expected_pages=expected_pages,
        retrieved_pages=sorted(set(retrieved_pages)),
        page_hit=first_relevant_rank is not None,
        first_relevant_rank=first_relevant_rank,
        expected_keywords=expected_keywords,
        missing_keywords=missing,
        # Seluruh kata kunci wajib ada, bukan salah satu.
        keyword_hit=bool(expected_keywords) and not missing,
        refused=outcome.refused,
        refusal_reason=outcome.refusal_reason,
        effective_query=outcome.effective_query,
        was_rewritten=outcome.was_rewritten,
        rerank_failed=outcome.rerank_failed,
        gate={
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in outcome.gate.items()
        },
        top_dense_score=round(
            max((c.dense_score for c in outcome.candidates), default=0.0), 4
        ),
        latency_ms=round(latency_ms, 1),
        timings_ms={k: round(v, 1) for k, v in outcome.timings_ms.items()},
    )


def aggregate(results: list[QuestionResult]) -> dict:
    """Hitung metrik ringkas dari hasil per pertanyaan."""
    in_scope = [r for r in results if r.category == "in_scope"]
    out_of_scope = [r for r in results if r.category == "out_of_scope"]
    conversational = [r for r in in_scope if r.is_conversational]

    scorable = [r for r in in_scope if r.expected_pages]

    page_hits = sum(1 for r in scorable if r.page_hit)
    keyword_scorable = [r for r in in_scope if r.expected_keywords]
    keyword_hits = sum(1 for r in keyword_scorable if r.keyword_hit)
    false_refusals = sum(1 for r in in_scope if r.refused)
    correct_refusals = sum(1 for r in out_of_scope if r.refused)
    conversational_hits = sum(1 for r in conversational if r.page_hit)

    latencies = [r.latency_ms for r in results]
    rerank_failures = sum(1 for r in results if r.rerank_failed)

    def ratio(hits: int, total: int) -> float:
        return hits / total if total else 0.0

    return {
        # Angka ini wajib dibaca bersama akurasi penolakan: setiap kegagalan
        # reranker berarti satu pertanyaan yang gate presisinya tidak berjalan,
        # sehingga akurasi penolakan pada run itu tidak mencerminkan desainnya.
        "rerank_failures": rerank_failures,
        "page_recall": ratio(page_hits, len(scorable)),
        "page_recall_hits": page_hits,
        "page_recall_total": len(scorable),
        "mrr": (
            statistics.mean(r.reciprocal_rank for r in scorable) if scorable else 0.0
        ),
        "keyword_coverage": ratio(keyword_hits, len(keyword_scorable)),
        "keyword_hits": keyword_hits,
        "keyword_total": len(keyword_scorable),
        "refusal_accuracy": ratio(correct_refusals, len(out_of_scope)),
        "correct_refusals": correct_refusals,
        "out_of_scope_total": len(out_of_scope),
        "false_refusal_rate": ratio(false_refusals, len(in_scope)),
        "false_refusals": false_refusals,
        "in_scope_total": len(in_scope),
        "conversational_recall": ratio(conversational_hits, len(conversational)),
        "conversational_hits": conversational_hits,
        "conversational_total": len(conversational),
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
    }


# ──────────────────────────────────────────────
# Eksekusi
# ──────────────────────────────────────────────

def evaluate_preset(
    retriever: HybridRetriever,
    questions: list[dict],
    config: RetrievalConfig,
    preset_name: str,
    api_key: str | None = None,
    verbose: bool = True,
    delay: float = 0.0,
) -> tuple[list[QuestionResult], dict]:
    """
    Jalankan seluruh pertanyaan uji pada satu konfigurasi retrieval.

    Returns:
        Tuple ``(hasil per pertanyaan, metrik agregat)``.
    """
    label = PRESET_LABELS.get(preset_name, preset_name)
    print(f"\n{'=' * 74}")
    print(f"⚙️  KONFIGURASI: {label}")
    print(f"    hybrid={config.use_hybrid}  rerank={config.use_rerank}  "
          f"rewrite={config.use_query_rewrite}  top_k={config.top_k}")
    print(f"{'=' * 74}")

    # Kosongkan cache embedding agar tiap konfigurasi diukur dari kondisi yang
    # sama. Tanpa ini, konfigurasi yang dijalankan lebih dulu menanggung semua
    # biaya panggilan API dan konfigurasi berikutnya tampak jauh lebih cepat
    # hanya karena embedding-nya sudah tersimpan — perbandingan latensinya
    # menjadi menyesatkan.
    from app.llm_client import clear_query_embedding_cache

    clear_query_embedding_cache()

    results: list[QuestionResult] = []

    for position, question in enumerate(questions):
        # Beri jeda antar pertanyaan bila diminta. Menjalankan tiga
        # konfigurasi secara berurutan berarti puluhan panggilan API dalam
        # hitungan detik, dan pada kuota gratis reranker-lah yang pertama
        # terkena rate limit — tepat di tahap yang menjadi gate penolakan.
        if delay and position:
            time.sleep(delay)

        outcome, latency_ms = _retrieve_with_retry(
            retriever, question, config, api_key
        )
        if outcome is None:
            continue

        result = score_question(question, outcome, latency_ms)
        results.append(result)

        if verbose:
            _print_question_result(result)

    metrics = aggregate(results)
    _print_preset_summary(metrics)
    return results, metrics


#: Gangguan sesaat dari sisi layanan, bukan cerminan kualitas retrieval.
_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500")


def _retrieve_with_retry(
    retriever: HybridRetriever,
    question: dict,
    config: RetrievalConfig,
    api_key: str | None,
    max_attempts: int = 3,
) -> tuple[RetrievalOutcome | None, float]:
    """
    Jalankan retrieval dengan percobaan ulang untuk galat sesaat.

    Error 503/429 dari sisi layanan tidak berkaitan dengan kualitas
    retrieval. Bila pertanyaan yang terkena galat itu dilewati, jumlah
    pertanyaan antar konfigurasi menjadi berbeda dan tabel perbandingannya
    tidak lagi setara.

    Returns:
        Tuple ``(outcome, latency_ms)``. ``outcome`` bernilai ``None`` bila
        seluruh percobaan gagal.
    """
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            outcome = retriever.retrieve(
                question["question"],
                config=config,
                chat_history=question.get("chat_history"),
                api_key=api_key,
            )
            return outcome, (time.perf_counter() - started) * 1000
        except Exception as e:
            message = str(e)
            transient = any(marker in message for marker in _TRANSIENT_MARKERS)
            if not transient or attempt == max_attempts:
                print(f"  ❌ Q{question['id']} gagal: {message[:150]}")
                return None, 0.0
            wait = 2.0 * attempt
            print(f"  ⏳ Q{question['id']} galat sesaat, mencoba lagi dalam "
                  f"{wait:.0f}s (percobaan {attempt}/{max_attempts})")
            time.sleep(wait)

    return None, 0.0


def _print_question_result(result: QuestionResult) -> None:
    """Cetak satu baris hasil pertanyaan beserta alasannya."""
    if result.category == "out_of_scope":
        icon = "✅" if result.refused else "❌"
        verdict = "ditolak" if result.refused else "TIDAK ditolak"
    else:
        icon = "✅" if result.page_hit else "❌"
        verdict = "halaman relevan ditemukan" if result.page_hit else "halaman relevan TIDAK ditemukan"
        if result.refused:
            icon, verdict = "❌", "ditolak padahal seharusnya dijawab"

    tag = " [percakapan]" if result.is_conversational else ""
    print(f"\n  {icon} Q{result.id}{tag}: {result.question}")
    print(f"       {verdict}")

    if result.was_rewritten:
        print(f"       ditulis ulang → {result.effective_query!r}")

    if result.category == "in_scope":
        print(f"       halaman diharapkan {result.expected_pages} · "
              f"terambil {result.retrieved_pages}")
        if result.first_relevant_rank:
            print(f"       peringkat relevan pertama: {result.first_relevant_rank} "
                  f"(RR={result.reciprocal_rank:.2f})")
        if result.missing_keywords:
            print(f"       kata kunci tidak ditemukan: {result.missing_keywords}")

    if result.refused and result.refusal_reason:
        print(f"       alasan: {result.refusal_reason}")

    print(f"       cosine tertinggi {result.top_dense_score:.3f} · "
          f"latensi {result.latency_ms:.0f} ms")


def _print_preset_summary(metrics: dict) -> None:
    print(f"\n  {'─' * 70}")
    print(f"  📊 Page Recall@k        : {metrics['page_recall']:.1%} "
          f"({metrics['page_recall_hits']}/{metrics['page_recall_total']})")
    print(f"  📈 MRR                  : {metrics['mrr']:.3f}")
    print(f"  🔤 Cakupan kata kunci   : {metrics['keyword_coverage']:.1%} "
          f"({metrics['keyword_hits']}/{metrics['keyword_total']}) — semua kata kunci wajib ada")
    print(f"  🚫 Akurasi penolakan    : {metrics['refusal_accuracy']:.1%} "
          f"({metrics['correct_refusals']}/{metrics['out_of_scope_total']})")
    print(f"  ⚠️  Penolakan salah      : {metrics['false_refusal_rate']:.1%} "
          f"({metrics['false_refusals']}/{metrics['in_scope_total']})")
    if metrics["rerank_failures"]:
        print(f"  🔥 Reranker GAGAL       : {metrics['rerank_failures']} pertanyaan — "
              f"angka penolakan di atas TIDAK sahih untuk run ini")
    if metrics["conversational_total"]:
        print(f"  💬 Recall percakapan    : {metrics['conversational_recall']:.1%} "
              f"({metrics['conversational_hits']}/{metrics['conversational_total']})")
    print(f"  ⏱️  Latensi              : rata-rata {metrics['avg_latency_ms']:.0f} ms · "
          f"median {metrics['median_latency_ms']:.0f} ms")
    print(f"  {'─' * 70}")


def print_comparison(all_metrics: dict[str, dict]) -> None:
    """Cetak tabel perbandingan antar konfigurasi."""
    if len(all_metrics) < 2:
        return

    names = [name for name in PRESET_ORDER if name in all_metrics]
    names += [name for name in all_metrics if name not in names]

    rows = [
        ("Page Recall@k", "page_recall", "pct"),
        ("MRR", "mrr", "num"),
        ("Cakupan kata kunci", "keyword_coverage", "pct"),
        ("Akurasi penolakan", "refusal_accuracy", "pct"),
        ("Penolakan salah", "false_refusal_rate", "pct"),
        ("Recall percakapan", "conversational_recall", "pct"),
        ("Latensi rata-rata", "avg_latency_ms", "ms"),
    ]

    print(f"\n{'=' * 74}")
    print("📋 PERBANDINGAN KONFIGURASI")
    print(f"{'=' * 74}")

    header = f"  {'Metrik':<22}" + "".join(f"{name:>17}" for name in names)
    print(header)
    print(f"  {'-' * (22 + 17 * len(names))}")

    for label, key, kind in rows:
        cells = []
        for name in names:
            value = all_metrics[name].get(key, 0.0)
            if kind == "pct":
                cells.append(f"{value:>16.1%}")
            elif kind == "ms":
                cells.append(f"{value:>14.0f} ms")
            else:
                cells.append(f"{value:>16.3f}")
        print(f"  {label:<22}" + "".join(cells))

    print(f"\n  Catatan: 'Penolakan salah' dan 'Latensi' makin kecil makin baik.")


def run_full_evaluation(
    modes: list[str],
    api_key: str | None = None,
    skip_conversational: bool = False,
    verbose: bool = True,
    delay: float = 0.0,
) -> dict:
    """Jalankan evaluasi untuk seluruh konfigurasi yang diminta."""
    print("=" * 74)
    print("🚀 EVALUASI RETRIEVAL — RAG Chatbot Kampus UAJY")
    print("=" * 74)

    questions = load_test_questions()
    if skip_conversational:
        questions = [q for q in questions if not q.get("chat_history")]

    retriever = HybridRetriever()
    info = retriever.document_info

    in_scope = [q for q in questions if q.get("category") == "in_scope"]
    out_of_scope = [q for q in questions if q.get("category") == "out_of_scope"]
    conversational = [q for q in questions if q.get("chat_history")]

    print(f"\n📚 Index    : {info['total_chunks']} chunk · {info['total_pages']} halaman · "
          f"maks {info['max_pages_per_chunk']} halaman/chunk "
          f"(rata-rata {info['avg_pages_per_chunk']})")
    print(f"   Embedding: task type {retriever.embedding_task_type or 'generik (index lama)'}")
    print(f"📝 Pertanyaan: {len(questions)} total · {len(in_scope)} in-scope · "
          f"{len(out_of_scope)} out-of-scope · {len(conversational)} percakapan lanjutan")

    all_results: dict[str, list[dict]] = {}
    all_metrics: dict[str, dict] = {}

    for mode in modes:
        config = ABLATION_PRESETS[mode]
        # Pertanyaan percakapan butuh penulisan ulang agar bermakna; tanpa itu
        # yang terukur bukan lagi kemampuan pipeline melainkan keberuntungan.
        if conversational and not config.use_query_rewrite:
            config = config.with_overrides(use_query_rewrite=True)

        results, metrics = evaluate_preset(
            retriever, questions, config, mode,
            api_key=api_key, verbose=verbose, delay=delay,
        )
        all_results[mode] = [r.to_dict() for r in results]
        all_metrics[mode] = metrics

    print_comparison(all_metrics)

    payload = {
        "index": {
            "total_chunks": info["total_chunks"],
            "total_pages": info["total_pages"],
            "max_pages_per_chunk": info["max_pages_per_chunk"],
            "avg_pages_per_chunk": info["avg_pages_per_chunk"],
            "embedding_task_type": retriever.embedding_task_type,
        },
        "questions": {
            "total": len(questions),
            "in_scope": len(in_scope),
            "out_of_scope": len(out_of_scope),
            "conversational": len(conversational),
        },
        "metrics": all_metrics,
        "results": all_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Hasil lengkap disimpan ke: {RESULTS_PATH}")

    return payload


def main() -> None:
    enable_utf8_stdout()

    parser = argparse.ArgumentParser(description="Evaluasi retrieval RAG Chatbot Kampus")
    parser.add_argument(
        "--mode",
        action="append",
        choices=list(ABLATION_PRESETS),
        help="Konfigurasi yang diuji. Bisa diulang. Default: semuanya.",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Google Gemini API key")
    parser.add_argument(
        "--skip-conversational",
        action="store_true",
        help="Lewati pertanyaan lanjutan berbasis riwayat chat",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hanya cetak ringkasan, tanpa rincian per pertanyaan",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        metavar="DETIK",
        help="Jeda antar pertanyaan. Pada kuota gratis, gunakan 2 agar "
             "reranker tidak terkena rate limit (yang membuat angka "
             "penolakan tidak sahih).",
    )
    args = parser.parse_args()

    modes = args.mode or PRESET_ORDER

    from app.llm_client import LLMConfigError, resolve_api_key

    try:
        resolve_api_key(args.api_key)
    except LLMConfigError as e:
        print(f"❌ {e}")
        sys.exit(1)

    try:
        run_full_evaluation(
            modes=modes,
            api_key=args.api_key,
            skip_conversational=args.skip_conversational,
            verbose=not args.quiet,
            delay=args.delay,
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
