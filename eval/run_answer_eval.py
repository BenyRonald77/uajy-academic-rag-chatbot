"""
run_answer_eval.py — Evaluasi mutu jawaban akhir (bukan hanya retrieval).

Usage:
    python eval/run_answer_eval.py                    # seluruh pertanyaan
    python eval/run_answer_eval.py --limit 6          # cicil, hemat kuota
    python eval/run_answer_eval.py --category out_of_scope
    python eval/run_answer_eval.py --no-judge         # tanpa LLM-as-judge

Berbeda dari `run_eval.py` yang berhenti di tingkat retrieval, skrip ini
menjalankan pipeline sampai jawaban akhir lalu mengukur tiga hal:

1. **Groundedness** — apakah pernyataan dalam jawaban didukung konteks.
2. **Akurasi sitasi** — apakah nomor halaman yang ditulis di jawaban benar.
3. **Penolakan end-to-end** — apakah pertanyaan di luar cakupan sungguh
   ditolak pada jawaban akhirnya, bukan hanya pada gate retrieval.

Biayanya lebih besar daripada `run_eval.py`: setiap pertanyaan memerlukan satu
panggilan pembuat jawaban ditambah satu panggilan penilai. Karena itu ada flag
`--limit` dan `--no-judge` untuk mencicil.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cli_utils import enable_utf8_stdout
from app.config import DEFAULT_RETRIEVAL_CONFIG, EVAL_DIR, RetrievalConfig
from app.llm_client import call_llm
from app.prompt_builder import build_no_context_response, build_prompt
from app.retrieval import HybridRetriever
from eval.answer_eval import (
    GROUNDEDNESS_PASS_SCORE,
    AnswerResult,
    aggregate_answer_metrics,
    check_citations,
    detect_refusal,
    judge_groundedness,
)
from eval.run_eval import load_test_questions

RESULTS_PATH = EVAL_DIR / "answer_eval_results.json"
PARTIAL_RESULTS_PATH = EVAL_DIR / "answer_eval_results_partial.json"


def evaluate_question(
    question: dict,
    retriever: HybridRetriever,
    config: RetrievalConfig,
    api_key: str | None = None,
    use_judge: bool = True,
) -> AnswerResult:
    """
    Jalankan pipeline lengkap untuk satu pertanyaan lalu nilai jawabannya.

    Args:
        question: Entri dari `test_questions.json`.
        retriever: Retriever yang sudah dimuat.
        config: Konfigurasi retrieval.
        api_key: Override API key Gemini.
        use_judge: Jalankan penilaian groundedness.

    Returns:
        `AnswerResult` berisi jawaban beserta seluruh hasil penilaiannya.
    """
    started = time.perf_counter()
    expected_pages = question.get("expected_pages") or []

    result = AnswerResult(
        id=question["id"],
        question=question["question"],
        category=question.get("category", "in_scope"),
        expected_pages=expected_pages,
    )

    outcome = retriever.retrieve(
        question["question"],
        config=config,
        chat_history=question.get("chat_history"),
        api_key=api_key,
    )

    if outcome.refused:
        # Gate retrieval menolak, jadi LLM tidak dipanggil sama sekali.
        result.refused_by_retrieval = True
        result.answer = build_no_context_response()
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    system_prompt, user_prompt = build_prompt(
        question["question"], outcome.contexts, question.get("chat_history")
    )

    # `raise_on_error=True` supaya kegagalan API tidak menyelinap masuk
    # sebagai "jawaban". Pesan error tidak memuat klaim faktual, sehingga
    # penilai groundedness akan memberinya nilai sempurna dan seluruh metrik
    # ikut terdistorsi.
    try:
        result.answer = call_llm(
            user_prompt,
            system_instruction=system_prompt,
            api_key=api_key,
            raise_on_error=True,
        )
    except Exception as e:
        result.generation_failed = True
        result.generation_error = str(e)[:200]
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    result.refused_in_answer = detect_refusal(result.answer)

    citation = check_citations(result.answer, outcome.contexts)
    result.cited_pages = citation.cited_pages
    result.context_pages = citation.context_pages
    result.invalid_citations = citation.invalid_pages
    result.has_citation = citation.has_citation
    result.citation_valid = citation.is_valid
    result.citation_precision = citation.precision
    result.cites_expected_page = bool(set(citation.cited_pages) & set(expected_pages))

    if use_judge and not result.refused_in_answer:
        verdict = judge_groundedness(
            question["question"], outcome.contexts, result.answer, api_key=api_key
        )
        result.groundedness_score = verdict.score
        result.unsupported_claims = verdict.unsupported_claims
        result.grounded = verdict.is_grounded
        result.judge_failed = verdict.judge_failed

    result.latency_ms = (time.perf_counter() - started) * 1000
    return result


def print_question_result(result: AnswerResult) -> None:
    """Cetak satu hasil beserta alasan penilaiannya."""
    if result.generation_failed:
        print(f"\n  🔥 Q{result.id}: {result.question}")
        print(f"       gagal menghasilkan jawaban — dikeluarkan dari seluruh metrik")
        print(f"       {result.generation_error}")
        return

    if result.category == "out_of_scope":
        icon = "✅" if result.refused else "❌"
        verdict = "ditolak" if result.refused else "TIDAK ditolak — berpotensi halusinasi"
    elif result.refused:
        icon, verdict = "❌", "ditolak padahal seharusnya dijawab"
    else:
        masalah = []
        if result.judge_failed:
            masalah.append("penilai gagal")
        elif not result.grounded:
            masalah.append(f"groundedness {result.groundedness_score}")
        if not result.citation_valid:
            masalah.append("sitasi tidak valid")
        icon = "✅" if not masalah else "⚠️"
        verdict = "dijawab dengan baik" if not masalah else ", ".join(masalah)

    print(f"\n  {icon} Q{result.id}: {result.question}")
    print(f"       {verdict}")

    if result.category == "in_scope" and not result.refused:
        if result.groundedness_score is not None:
            print(f"       groundedness : {result.groundedness_score:.1f}/10 "
                  f"(lolos ≥ {GROUNDEDNESS_PASS_SCORE:.0f})")
        for claim in result.unsupported_claims[:2]:
            print(f"         ! tidak didukung: {claim[:88]}")

        print(f"       sitasi       : halaman {result.cited_pages or '(tidak ada)'} "
              f"· konteks {result.context_pages}")
        if result.invalid_citations:
            print(f"         ! mengutip halaman di luar konteks: {result.invalid_citations}")
        if result.expected_pages:
            tanda = "ya" if result.cites_expected_page else "tidak"
            print(f"       kutip halaman kunci {result.expected_pages}: {tanda}")

    print(f"       latensi      : {result.latency_ms:.0f} ms")


def print_summary(metrics: dict) -> None:
    print(f"\n{'=' * 74}")
    print("📊 RINGKASAN EVALUASI JAWABAN")
    print(f"{'=' * 74}")
    print(f"  Pertanyaan dijawab       : {metrics['answered_total']}/{metrics['in_scope_total']} in-scope")
    if metrics["generation_failures"]:
        print(f"  🔥 Gagal dihasilkan      : {metrics['generation_failures']} — "
              f"dikeluarkan dari seluruh metrik di bawah")
    print()
    print("  ── Kesetiaan pada dokumen ──")
    print(f"  Groundedness             : {metrics['groundedness_rate']:.1%} "
          f"({metrics['judged_total']} jawaban dinilai)")
    print(f"  Skor groundedness rata²  : {metrics['avg_groundedness_score']:.2f}/10")
    if metrics["judge_failures"]:
        print(f"  🔥 Penilai gagal         : {metrics['judge_failures']} — "
              f"angka groundedness di atas tidak mencakup jawaban ini")
    print()
    print("  ── Sitasi ──")
    print(f"  Menyertakan sitasi       : {metrics['citation_presence']:.1%}")
    print(f"  Sitasi valid             : {metrics['citation_validity']:.1%} "
          f"(semua halaman terkutip berasal dari konteks)")
    print(f"  Presisi sitasi rata²     : {metrics['avg_citation_precision']:.1%}")
    print(f"  Mengutip halaman kunci   : {metrics['cites_expected_page']:.1%}")
    print()
    print("  ── Penolakan end-to-end ──")
    print(f"  Akurasi penolakan        : {metrics['refusal_accuracy_e2e']:.1%} "
          f"({metrics['out_of_scope_total']} pertanyaan di luar cakupan)")
    print(f"  Penolakan salah          : {metrics['false_refusal_e2e']:.1%} ↓")
    print()
    print(f"  Latensi rata-rata        : {metrics['avg_latency_ms']:.0f} ms")
    print(f"{'=' * 74}")


def run(
    limit: int | None = None,
    category: str | None = None,
    api_key: str | None = None,
    use_judge: bool = True,
    verbose: bool = True,
    delay: float = 0.0,
) -> dict:
    """Jalankan evaluasi jawaban untuk pertanyaan yang dipilih."""
    print("=" * 74)
    print("🚀 EVALUASI JAWABAN — RAG Chatbot Kampus UAJY")
    print("=" * 74)

    questions = load_test_questions()
    if category:
        questions = [q for q in questions if q.get("category") == category]
    if limit:
        questions = questions[:limit]

    retriever = HybridRetriever()
    info = retriever.document_info
    config = DEFAULT_RETRIEVAL_CONFIG

    print(f"\n📚 Index      : {info['total_chunks']} chunk · {info['total_pages']} halaman")
    print(f"📝 Pertanyaan  : {len(questions)}")
    print(f"⚖️  Penilai     : {'aktif' if use_judge else 'dimatikan'}")
    print(f"🔀 Konfigurasi : hybrid={config.use_hybrid} rerank={config.use_rerank} "
          f"rewrite={config.use_query_rewrite}")

    results: list[AnswerResult] = []
    for position, question in enumerate(questions):
        # Setiap pertanyaan memakan tiga panggilan API (embedding, penjawab,
        # penilai). Tanpa jeda, kuota gratis kehabisan di tengah jalan dan
        # kegagalannya justru menimpa tahap penilaian.
        if delay and position:
            time.sleep(delay)

        try:
            result = evaluate_question(
                question, retriever, config, api_key=api_key, use_judge=use_judge
            )
        except Exception as e:
            print(f"\n  ❌ Q{question['id']} gagal: {str(e)[:150]}")
            continue

        results.append(result)
        if verbose:
            print_question_result(result)

    metrics = aggregate_answer_metrics(results)
    print_summary(metrics)

    is_partial = bool(limit or category)
    payload = {
        "partial_run": is_partial,
        "filters": {"limit": limit, "category": category},
        "index": {
            "total_chunks": info["total_chunks"],
            "total_pages": info["total_pages"],
        },
        "config": {
            "use_hybrid": config.use_hybrid,
            "use_rerank": config.use_rerank,
            "use_query_rewrite": config.use_query_rewrite,
            "top_k": config.top_k,
            "judge_enabled": use_judge,
            "groundedness_pass_score": GROUNDEDNESS_PASS_SCORE,
        },
        "metrics": metrics,
        "results": [r.to_dict() for r in results],
    }

    # Run parsial ditulis ke berkas terpisah. Menimpa hasil run penuh dengan
    # sebagian pertanyaan membuat artefaknya menyesatkan — persis yang terjadi
    # saat run pengujian singkat menghapus hasil pengukuran lengkap.
    path = PARTIAL_RESULTS_PATH if is_partial else RESULTS_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    label = "Hasil parsial" if is_partial else "Hasil lengkap"
    print(f"\n💾 {label} disimpan ke: {path}")
    if is_partial:
        print(f"   Berkas hasil run penuh ({RESULTS_PATH.name}) tidak diubah.")

    return payload


def main() -> None:
    enable_utf8_stdout()

    parser = argparse.ArgumentParser(
        description="Evaluasi mutu jawaban RAG Chatbot Kampus",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Google Gemini API key")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Batasi jumlah pertanyaan (hemat kuota)",
    )
    parser.add_argument(
        "--category", choices=["in_scope", "out_of_scope"], default=None,
        help="Hanya uji kategori tertentu",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Lewati penilaian groundedness (hanya sitasi & penolakan)",
    )
    parser.add_argument("--quiet", action="store_true", help="Hanya cetak ringkasan")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        metavar="DETIK",
        help="Jeda antar pertanyaan. Pada kuota gratis gunakan 3, sebab tiap "
             "pertanyaan memakan tiga panggilan API.",
    )
    args = parser.parse_args()

    from app.llm_client import LLMConfigError, resolve_api_key

    try:
        resolve_api_key(args.api_key)
    except LLMConfigError as e:
        print(f"❌ {e}")
        sys.exit(1)

    try:
        run(
            limit=args.limit,
            category=args.category,
            api_key=args.api_key,
            use_judge=not args.no_judge,
            verbose=not args.quiet,
            delay=args.delay,
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
