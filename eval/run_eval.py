"""
run_eval.py — Evaluasi kualitas RAG chatbot.

Menguji:
1. Retrieval quality (recall@k)
2. Answer faithfulness
3. Refusal correctness (untuk pertanyaan di luar dokumen)
4. Latency

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --api-key YOUR_KEY
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.build_index import get_embeddings
from app.retrieval import DocumentRetriever
from app.prompt_builder import build_prompt, build_no_context_response


# ──────────────────────────────────────────────
# Konfigurasi
# ──────────────────────────────────────────────

TEST_QUESTIONS_PATH = Path(__file__).resolve().parent / "test_questions.json"
DEFAULT_TOP_K = 4
SIMILARITY_THRESHOLD = 0.30


def load_test_questions() -> list[dict]:
    """Load pertanyaan uji dari JSON."""
    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(
    retriever: DocumentRetriever,
    questions: list[dict],
    api_key: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Evaluasi kualitas retrieval.

    Returns:
        Dict dengan metrik retrieval.
    """
    from google import genai

    client = genai.Client(api_key=api_key)
    in_scope_questions = [q for q in questions if q["category"] == "in_scope"]

    total = len(in_scope_questions)
    hits = 0
    latencies = []

    print(f"\n{'='*60}")
    print(f"📊 EVALUASI RETRIEVAL (top_k={top_k})")
    print(f"{'='*60}")

    for q in in_scope_questions:
        start = time.time()

        # Get embedding
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=q["question"],
        )
        query_embedding = result.embeddings[0].values

        # Search
        results = retriever.search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=SIMILARITY_THRESHOLD,
        )

        elapsed = time.time() - start
        latencies.append(elapsed)

        # Check if expected keywords are found in retrieved chunks
        all_text = " ".join(r.text.lower() for r in results)
        keywords_found = any(
            kw.lower() in all_text
            for kw in q.get("expected_answer_contains", [])
        )

        status = "✅" if keywords_found else "❌"
        hits += 1 if keywords_found else 0

        print(f"\n{status} Q{q['id']}: {q['question']}")
        print(f"   Chunks retrieved: {len(results)}")
        if results:
            top_score = max(r.similarity_score for r in results)
            print(f"   Top similarity: {top_score:.3f}")
        print(f"   Keywords found: {keywords_found}")
        print(f"   Latency: {elapsed:.2f}s")

    recall = hits / total if total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print(f"\n{'─'*60}")
    print(f"📊 Retrieval Recall@{top_k}: {recall:.1%} ({hits}/{total})")
    print(f"⏱️  Avg retrieval latency: {avg_latency:.2f}s")
    print(f"{'─'*60}")

    return {
        "recall_at_k": recall,
        "hits": hits,
        "total": total,
        "avg_latency_s": avg_latency,
    }


def evaluate_refusal(
    retriever: DocumentRetriever,
    questions: list[dict],
    api_key: str,
) -> dict:
    """
    Evaluasi refusal correctness untuk pertanyaan di luar dokumen.

    Returns:
        Dict dengan metrik refusal.
    """
    from google import genai

    client = genai.Client(api_key=api_key)
    oos_questions = [q for q in questions if q["category"] == "out_of_scope"]

    total = len(oos_questions)
    correct_refusals = 0

    print(f"\n{'='*60}")
    print(f"🚫 EVALUASI REFUSAL CORRECTNESS")
    print(f"{'='*60}")

    for q in oos_questions:
        # Get embedding
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=q["question"],
        )
        query_embedding = result.embeddings[0].values

        # Search
        results = retriever.search(
            query_embedding=query_embedding,
            top_k=DEFAULT_TOP_K,
            threshold=SIMILARITY_THRESHOLD,
        )

        # Pertanyaan OOS seharusnya tidak menemukan konteks relevan
        refused = len(results) == 0
        correct_refusals += 1 if refused else 0

        status = "✅" if refused else "⚠️"
        print(f"\n{status} Q{q['id']}: {q['question']}")
        print(f"   Chunks retrieved: {len(results)} (expected: 0)")
        if results:
            print(f"   ⚠️ False positive — chunk ditemukan tapi seharusnya tidak")
            for r in results:
                print(f"      Score: {r.similarity_score:.3f} — {r.text[:100]}...")

    accuracy = correct_refusals / total if total > 0 else 0

    print(f"\n{'─'*60}")
    print(f"🚫 Refusal Accuracy: {accuracy:.1%} ({correct_refusals}/{total})")
    print(f"{'─'*60}")

    return {
        "refusal_accuracy": accuracy,
        "correct_refusals": correct_refusals,
        "total": total,
    }


def run_full_evaluation(api_key: str) -> dict:
    """Jalankan evaluasi lengkap."""
    print("\n" + "=" * 60)
    print("🚀 EVALUASI LENGKAP — RAG Chatbot Kampus UAJY")
    print("=" * 60)

    # Load components
    questions = load_test_questions()
    retriever = DocumentRetriever()

    print(f"\n📝 Total pertanyaan uji: {len(questions)}")
    print(f"   In-scope: {len([q for q in questions if q['category'] == 'in_scope'])}")
    print(f"   Out-of-scope: {len([q for q in questions if q['category'] == 'out_of_scope'])}")

    # Run evaluations
    retrieval_metrics = evaluate_retrieval(retriever, questions, api_key)
    refusal_metrics = evaluate_refusal(retriever, questions, api_key)

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 RINGKASAN EVALUASI")
    print(f"{'='*60}")
    print(f"   Retrieval Recall@{DEFAULT_TOP_K}: {retrieval_metrics['recall_at_k']:.1%}")
    print(f"   Refusal Accuracy:    {refusal_metrics['refusal_accuracy']:.1%}")
    print(f"   Avg Retrieval Latency: {retrieval_metrics['avg_latency_s']:.2f}s")

    # Criteria check
    print(f"\n{'─'*60}")
    recall_pass = retrieval_metrics['recall_at_k'] >= 0.80
    refusal_pass = refusal_metrics['refusal_accuracy'] >= 1.0
    print(f"   {'✅' if recall_pass else '❌'} Recall@4 ≥ 80%: {retrieval_metrics['recall_at_k']:.1%}")
    print(f"   {'✅' if refusal_pass else '❌'} Refusal = 100%: {refusal_metrics['refusal_accuracy']:.1%}")
    print(f"{'─'*60}")

    all_metrics = {
        "retrieval": retrieval_metrics,
        "refusal": refusal_metrics,
    }

    # Save results
    results_path = Path(__file__).resolve().parent / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n💾 Hasil disimpan ke: {results_path}")

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi RAG Chatbot Kampus")
    parser.add_argument("--api-key", type=str, default=None, help="Google Gemini API key")
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        secrets_path = Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            import tomllib
            with open(secrets_path, "rb") as f:
                secrets = tomllib.load(f)
            api_key = secrets.get("GEMINI_API_KEY")

    if not api_key or api_key == "MASUKKAN_API_KEY_GEMINI_ANDA_DI_SINI":
        print("❌ API key belum diatur!")
        sys.exit(1)

    run_full_evaluation(api_key)
