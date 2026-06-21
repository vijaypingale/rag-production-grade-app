"""
RAG Evaluation Runner — Section 10 (with routing harness)

Run with:
    venv/Scripts/python.exe scripts/eval.py                 # full set
    venv/Scripts/python.exe scripts/eval.py --max 2         # 2 per category (quick smoke)
    venv/Scripts/python.exe scripts/eval.py --category adversarial

What it does:
-------------
1. Loads the stratified gold dataset (questions tagged with category + eval_type)
2. Runs the LIVE RAG pipeline (ask()) for each question
3. ROUTES each question by eval_type:
     - "ragas"      -> collected and scored by RAGAS (faithfulness, relevancy,
                       context precision/recall) against the reference answer
     - "behavioral" -> deterministic check (abstain / resist / no-hallucination)
4. Prints a combined report: RAGAS metric scores + behavioral pass rates,
   broken down by category, and applies CI-gate thresholds.

Cost / time note:
-----------------
Every question runs the full pipeline (≥2 LLM calls each), and RAGAS adds
several judge calls per answerable question. The full 70-question set takes
several minutes and real tokens. Use --max for a quick smoke test.

Exit code:
----------
Non-zero if any gated RAGAS metric is below threshold OR any behavioral
category has failures — so this doubles as a CI quality gate.
"""

import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ask_service import ask
from app.evaluation.ragas_eval import run_ragas_evaluation, EvalSample
from app.evaluation.behavioral import run_behavioral_check


# RAGAS CI-gate thresholds. context_recall is report-only (corpus-dependent).
THRESHOLDS = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.85,
    "context_precision": 0.80,
}


def load_samples(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["samples"]


def maybe_limit(samples: list[dict], max_per_cat: int | None, only_cat: str | None) -> list[dict]:
    """Optionally filter by category and cap N per category (for quick runs)."""
    if only_cat:
        samples = [s for s in samples if s["category"] == only_cat]
    if max_per_cat is None:
        return samples
    seen: dict = defaultdict(int)
    out = []
    for s in samples:
        c = s["category"]
        if seen[c] < max_per_cat:
            out.append(s)
            seen[c] += 1
    return out


def run(samples: list[dict]):
    """
    Execute the pipeline for each sample and route results.
    Returns (ragas_samples, behavioral_results).
    """
    ragas_samples: list[EvalSample] = []
    behavioral_results: list[dict] = []

    for i, s in enumerate(samples, start=1):
        q = s["question"]
        print(f"  [{i}/{len(samples)}] ({s['category']}) {q[:55]}...")

        result = ask(query=q)

        if s["eval_type"] == "ragas":
            ragas_samples.append(EvalSample(
                user_input=q,
                response=result.answer,
                retrieved_contexts=result.retrieved_contexts,
                reference=s["reference"],
            ))
        else:  # behavioral
            passed, detail = run_behavioral_check(s, result)
            behavioral_results.append({
                "category": s["category"],
                "question": q,
                "passed":   passed,
                "detail":   detail,
            })

    return ragas_samples, behavioral_results


def print_report(ragas_eval: dict | None, behavioral_results: list[dict]) -> bool:
    all_passed = True

    print("\n" + "=" * 64)
    print("  EVALUATION REPORT")
    print("=" * 64)

    # --- RAGAS section ---
    if ragas_eval:
        scores = ragas_eval["scores"]
        print(f"\n  RAGAS metrics ({ragas_eval['n_samples']} answerable questions)")
        print(f"  {'Metric':<22}{'Score':<10}{'Target':<10}{'Status'}")
        print("  " + "-" * 52)
        for metric, score in scores.items():
            target = THRESHOLDS.get(metric)
            if target is not None:
                passed = score >= target
                all_passed = all_passed and passed
                print(f"  {metric:<22}{score:<10.3f}{target:<10.2f}{'PASS' if passed else 'FAIL'}")
            else:
                print(f"  {metric:<22}{score:<10.3f}{'(report)':<10}-")

    # --- Behavioral section ---
    if behavioral_results:
        by_cat: dict = defaultdict(lambda: {"pass": 0, "total": 0})
        for r in behavioral_results:
            by_cat[r["category"]]["total"] += 1
            if r["passed"]:
                by_cat[r["category"]]["pass"] += 1

        print(f"\n  Behavioral checks ({len(behavioral_results)} questions)")
        print(f"  {'Category':<22}{'Passed':<12}{'Status'}")
        print("  " + "-" * 52)
        for cat, c in by_cat.items():
            cat_passed = c["pass"] == c["total"]
            all_passed = all_passed and cat_passed
            ratio = f"{c['pass']}/{c['total']}"
            print(f"  {cat:<22}{ratio:<12}{'PASS' if cat_passed else 'FAIL'}")

        # Show individual failures for debugging
        failures = [r for r in behavioral_results if not r["passed"]]
        if failures:
            print("\n  Behavioral failures:")
            for f in failures:
                print(f"    - [{f['category']}] {f['question'][:50]} -> {f['detail']}")

    print("\n  " + "-" * 52)
    print(f"  Overall gate: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 64 + "\n")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Run stratified RAG evaluation.")
    parser.add_argument("--dataset", default="data/eval/wiser_eval_set.json")
    parser.add_argument("--max", type=int, default=None,
                        help="Max questions PER CATEGORY (quick smoke test).")
    parser.add_argument("--category", default=None,
                        help="Only run this category (e.g. adversarial).")
    args = parser.parse_args()

    print(f"\nLoading dataset: {args.dataset}")
    samples = load_samples(args.dataset)
    samples = maybe_limit(samples, args.max, args.category)
    print(f"Running {len(samples)} question(s).\n")

    ragas_samples, behavioral_results = run(samples)

    ragas_eval = None
    if ragas_samples:
        print(f"\nScoring {len(ragas_samples)} answerable question(s) with RAGAS...")
        ragas_eval = run_ragas_evaluation(ragas_samples)

    all_passed = print_report(ragas_eval, behavioral_results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
