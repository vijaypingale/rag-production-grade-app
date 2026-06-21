"""
RAG Evaluation Runner — Section 10

Run with:
    venv/Scripts/python.exe scripts/eval.py
    venv/Scripts/python.exe scripts/eval.py --dataset data/eval/wiser_eval_set.json

What it does:
-------------
1. Loads the gold eval dataset (questions + reference answers)
2. Runs the LIVE RAG pipeline (ask()) for each question to collect the
   generated answer + the chunks it retrieved
3. Hands those to RAGAS, which scores faithfulness / answer_relevancy /
   context_precision / context_recall using an LLM-as-judge
4. Prints an aggregate report and checks each metric against a target
   threshold (the same idea as a CI quality gate)

Cost / time note:
-----------------
This makes many LLM calls (the pipeline answer + faithfulness check per
question, PLUS several RAGAS judge calls per metric per question). For a
7-question set expect a few minutes and a few cents of tokens. That's why
eval is an offline batch job, never on the live request path.

Exit code:
----------
Returns non-zero if any metric falls below its target threshold — so this
script can double as a CI gate (fail the build on quality regression).
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure project root is importable when run as `python scripts/eval.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ask_service import ask
from app.evaluation.ragas_eval import run_ragas_evaluation, EvalSample
from app.utils.logger import logger


# Target thresholds — the "CI gate". Industry guidance:
#   faithfulness >= 0.85, answer_relevancy >= 0.85, context_precision >= 0.8
# context_recall is reported but not gated by default (corpus-dependent).
THRESHOLDS = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.85,
    "context_precision": 0.80,
}


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"]


def collect_samples(gold: list[dict]) -> list[EvalSample]:
    """
    Run the live pipeline for each gold question and assemble EvalSamples.
    """
    samples: list[EvalSample] = []

    for i, item in enumerate(gold, start=1):
        question  = item["question"]
        reference = item["reference"]

        print(f"  [{i}/{len(gold)}] Running pipeline: {question[:60]}...")

        result = ask(query=question)

        samples.append(EvalSample(
            user_input=question,
            response=result.answer,
            retrieved_contexts=result.retrieved_contexts,
            reference=reference,
        ))

    return samples


def print_report(eval_result: dict) -> bool:
    """
    Print the aggregate + per-sample report. Returns True if all gated
    metrics passed their thresholds, False otherwise.
    """
    scores = eval_result["scores"]

    print("\n" + "=" * 60)
    print(f"  RAGAS EVALUATION REPORT  ({eval_result['n_samples']} questions)")
    print("=" * 60)

    all_passed = True
    print(f"\n  {'Metric':<22}{'Score':<10}{'Target':<10}{'Status'}")
    print("  " + "-" * 50)
    for metric, score in scores.items():
        target = THRESHOLDS.get(metric)
        if target is not None:
            passed = score >= target
            all_passed = all_passed and passed
            status = "PASS" if passed else "FAIL"
            print(f"  {metric:<22}{score:<10.3f}{target:<10.2f}{status}")
        else:
            print(f"  {metric:<22}{score:<10.3f}{'(report)':<10}-")

    print("\n  " + "-" * 50)
    print(f"  Overall gate: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 60 + "\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG pipeline.")
    parser.add_argument(
        "--dataset",
        default="data/eval/wiser_eval_set.json",
        help="Path to the gold eval dataset JSON.",
    )
    args = parser.parse_args()

    print(f"\nLoading eval dataset: {args.dataset}")
    gold = load_dataset(args.dataset)
    print(f"Loaded {len(gold)} gold question(s).\n")

    print("Collecting live pipeline answers...")
    samples = collect_samples(gold)

    print("\nRunning RAGAS evaluation (this calls the LLM many times)...")
    eval_result = run_ragas_evaluation(samples)

    all_passed = print_report(eval_result)

    # Non-zero exit on failure so this can act as a CI gate.
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
