"""
RAGAS Evaluation — Section 10

What this is:
-------------
OFFLINE, batch quality measurement of the RAG pipeline — the "test suite for
the AI", distinct from the per-request runtime guardrails in Section 9.

The four metrics (the de-facto standard, popularized by RAGAS):
---------------------------------------------------------------
- faithfulness        : Is the answer supported by the retrieved context?
                        (catches hallucination — claims not in the context)
- answer_relevancy    : Does the answer actually address the question?
                        (catches off-topic / evasive answers)
- context_precision   : Of the chunks retrieved, how many were relevant?
                        (signal quality of retrieval — noise vs. signal)
- context_recall      : Did retrieval fetch ALL the info needed to answer?
                        (catches the "list all codes" coverage gap — the
                         dimension that faithfulness is blind to)

How RAGAS computes them:
------------------------
RAGAS is NOT magic — under the hood every metric is an LLM-as-judge call
(and embeddings for answer_relevancy). We hand RAGAS our own LLM + embedding
clients (wrapped via LangchainLLMWrapper / LangchainEmbeddingsWrapper) so the
eval uses the SAME models the app uses. RAGAS just orchestrates the judging
and aggregation.

Why this needs a "reference" (ground truth):
--------------------------------------------
context_recall and context_precision (with reference) compare against a
gold answer. That's why data/eval/wiser_eval_set.json carries a `reference`
for every question — exactly like expected values in a unit test, except the
comparison is semantic (LLM-judged), not exact string equality.
"""

import os
from dataclasses import dataclass

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config.settings import LLM_MODEL, EMBEDDING_MODEL
from app.utils.logger import logger


# The four metrics we report. Order here = column order in the report.
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


@dataclass
class EvalSample:
    """
    One row going into RAGAS. Mirrors RAGAS's expected schema:
        user_input         : the question
        response           : the pipeline's generated answer
        retrieved_contexts : the chunk texts the pipeline retrieved
        reference          : the gold/ground-truth answer
    """
    user_input:         str
    response:           str
    retrieved_contexts: list[str]
    reference:          str


def _get_judge_llm() -> LangchainLLMWrapper:
    """
    The LLM RAGAS uses to JUDGE (decompose claims, score relevance).
    We reuse the app's configured model so eval reflects production.
    temperature=0 for repeatable judging.
    """
    return LangchainLLMWrapper(
        ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )


def _get_judge_embeddings() -> LangchainEmbeddingsWrapper:
    """Embeddings RAGAS uses for answer_relevancy similarity scoring."""
    return LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )


def run_ragas_evaluation(samples: list[EvalSample]) -> dict:
    """
    Run RAGAS over the collected samples and return aggregate scores.

    Args:
        samples : list of EvalSample (one per eval question), already
                  populated with the live pipeline's answer + contexts.

    Returns:
        dict with:
            - "scores"  : { metric_name: float }  aggregate (mean) per metric
            - "per_sample" : list of per-question metric dicts
            - "n_samples"  : how many questions were evaluated

    Note:
        This makes MANY LLM calls (several per sample per metric), so it is
        slow and costs tokens. That's expected — it's an offline batch job,
        not a live request path.
    """
    if not samples:
        raise ValueError("No samples provided to RAGAS evaluation.")

    # Build the RAGAS dataset from our samples.
    dataset = EvaluationDataset.from_list([
        {
            "user_input":         s.user_input,
            "response":           s.response,
            "retrieved_contexts": s.retrieved_contexts,
            "reference":          s.reference,
        }
        for s in samples
    ])

    logger.info("ragas_evaluation_started", n_samples=len(samples), metrics=[m.name for m in METRICS])

    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=_get_judge_llm(),
        embeddings=_get_judge_embeddings(),
    )

    # result is a RAGAS EvaluationResult; convert to a pandas DataFrame
    # to extract aggregate + per-sample scores cleanly.
    df = result.to_pandas()

    metric_names = [m.name for m in METRICS]
    aggregate = {
        name: float(df[name].mean())
        for name in metric_names
        if name in df.columns
    }

    per_sample = df[
        [c for c in (["user_input"] + metric_names) if c in df.columns]
    ].to_dict(orient="records")

    logger.info("ragas_evaluation_completed", aggregate=aggregate)

    return {
        "scores":     aggregate,
        "per_sample": per_sample,
        "n_samples":  len(samples),
    }
