"""
Cost calculation for cost-per-query observability — Section 12

Turns token counts into a USD cost using the per-model pricing table in
settings. This is what powers "cost per query" dashboards: every LLM call
reports prompt/completion tokens, and we convert that to dollars here.
"""

from app.config.settings import MODEL_PRICING_PER_1M


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Return the USD cost of a single LLM call.

    Args:
        model             : model identifier (e.g. "gpt-4o-mini")
        prompt_tokens     : input tokens consumed
        completion_tokens : output tokens generated

    Returns:
        Cost in USD (float, rounded to 6 dp). Returns 0.0 for unknown models
        rather than raising — a missing price should never break a request.
    """
    pricing = MODEL_PRICING_PER_1M.get(model)
    if not pricing:
        return 0.0

    cost = (
        prompt_tokens     / 1_000_000 * pricing["input"]
        + completion_tokens / 1_000_000 * pricing["output"]
    )
    return round(cost, 6)
