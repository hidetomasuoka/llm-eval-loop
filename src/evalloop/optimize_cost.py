"""Pre-run cost estimation for `evalloop optimize` (APO-10).

Extracted from evalloop.optimize (kept there as re-exports for backward
compatibility).
"""

from __future__ import annotations

from dataclasses import dataclass

from evalloop.build import ESTIMATED_OUTPUT_TOKENS
from evalloop.dspy_lm import _reflection_registry_model
from evalloop.schemas import Config, GoldenCase
from evalloop.token_counting import average_input_tokens, render_case_prompts

# ---------------------------------------------------------------------------
# pre-run cost estimate (APO-10): build.py already warns before an eval run;
# optimize multiplies that by the optimizer's iteration budget, which is where
# surprise costs come from (OPRO's own docs warn about this). Everything here
# is a deliberate order-of-magnitude guess -- the goal is a digit-count
# warning before the first rollout is spent, not accounting.
# ---------------------------------------------------------------------------

# "How many optimizer rounds" per method. Actual counts depend on dspy
# internals and early stopping; one round is modeled as evaluating one
# candidate instruction over the full train split plus one reflection call.
#   gepa / miprov2: the candidate budget scales with optimize.auto
#   copro: breadth candidates per depth round (see _rollout_factor)
_AUTO_ROLLOUT_FACTORS = {"light": 10, "medium": 25, "heavy": 60}

# A reflection/proposal call carries the current instructions plus a batch of
# failing examples with feedback (much larger than a single task rollout) and
# returns a rewritten instruction.
REFLECTION_INPUT_TOKENS_ESTIMATE = 3000
REFLECTION_OUTPUT_TOKENS_ESTIMATE = 500


@dataclass
class OptimizeCostEstimate:
    method: str
    train_case_count: int
    rollout_factor: int  # optimizer rounds (candidates evaluated)
    rollout_count: int  # target-model calls: rollout_factor x train cases
    reflection_call_count: int  # instruction proposals by the reflection model
    target_input_tokens: int
    target_token_count_method: str
    target_usd: float
    reflection_usd: float | None  # None -- reflection provider absent from the price registry
    total_usd: float


def _rollout_factor(cfg: Config) -> int:
    if cfg.optimize.method == "copro":
        breadth = int(cfg.optimize.params.get("breadth", 10))
        depth = int(cfg.optimize.params.get("depth", 3))
        return breadth * depth
    if cfg.optimize.method == "tapo":
        population_size = int(cfg.optimize.params.get("population_size", 4))
        generations = int(cfg.optimize.params.get("generations", 3))
        return max(1, population_size * generations)
    return _AUTO_ROLLOUT_FACTORS.get(cfg.optimize.auto, _AUTO_ROLLOUT_FACTORS["medium"])


def estimate_optimize_cost(cfg: Config, train_cases: list[GoldenCase], prompt_template: str) -> OptimizeCostEstimate:
    """Rough optimize cost from the config.yaml price table: target-model
    rollouts (train size x method factor) plus reflection calls. Target-model
    input counting shares the provider-aware implementation used by build.py;
    reflection prompts remain a documented order-of-magnitude assumption.
    """
    factor = _rollout_factor(cfg)
    rollout_count = factor * len(train_cases)
    reflection_call_count = factor  # ~one instruction proposal per optimizer round

    target = cfg.model_by_alias(cfg.optimize.target_alias)
    rendered_prompts = render_case_prompts(prompt_template, [c.input for c in train_cases])
    token_count = average_input_tokens(target.provider, rendered_prompts)
    in_tokens = token_count.average_input_tokens
    out_tokens = ESTIMATED_OUTPUT_TOKENS.get(cfg.task.answer_type, 100)
    target_usd = rollout_count * (
        in_tokens / 1_000_000 * target.price_in_per_mtok + out_tokens / 1_000_000 * target.price_out_per_mtok
    )

    reflection_model = _reflection_registry_model(cfg)
    reflection_usd = None
    if reflection_model is not None:
        reflection_usd = reflection_call_count * (
            REFLECTION_INPUT_TOKENS_ESTIMATE / 1_000_000 * reflection_model.price_in_per_mtok
            + REFLECTION_OUTPUT_TOKENS_ESTIMATE / 1_000_000 * reflection_model.price_out_per_mtok
        )

    return OptimizeCostEstimate(
        method=cfg.optimize.method,
        train_case_count=len(train_cases),
        rollout_factor=factor,
        rollout_count=rollout_count,
        reflection_call_count=reflection_call_count,
        target_input_tokens=in_tokens,
        target_token_count_method=token_count.method,
        target_usd=target_usd,
        reflection_usd=reflection_usd,
        total_usd=target_usd + (reflection_usd or 0.0),
    )
