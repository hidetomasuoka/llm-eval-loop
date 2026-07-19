"""promptfoo provider <-> dspy/litellm bridging: model-string translation,
sampling-param quirks, and post-hoc LM history cost summaries.

Extracted from evalloop.optimize (kept there as re-exports for backward
compatibility, following the APO-03 metrics precedent).
"""

from __future__ import annotations

from dataclasses import dataclass

from evalloop.optimizers.base import OptimizeError
from evalloop.schemas import Config, ModelConfig

# ---------------------------------------------------------------------------
# promptfoo provider id -> dspy/litellm model string
# ---------------------------------------------------------------------------


def promptfoo_provider_to_dspy_lm(provider: str) -> str:
    if provider.startswith("anthropic:messages:"):
        return "anthropic/" + provider.split(":", 2)[2]
    if provider.startswith("ollama:chat:"):
        return "ollama_chat/" + provider.split(":", 2)[2]
    # TODO: add a case here (and verify against https://dspy.ai/ provider docs)
    # before using any provider prefix other than the two above in config.yaml.
    raise OptimizeError(
        f"don't know how to translate promptfoo provider {provider!r} into a dspy LM string "
        "(only anthropic:messages: and ollama:chat: are mapped so far) -- add a case to "
        "promptfoo_provider_to_dspy_lm() in optimize.py"
    )


def _dspy_temperature(supports_sampling_params: bool, temperature: float) -> float | None:
    """claude-opus-4-8 / claude-fable-5 reject sampling params with HTTP 400 on
    the dspy/litellm path too, not just through promptfoo. litellm drops
    None-valued params from the request (verified against the installed
    litellm: get_optional_params(temperature=None) omits the key), so None is
    how we avoid sending temperature to those models.
    """
    return temperature if supports_sampling_params else None


def _reflection_registry_model(cfg: Config) -> ModelConfig | None:
    """optimize.reflection_provider is a dspy/litellm string; map it back to
    its config.yaml model registry entry when one exists (for sampling-param
    and price lookups). Returns None when nothing in the registry matches.
    """
    for m in cfg.models:
        try:
            if promptfoo_provider_to_dspy_lm(m.provider) == cfg.optimize.reflection_provider:
                return m
        except OptimizeError:
            continue  # registry entries with unmapped provider prefixes can't match
    return None


def _reflection_supports_sampling(cfg: Config) -> bool:
    """A registry model marked supports_sampling_params=false must not receive
    temperature on the dspy path either (the bundled configs point reflection
    at anthropic/claude-opus-4-8, which 400s on it). Providers with no
    registry match default to True (send temperature, the historical behavior).
    """
    model = _reflection_registry_model(cfg)
    return model.supports_sampling_params if model is not None else True


@dataclass
class SearchCostSummary:
    """Post-hoc exploration cost from dspy LM histories (APO-14 / issue #73)."""

    search_cost_usd: float | None
    search_lm_call_count: int


def _history_entries(lm) -> list:
    history = getattr(lm, "history", None)
    if not history:
        return []
    return list(history)


def _tokens_from_usage(usage: object) -> tuple[int, int] | None:
    if not isinstance(usage, dict):
        return None
    in_raw = usage.get("prompt_tokens", usage.get("input_tokens"))
    out_raw = usage.get("completion_tokens", usage.get("output_tokens"))
    if in_raw is None and out_raw is None:
        return None
    try:
        in_tok = int(in_raw or 0)
        out_tok = int(out_raw or 0)
    except (TypeError, ValueError):
        return None
    return in_tok, out_tok


def _cost_from_history_entry(entry: object, model: ModelConfig | None) -> float | None:
    """Prefer LiteLLM ``cost``; else token usage × registry prices."""
    if not isinstance(entry, dict):
        return None
    raw_cost = entry.get("cost")
    if raw_cost is not None:
        try:
            return float(raw_cost)
        except (TypeError, ValueError):
            pass
    tokens = _tokens_from_usage(entry.get("usage"))
    if tokens is None or model is None:
        return None
    in_tok, out_tok = tokens
    return in_tok / 1_000_000 * model.price_in_per_mtok + out_tok / 1_000_000 * model.price_out_per_mtok


def summarize_lm_search_cost(task_lm, reflection_lm, cfg: Config) -> SearchCostSummary:
    """Sum dspy ``lm.history`` costs for target + reflection LMs after optimize.

    Returns ``search_cost_usd=None`` when history is empty or any call cannot be
    priced (compare / logs then show ``n/a``).
    """
    target = cfg.model_by_alias(cfg.optimize.target_alias)
    reflection = _reflection_registry_model(cfg)
    pairs = ((task_lm, target), (reflection_lm, reflection))
    total = 0.0
    call_count = 0
    priced_any = False
    for lm, model in pairs:
        for entry in _history_entries(lm):
            call_count += 1
            cost = _cost_from_history_entry(entry, model)
            if cost is None:
                return SearchCostSummary(search_cost_usd=None, search_lm_call_count=call_count)
            total += cost
            priced_any = True
    if call_count == 0 or not priced_any:
        return SearchCostSummary(search_cost_usd=None, search_lm_call_count=call_count)
    return SearchCostSummary(search_cost_usd=round(total, 6), search_lm_call_count=call_count)
