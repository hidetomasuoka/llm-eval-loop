"""Provider-aware input token counting for pre-run cost estimates.

The cost warning must work in a fresh, offline clone, so token counting is a
best-effort ladder rather than a hard dependency on a provider API:

1. Anthropic's free ``/v1/messages/count_tokens`` endpoint when an API key is
   available (set ``EVALLOOP_TOKEN_COUNT_API=off`` to disable network access).
2. tiktoken for OpenAI model IDs it recognizes locally.
3. An explicit mixed Japanese/English heuristic for every other provider.

Every result carries the method name.  Callers surface it next to the dollar
estimate so a fallback can never masquerade as a model-accurate count.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import tiktoken

ANTHROPIC_COUNT_TOKENS_URL = "https://api.anthropic.com/v1/messages/count_tokens"
ANTHROPIC_API_VERSION = "2023-06-01"
MAX_API_SAMPLE_CASES = 20
_API_TIMEOUT_SECONDS = 3
_API_MAX_WORKERS = 8
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True)
class TokenCount:
    average_input_tokens: int
    method: str
    sampled_case_count: int


def render_case_prompts(prompt_template: str, inputs: list[str]) -> list[str]:
    """Render the only promptfoo variable owned by the golden dataset."""
    return [prompt_template.replace("{{input}}", input_text) for input_text in inputs]


def _sample_evenly(texts: list[str], limit: int = MAX_API_SAMPLE_CASES) -> list[str]:
    if len(texts) <= limit:
        return texts
    if limit <= 1:
        return [texts[0]]
    indexes = [round(i * (len(texts) - 1) / (limit - 1)) for i in range(limit)]
    return [texts[i] for i in indexes]


def _anthropic_model(provider: str) -> str | None:
    prefix = "anthropic:messages:"
    return provider.removeprefix(prefix) if provider.startswith(prefix) else None


def _openai_model(provider: str) -> str | None:
    for prefix in ("openai:chat:", "openai:"):
        if provider.startswith(prefix):
            return provider.removeprefix(prefix)
    return None


def _anthropic_api_enabled() -> bool:
    value = os.environ.get("EVALLOOP_TOKEN_COUNT_API", "auto").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _count_one_anthropic_prompt(model: str, api_key: str, text: str) -> int | None:
    """Count tokens for a single rendered prompt (one eval case = one request)."""
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": text}]}).encode("utf-8")
    request = urllib.request.Request(
        ANTHROPIC_COUNT_TOKENS_URL,
        data=payload,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_API_TIMEOUT_SECONDS) as response:
            raw = json.loads(response.read().decode("utf-8"))
        total = int(raw["input_tokens"])
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError):
        return None
    return total if total > 0 else None


def _count_with_anthropic_api(model: str, texts: list[str]) -> TokenCount | None:
    """Average per-case counts so the estimate matches build/optimize call shape.

    Each sampled case is counted as its own ``user`` message — the same unit
    promptfoo sends — rather than concatenating cases into one blob.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not _anthropic_api_enabled():
        return None

    sampled = _sample_evenly(texts)
    counts: list[int] = []
    workers = min(_API_MAX_WORKERS, len(sampled))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_count_one_anthropic_prompt, model, api_key, text) for text in sampled]
            for future in as_completed(futures):
                counted = future.result()
                if counted is None:
                    return None
                counts.append(counted)
    except (OSError, ValueError, TypeError):
        return None
    if not counts:
        return None
    return TokenCount(
        average_input_tokens=max(1, math.ceil(sum(counts) / len(counts))),
        method="anthropic-count-tokens-api",
        sampled_case_count=len(sampled),
    )


def _count_with_tiktoken(model: str, texts: list[str]) -> TokenCount | None:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except (KeyError, ValueError):
        return None
    counts = [len(encoding.encode(text, disallowed_special=())) for text in texts]
    return TokenCount(
        average_input_tokens=max(1, math.ceil(sum(counts) / len(counts))),
        method=f"tiktoken:{encoding.name}",
        sampled_case_count=len(texts),
    )


def heuristic_token_count(text: str) -> int:
    """Offline approximation tuned for mixed Japanese/English business text.

    CJK characters and punctuation usually form dense token boundaries, while
    contiguous ASCII words average roughly four characters per token.  This is
    intentionally named as a heuristic in all user-visible output.
    """
    tokens = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        word = _ASCII_WORD_RE.match(text, index)
        if word:
            tokens += math.ceil(len(word.group(0)) / 4)
            index = word.end()
            continue
        if _CJK_RE.match(char) or not char.isalnum():
            tokens += 1
        else:
            # Non-ASCII alphabetic scripts: avoid silently treating a long
            # sequence as one token when no provider tokenizer is available.
            tokens += 1
        index += 1
    return max(1, tokens)


def average_input_tokens(provider: str, texts: list[str]) -> TokenCount:
    """Return a provider-aware average, always falling back without raising."""
    if not texts:
        return TokenCount(average_input_tokens=1, method="heuristic:mixed-text-v1", sampled_case_count=0)

    anthropic_model = _anthropic_model(provider)
    if anthropic_model:
        counted = _count_with_anthropic_api(anthropic_model, texts)
        if counted is not None:
            return counted

    openai_model = _openai_model(provider)
    if openai_model:
        counted = _count_with_tiktoken(openai_model, texts)
        if counted is not None:
            return counted

    counts = [heuristic_token_count(text) for text in texts]
    return TokenCount(
        average_input_tokens=max(1, math.ceil(sum(counts) / len(counts))),
        method="heuristic:mixed-text-v1",
        sampled_case_count=len(texts),
    )
