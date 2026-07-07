"""Deterministic proxy metrics and prompt-template round-trip helpers, shared
by every prompt-optimization method in this package.

Scope note: GEPA needs a fast in-process metric -- it cannot shell out to
promptfoo per candidate rollout, and the iron rule "Python never calls a model
provider directly" rules out an in-process LLM judge. Training therefore uses
a deterministic proxy metric per answer_type:

    label -- port of asserts/label_match.js (identical verdict to the final
             promptfoo grading; pinned by tests/fixtures/label_normalization_cases.json)
    text  -- SQuAD-style token F1 against the gold span(s). The FINAL
             evaluation stays promptfoo's llm-rubric: training proxy and
             final judge are deliberately different things, and their
             divergence is a measurement target of the GEPA case study,
             not something this module hides
    json  -- port of asserts/json_field_match.js deep-equality (pinned by
             tests/fixtures/json_field_match_cases.json)
"""

from __future__ import annotations

import json
import re
from collections import Counter

from evalloop.optimizers.base import OptimizeError

# ---------------------------------------------------------------------------
# prompt template <-> dspy instructions round-trip
# ---------------------------------------------------------------------------

_INPUT_MARKER = "{{input}}"


def _split_template(template: str) -> tuple[str, str]:
    """Split a Jinja-style prompts/base/task.txt into (instructions, trailer),
    where `trailer` is the last blank-line-separated paragraph containing the
    {{input}} placeholder (verbatim), and `instructions` is everything before
    it. GEPA is only allowed to rewrite `instructions`; `trailer` (the actual
    variable substitution promptfoo needs) is preserved as-is.
    """
    paragraphs = template.split("\n\n")
    for i, para in enumerate(paragraphs):
        if _INPUT_MARKER in para:
            instructions = "\n\n".join(paragraphs[:i]).strip()
            trailer = "\n\n".join(paragraphs[i:]).strip()
            return instructions, trailer
    return template.strip(), _INPUT_MARKER


def extract_instructions_from_template(template: str) -> str:
    instructions, _trailer = _split_template(template)
    return instructions


def render_optimized_template(instructions: str, original_template: str) -> str:
    _orig_instructions, trailer = _split_template(original_template)
    return f"{instructions.strip()}\n\n{trailer}\n"


# ---------------------------------------------------------------------------
# metric: Python port of asserts/label_match.js (GEPA needs an in-process,
# fast metric -- it cannot shell out to promptfoo per candidate rollout)
# ---------------------------------------------------------------------------


def _normalize_label(value) -> str:
    # Must stay in lockstep with normalizeLabel() in asserts/label_match.js,
    # or GEPA trains against a different verdict than promptfoo's final
    # grading. tests/test_label_normalization.py pins both implementations to
    # the same fixture table -- extend that fixture when changing either side.
    if not isinstance(value, str):
        return ""
    s = value.strip()
    s = "".join(chr(ord(ch) - 0xFEE0) if "！" <= ch <= "～" else ch for ch in s)
    s = re.sub(r"^[\"'「『\[]+", "", s)
    s = re.sub(r"[\"'」』\]]+$", "", s)
    s = re.sub(r"[。.、,]+$", "", s)
    return s.strip()


def label_score_and_feedback(output: str, expected: str, labels: list[str]) -> tuple[float, str]:
    norm_output = _normalize_label(output or "")
    norm_expected = _normalize_label(expected)

    if norm_output == norm_expected:
        return 1.0, f'output "{output}" correctly matches expected label "{expected}".'

    norm_labels = [_normalize_label(label) for label in labels]
    contained = sorted({label for label in norm_labels if label and label in norm_output})

    if len(contained) == 1 and contained[0] == norm_expected:
        return 1.0, f'output "{output}" contains exactly the expected label "{expected}".'
    if contained:
        return 0.0, (
            f'output "{output}" reads as label "{contained[0]}" but the expected label was "{expected}". '
            f"Rewrite the instructions so the model outputs only the single correct label from {labels}."
        )
    return 0.0, (
        f'output "{output}" does not contain any of the known labels {labels}; expected "{expected}". '
        "Rewrite the instructions to make the model output exactly one label from the list, with no extra text."
    )


# ---------------------------------------------------------------------------
# metric: answer_type=text (extractive tasks, e.g. CUAD clause extraction).
# Deterministic recall-weighted token score -- see the module docstring for why
# the training metric is a proxy and not the llm-rubric judge.
#
# The first GEPA run (20260706-075752) maximized plain token F1, which pushed
# the optimizer toward "extract the shortest possible span" (precision rises
# when you drop words). The final llm-rubric judge, however, rewards covering
# the full clause and tolerates mild over-extraction. We therefore weight
# recall 0.8 / precision 0.2 so that MISSING gold tokens hurt much more than
# EXTRA output tokens -- steering GEPA away from the over-truncation failure
# mode observed in the first run.
# ---------------------------------------------------------------------------

# prompts/base/task.txt instructs the model to answer exactly this string when
# no clause of the requested category exists in the excerpt
NO_CLAUSE_ANSWER = "該当条項なし"

_EN_ARTICLES = {"a", "an", "the"}

# recall / precision weights for the span-set score. 0.8 recall was chosen so
# that dropping half the gold span (recall 0.5) yields 0.5*0.8 + 1.0*0.2 = 0.6
# (well below 1.0) while doubling the output length (precision 0.5, recall 1.0)
# still scores 1.0*0.8 + 0.5*0.2 = 0.9 -- close enough to 1.0 that GEPA prefers
# over-extraction (which the rubric tolerates) over under-extraction (which it
# fails). See EXPERIMENT.md for the first-run divergence that motivated this.
RECALL_WEIGHT = 0.8
PRECISION_WEIGHT = 0.2


def _f1_tokens(text: str) -> list[str]:
    """SQuAD-style normalization: lowercase, strip punctuation, drop English
    articles, whitespace-tokenize. Gold spans in CUAD are English contract
    text, so whitespace tokenization is adequate; a Japanese-answer task
    would need a real tokenizer here.
    """
    s = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in s.split() if t not in _EN_ARTICLES]


def _token_overlap(pred_tokens: list[str], gold_tokens: list[str]) -> int:
    """Number of overlapping tokens (with multiplicity) between pred and gold."""
    return sum((Counter(pred_tokens) & Counter(gold_tokens)).values())


def _token_f1(pred: str, gold: str) -> float:
    """Plain token F1 between two single spans (kept for the per-span greedy
    alignment in _span_set_score -- the SPAN PAIRING still uses F1 so that a
    gold span matches its best output span; only the AGGREGATE across spans
    switches to recall-weighting below).
    """
    pred_tokens = _f1_tokens(pred)
    gold_tokens = _f1_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    overlap = _token_overlap(pred_tokens, gold_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _split_spans(text: str) -> list[str]:
    """The task prompt instructs semicolon-separated listing of multiple hits.
    Clause text itself can legitimately contain semicolons (it does in the
    real CUAD data), but the model must quote verbatim, so both sides fragment
    at the same places and the greedy alignment below still pairs fragments up.
    """
    spans = [s.strip() for s in re.split(r"[;；]", text) if s.strip()]
    return spans or [""]


def _span_set_score(output: str, expected: str) -> float:
    """Recall-weighted score in [0, 1] between output spans and gold spans.

    Per-span PAIRING uses plain F1 (so each gold span finds its best-matching
    output span), but the overall AGGREGATE is 0.8*recall + 0.2*precision --
    not F1. Recall is computed over the union of gold tokens matched by the
    greedy alignment (so missing a gold span hurts), and precision over the
    union of output tokens consumed (so extra spans hurt, but mildly).

    Reduces to a single-span recall-weighted score when both sides are one
    span. Capped at 1.0 in case both weights saturate.
    """
    out_spans = _split_spans(output)
    exp_spans = _split_spans(expected)
    remaining = list(out_spans)
    matched_total = 0.0
    for exp_span in exp_spans:
        if not remaining:
            break
        scores = [_token_f1(out_span, exp_span) for out_span in remaining]
        best_i = max(range(len(scores)), key=lambda i: scores[i])
        matched_total += scores[best_i]
        remaining.pop(best_i)
    # matched_total is at most len(exp_spans); span_count_penalty is the
    # fraction of gold spans that found a partner. recall and precision both
    # already live in [0, 1] via the per-span F1s, so we just combine them.
    recall = matched_total / len(exp_spans) if exp_spans else 1.0
    # precision: matched_total over the larger of (out_spans, exp_spans) -- a
    # crude proxy for "how much of what the model said was on-target". Extra
    # spans (len(out) > len(exp)) lower this.
    span_count_penalty = matched_total / max(len(exp_spans), len(out_spans)) if (exp_spans or out_spans) else 1.0
    score = RECALL_WEIGHT * recall + PRECISION_WEIGHT * span_count_penalty
    return min(1.0, score)


def text_score_and_feedback(output, expected) -> tuple[float, str]:
    """Continuous score in [0, 1] (GEPA accepts float scores; the gradient of
    a partial-overlap F1 gives the optimizer more signal than thresholding
    to 0/1 would).
    """
    out_text = output if isinstance(output, str) else ""
    exp_text = expected if isinstance(expected, str) else ""
    # _normalize_label strips wrapping quotes/trailing punctuation, so
    # 「該当条項なし。」 still counts as the no-clause answer
    out_is_no_clause = _normalize_label(out_text) == NO_CLAUSE_ANSWER
    exp_is_no_clause = _normalize_label(exp_text) == NO_CLAUSE_ANSWER

    if exp_is_no_clause and out_is_no_clause:
        return 1.0, f'output correctly answered "{NO_CLAUSE_ANSWER}" (gold agrees no clause applies).'
    if exp_is_no_clause:
        return 0.0, (
            f'output extracted text but the gold answer is "{NO_CLAUSE_ANSWER}" (no applicable clause). '
            f'output was: "{out_text[:160]}". Rewrite the instructions so the model answers exactly '
            f'"{NO_CLAUSE_ANSWER}" when the excerpt contains no clause of the requested category.'
        )
    if out_is_no_clause:
        return 0.0, (
            f'output answered "{NO_CLAUSE_ANSWER}" but the gold answer contains a clause: "{exp_text[:160]}". '
            "Rewrite the instructions so the model searches the excerpt more thoroughly before "
            "concluding that no clause applies."
        )

    score = _span_set_score(out_text, exp_text)
    if score >= 1.0:
        return 1.0, "output covers all gold span(s) at recall-weighted score 1.00."
    return score, (
        f"output covers the gold span(s) at recall-weighted score {score:.2f} "
        f"(recall is weighted 0.8, precision 0.2 -- missing gold text hurts more than extra text). "
        f'gold: "{exp_text[:160]}" / output: "{out_text[:160]}". '
        "Rewrite the instructions so the model extracts the COMPLETE clause text -- "
        "do not truncate, do not reduce to a single heading word, quote the full span verbatim."
    )


# ---------------------------------------------------------------------------
# metric: answer_type=json (Python port of asserts/json_field_match.js).
# Must stay in lockstep with deepEqual() there --
# tests/fixtures/json_field_match_cases.json pins both implementations.
# ---------------------------------------------------------------------------


def _json_deep_equal(a, b) -> bool:
    # Python's == alone would treat True == 1 as equal, but JS's === keeps
    # booleans and numbers distinct -- check bools explicitly. Cross int/float
    # comparison stays allowed (JS has a single number type: 1 === 1.0).
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def json_score_and_feedback(output, expected) -> tuple[float, str]:
    try:
        parsed = json.loads(output) if isinstance(output, str) else output
    except json.JSONDecodeError as e:
        return 0.0, (
            f'output is not valid JSON ({e}). output was: "{str(output)[:160]}". '
            "Rewrite the instructions so the model emits exactly one JSON object and nothing else "
            "(no code fences, no explanations)."
        )
    if _json_deep_equal(parsed, expected):
        return 1.0, "parsed JSON deep-equals the expected object."
    got = json.dumps(parsed, ensure_ascii=False)
    want = json.dumps(expected, ensure_ascii=False)
    return 0.0, (
        f"parsed JSON does not match expected. got={got[:200]} expected={want[:200]}. "
        "Rewrite the instructions to pin down the exact field names and value formats."
    )


def _score_fn_for(cfg):
    """Return the (output, expected) -> (score, feedback) training metric for
    the task's answer_type. This is the GEPA training proxy, NOT the final
    evaluation -- promptfoo still grades text tasks with llm-rubric (see the
    module docstring).
    """
    if cfg.task.answer_type == "label":
        labels = cfg.task.labels
        return lambda output, expected: label_score_and_feedback(output, expected, labels)
    if cfg.task.answer_type == "text":
        return text_score_and_feedback
    if cfg.task.answer_type == "json":
        return json_score_and_feedback
    # unreachable while TaskConfig validates answer_type, but fail loudly if
    # a new type is added there without a metric here
    raise OptimizeError(f"no GEPA training metric for answer_type {cfg.task.answer_type!r}")
