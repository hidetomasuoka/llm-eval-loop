"""dspy GEPA optimization: golden.jsonl split=='train' -> optimized prompt ->
promptfoo variant config -> automatic run/report/compare.

Confirmed against the installed dspy==3.2.1 API (dspy.ai docs + `inspect.signature`):
    from dspy.teleprompt import GEPA
    GEPA(metric, *, auto=None, reflection_lm=None, seed=0, ...)
    GEPA.compile(student, *, trainset, teacher=None, valset=None)
    metric(gold, pred, trace, pred_name, pred_trace) -> dspy.Prediction(score=, feedback=)

Iron rules enforced here:
    1. split separation: this module reads ONLY split=='train' cases, and
       re-asserts (independently of build.py) that the train IDs it is about
       to train on are disjoint from data/build/tests_test.yaml's case IDs
       before spending a single GEPA rollout.

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
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import dspy
import yaml

from evalloop import build as build_mod
from evalloop import report as report_mod
from evalloop import run as run_mod
from evalloop.schemas import assert_split_disjoint, load_config, load_golden_jsonl, parse_promptfoo_output

REPO_ROOT = build_mod.REPO_ROOT
OPTIMIZED_DIR = REPO_ROOT / "prompts" / "optimized"
VARIANTS_DIR = REPO_ROOT / "promptfoo" / "variants"


class OptimizeError(RuntimeError):
    pass


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
# Deterministic SQuAD-style token F1 -- see the module docstring for why the
# training metric is a proxy and not the llm-rubric judge.
# ---------------------------------------------------------------------------

# prompts/base/task.txt instructs the model to answer exactly this string when
# no clause of the requested category exists in the excerpt
NO_CLAUSE_ANSWER = "該当条項なし"

_EN_ARTICLES = {"a", "an", "the"}


def _f1_tokens(text: str) -> list[str]:
    """SQuAD-style normalization: lowercase, strip punctuation, drop English
    articles, whitespace-tokenize. Gold spans in CUAD are English contract
    text, so whitespace tokenization is adequate; a Japanese-answer task
    would need a real tokenizer here.
    """
    s = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in s.split() if t not in _EN_ARTICLES]


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _f1_tokens(pred)
    gold_tokens = _f1_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
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


def _span_set_f1(output: str, expected: str) -> float:
    """Greedy max-F1 alignment between output spans and gold spans; the
    denominator is the larger span count, so missing and spurious spans both
    cost score. Reduces to plain token F1 when both sides are a single span.
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
    return matched_total / max(len(exp_spans), len(out_spans))


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

    f1 = _span_set_f1(out_text, exp_text)
    if f1 >= 1.0:
        return 1.0, "output token-matches the gold span(s) exactly (token F1 1.00)."
    return f1, (
        f"output overlaps the gold span(s) at token F1 {f1:.2f}. "
        f'gold: "{exp_text[:160]}" / output: "{out_text[:160]}". '
        "Rewrite the instructions so the model quotes the exact clause text verbatim -- "
        "no paraphrasing, no commentary, no partial extraction."
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


# ---------------------------------------------------------------------------
# variant config generation (reroots every file:// reference one level
# deeper, since promptfoo/variants/{name}.yaml lives one directory below
# promptfoo/promptfooconfig.yaml)
# ---------------------------------------------------------------------------


def _reroot_file_refs(obj, prefix: str):
    if isinstance(obj, dict):
        return {k: _reroot_file_refs(v, prefix) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_reroot_file_refs(v, prefix) for v in obj]
    if isinstance(obj, str) and obj.startswith("file://"):
        return "file://" + prefix + obj[len("file://") :]
    return obj


def to_variant_relpath(target: Path) -> str:
    rel = os.path.relpath(target, start=VARIANTS_DIR)
    return rel.replace(os.sep, "/")


def build_variant_config(target_alias: str, task_path: Path) -> dict:
    if not build_mod.PROMPTFOO_CONFIG_PATH.exists():
        raise OptimizeError(f"{build_mod.PROMPTFOO_CONFIG_PATH} not found; run `evalloop build` first")
    base_config = yaml.safe_load(build_mod.PROMPTFOO_CONFIG_PATH.read_text(encoding="utf-8"))
    variant_config = _reroot_file_refs(base_config, prefix="../")
    variant_config["prompts"] = [f"file://{to_variant_relpath(task_path)}"]
    variant_config["description"] = f"{base_config.get('description', '')} (optimized: {target_alias})"
    return variant_config


# ---------------------------------------------------------------------------
# GEPA orchestration
# ---------------------------------------------------------------------------


@dataclass
class OptimizeOutcome:
    variant_name: str
    task_path: Path
    variant_path: Path
    run_id: str
    base_run_id: str | None
    compare_path: Path | None


def _load_test_ids() -> set[str]:
    if not build_mod.TESTS_TEST_PATH.exists():
        raise OptimizeError(f"{build_mod.TESTS_TEST_PATH} not found; run `evalloop build` first")
    entries = yaml.safe_load(build_mod.TESTS_TEST_PATH.read_text(encoding="utf-8")) or []
    return {e["vars"]["case_id"] for e in entries}


def _find_latest_base_run(task_name: str) -> str | None:
    if not run_mod.INDEX_PATH.exists():
        return None
    candidates = []
    with run_mod.INDEX_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if (
                entry.get("task_name") == task_name
                and not entry.get("variant")
                and entry.get("promptfoo_exit_code") == 0
            ):
                candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(key=lambda e: e["created_at"])
    return candidates[-1]["run_id"]


def run_gepa(student, trainset, metric, reflection_lm, auto: str, seed: int = 0):
    """Thin, monkeypatchable wrapper around the real dspy.teleprompt.GEPA call
    so orchestration logic (file writing, variant config, run/report/compare)
    can be unit-tested without spending real API calls.
    """
    from dspy.teleprompt import GEPA

    optimizer = GEPA(metric=metric, reflection_lm=reflection_lm, auto=auto, seed=seed)
    return optimizer.compile(student=student, trainset=trainset)


def optimize(config_path: str | Path = REPO_ROOT / "config.yaml") -> OptimizeOutcome:
    cfg = load_config(config_path)
    score_fn = _score_fn_for(cfg)  # resolve the training metric first: fail fast on unsupported types

    test_ids = _load_test_ids()
    cases = load_golden_jsonl(build_mod.GOLDEN_PATH)
    train_cases = [c for c in cases if c.split == "train"]
    if not train_cases:
        raise OptimizeError("golden.jsonl has no split=='train' cases; nothing to optimize against")
    train_ids = {c.id for c in train_cases}
    assert_split_disjoint(train_ids, test_ids)  # iron rule #1, re-checked independently of build.py

    target_model = cfg.model_by_alias(cfg.optimize.target_alias)
    task_lm = dspy.LM(
        promptfoo_provider_to_dspy_lm(target_model.provider),
        temperature=cfg.run.temperature,
        max_tokens=cfg.run.max_tokens,
    )
    reflection_lm = dspy.LM(cfg.optimize.reflection_provider, temperature=1.0, max_tokens=32000)
    dspy.configure(lm=task_lm)

    original_template = (REPO_ROOT / cfg.task.prompt_file).read_text(encoding="utf-8")
    base_instructions = extract_instructions_from_template(original_template)
    signature = dspy.Signature("input -> output", instructions=base_instructions)
    student = dspy.Predict(signature)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        score, feedback = score_fn(getattr(pred, "output", ""), gold.expected)
        return dspy.Prediction(score=score, feedback=feedback)

    trainset = [dspy.Example(input=c.input, expected=c.expected).with_inputs("input") for c in train_cases]

    optimized_program = run_gepa(student, trainset, metric, reflection_lm, cfg.optimize.auto)
    optimized_instructions = optimized_program.signature.instructions
    optimized_template = render_optimized_template(optimized_instructions, original_template)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = OPTIMIZED_DIR / cfg.optimize.target_alias / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    task_path = out_dir / "task.txt"
    task_path.write_text(optimized_template, encoding="utf-8")
    log_path = out_dir / "optimize_log.json"
    log_path.write_text(
        json.dumps(
            {
                "target_alias": cfg.optimize.target_alias,
                "reflection_provider": cfg.optimize.reflection_provider,
                "auto": cfg.optimize.auto,
                "train_case_ids": sorted(train_ids),
                "base_instructions": base_instructions,
                "optimized_instructions": optimized_instructions,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[optimize] wrote {task_path}")
    print(f"[optimize] wrote {log_path}")

    variant_name = f"{cfg.optimize.target_alias}_{ts}"
    variant_config = build_variant_config(cfg.optimize.target_alias, task_path)
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)
    variant_path = VARIANTS_DIR / f"{variant_name}.yaml"
    variant_path.write_text(yaml.safe_dump(variant_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[optimize] wrote {variant_path}")

    outcome = run_mod.run(variant=variant_name, config_path=config_path)
    report_mod.report(outcome.run_id)

    base_run_id = _find_latest_base_run(cfg.task.name)
    compare_path = None
    if base_run_id:
        compare_path = compare(base_run_id, outcome.run_id)
    else:
        print("[optimize] no prior base run found in results/index.jsonl; skipping compare")

    return OptimizeOutcome(
        variant_name=variant_name,
        task_path=task_path,
        variant_path=variant_path,
        run_id=outcome.run_id,
        base_run_id=base_run_id,
        compare_path=compare_path,
    )


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _fmt_pct(v):
    return f"{v:.1%}" if v is not None else "n/a"


def _fmt_pct_signed(v):
    return f"{v:+.1%}" if v is not None else "n/a"


def _fmt_usd(v):
    return f"${v:.4f}" if v is not None else "n/a"


def _fmt_usd_signed(v):
    return f"{'+' if v >= 0 else ''}${v:.4f}" if v is not None else "n/a"


def compare(run_a: str, run_b: str) -> Path:
    output_a = run_mod.RUNS_DIR / run_a / "output.json"
    output_b = run_mod.RUNS_DIR / run_b / "output.json"
    if not output_a.exists():
        raise OptimizeError(f"run {run_a!r} not found ({output_a})")
    if not output_b.exists():
        raise OptimizeError(f"run {run_b!r} not found ({output_b})")

    stats_a = {s.alias: s for s in report_mod.compute_alias_stats(parse_promptfoo_output(output_a).results)}
    stats_b = {s.alias: s for s in report_mod.compute_alias_stats(parse_promptfoo_output(output_b).results)}
    aliases = sorted(set(stats_a) | set(stats_b))

    lines = [
        f"# Compare: {run_a} (A, before) vs {run_b} (B, after)",
        "",
        "| alias | pass_rate A | pass_rate B | delta | beyond_95ci | cost A | cost B | cost delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alias in aliases:
        a, b = stats_a.get(alias), stats_b.get(alias)
        pa = a.pass_rate if a else None
        pb = b.pass_rate if b else None
        delta = (pb - pa) if (pa is not None and pb is not None) else None
        # issue #11: flag whether the delta clears the noise floor -- "yes"
        # only when the two Wilson 95% intervals do not overlap at all
        if a and b and a.pass_ci_low is not None and b.pass_ci_low is not None:
            non_overlap = b.pass_ci_low > a.pass_ci_high or b.pass_ci_high < a.pass_ci_low
            beyond_ci = "yes" if non_overlap else "no"
        else:
            beyond_ci = "n/a"
        ca = a.total_cost_usd if a else None
        cb = b.total_cost_usd if b else None
        cdelta = (cb - ca) if (ca is not None and cb is not None) else None
        lines.append(
            f"| {alias} | {_fmt_pct(pa)} | {_fmt_pct(pb)} | {_fmt_pct_signed(delta)} | {beyond_ci} | "
            f"{_fmt_usd(ca)} | {_fmt_usd(cb)} | {_fmt_usd_signed(cdelta)} |"
        )
    lines.append("")
    lines.append(
        "> beyond_95ci: yes when the Wilson 95% intervals of A and B do not overlap "
        "(a conservative significance check; overlapping intervals mean the delta may be noise)."
    )
    lines.append("")

    report_mod.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = report_mod.REPORTS_DIR / f"compare_{run_a}_{run_b}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[compare] wrote {path}")
    return path
