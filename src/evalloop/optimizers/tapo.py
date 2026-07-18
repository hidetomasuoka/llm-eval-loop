"""TAPO-inspired prompt optimizer (APO-23 / issue #99).

Paper: Task-Referenced Adaptation for Prompt Optimization (ICASSP 2025,
arXiv:2501.06689). Official code (Apache-2.0) depends on OpenAI direct calls
plus torch / sentence-transformers / GPT-2 / nltk / rouge — incompatible with
this harness's iron rules and "no new deps" policy.

This module is a **scratch** adaptation of the paper's three modules onto the
existing PromptOptimizer contract:

1. Task-aware metric selection — ``answer_type`` picks proxy metrics
2. Multi-metric evaluation — primary proxy + deterministic secondary (no embeddings)
3. Evolution-based optimization — mutation via ``reflection_lm`` (dspy.LM),
   selection on the TRAIN split only

``run_tapo()`` is the monkeypatch seam (same convention as ``run_gepa`` /
``run_miprov2`` / ``run_copro``).
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable

import dspy

from evalloop.optimizers.base import OptimizeError, OptimizeResult
from evalloop.optimizers.metrics import compute_train_score
from evalloop.optimizers.miprov2 import _scalar_metric
from evalloop.schemas import Config

# Lightweight mutation / thinking templates (not copied from the official repo).
_MUTATION_PROMPTS = (
    "Rewrite the instruction to be clearer and more specific for this task.",
    "Tighten constraints so the model output format is unambiguous.",
    "Remove redundancy while preserving all necessary requirements.",
    "Add one short checklist the model must follow before answering.",
)

_THINKING_STYLES = (
    "Think step by step about failure modes of the current instruction.",
    "Prefer precision over verbosity.",
    "Optimize for consistent formatting across cases.",
)

_PRIMARY_WEIGHT = 0.8
_SECONDARY_WEIGHT = 0.2


def select_metrics_for_answer_type(answer_type: str) -> list[str]:
    """Task-aware metric names recorded in optimize_log (APO-23)."""
    if answer_type == "label":
        return ["label_match", "single_line_brevity"]
    if answer_type == "json":
        return ["json_deep_equal", "valid_json"]
    if answer_type == "text":
        return ["token_f1", "length_ratio"]
    raise OptimizeError(f"tapo: unsupported answer_type {answer_type!r}")


def _secondary_score(output: str, expected, answer_type: str) -> float:
    text = output or ""
    if answer_type == "label":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return 1.0 if len(lines) <= 1 and len(text) < 80 else 0.3
    if answer_type == "json":
        try:
            json.loads(text)
            return 1.0
        except (json.JSONDecodeError, TypeError):
            return 0.0
    # text: reward outputs whose length is within 0.5x..2x of expected text
    exp = expected if isinstance(expected, str) else json.dumps(expected, ensure_ascii=False)
    if not exp:
        return 0.5
    ratio = len(text) / max(len(exp), 1)
    if 0.5 <= ratio <= 2.0:
        return 1.0
    if 0.25 <= ratio <= 4.0:
        return 0.5
    return 0.0


def _lm_text(prompt_model, prompt: str) -> str:
    """Best-effort extract a string from a dspy.LM call."""
    raw = prompt_model(prompt)
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list) and raw:
        first = raw[0]
        return str(first).strip() if not isinstance(first, dict) else str(first.get("text", first)).strip()
    if isinstance(raw, dict):
        for key in ("text", "content", "output"):
            if key in raw and raw[key]:
                return str(raw[key]).strip()
    return str(raw).strip()


def _mutate_instruction(prompt_model, instruction: str, rng: random.Random) -> str:
    mut = rng.choice(_MUTATION_PROMPTS)
    style = rng.choice(_THINKING_STYLES)
    prompt = (
        f"{style}\n{mut}\n\nCurrent instruction:\n```\n{instruction}\n```\n\n"
        "Return ONLY the improved instruction text, with no commentary."
    )
    mutated = _lm_text(prompt_model, prompt)
    return mutated if mutated else instruction


def _fitness(
    instructions: str,
    trainset: list,
    metric: Callable,
    answer_type: str,
) -> float:
    signature = dspy.Signature("input -> output", instructions=instructions)
    program = dspy.Predict(signature)
    if not trainset:
        return 0.0
    total = 0.0
    for gold in trainset:
        pred = program(input=gold.input)
        primary = float(metric(gold, pred))
        secondary = _secondary_score(getattr(pred, "output", "") or "", gold.expected, answer_type)
        total += _PRIMARY_WEIGHT * primary + _SECONDARY_WEIGHT * secondary
    return total / len(trainset)


def run_tapo(
    student,
    trainset,
    metric,
    prompt_model,
    task_model,
    population_size: int,
    generations: int,
    seed: int,
    answer_type: str,
):
    """Evolutionary TAPO loop (monkeypatch target). Returns a dspy program."""
    if population_size < 2:
        raise OptimizeError("tapo population_size must be >= 2")
    if generations < 1:
        raise OptimizeError("tapo generations must be >= 1")
    if not trainset:
        raise OptimizeError("tapo requires a non-empty trainset")

    rng = random.Random(seed)
    dspy.configure(lm=task_model)
    base = student.signature.instructions
    selected = select_metrics_for_answer_type(answer_type)

    population = [base]
    while len(population) < population_size:
        population.append(_mutate_instruction(prompt_model, base, rng))

    generation_scores: list[dict] = []
    best_instr = base
    best_fitness = -1.0

    for gen in range(generations):
        scored = [(_fitness(instr, trainset, metric, answer_type), instr) for instr in population]
        scores = [s for s, _ in scored]
        gen_best = max(scores)
        gen_mean = sum(scores) / len(scores)
        generation_scores.append(
            {"generation": gen, "best_fitness": round(gen_best, 6), "mean_fitness": round(gen_mean, 6)}
        )
        for fit, instr in scored:
            if fit > best_fitness:
                best_fitness = fit
                best_instr = instr

        scored.sort(key=lambda x: -x[0])
        elite_n = max(1, population_size // 2)
        elites = [instr for _, instr in scored[:elite_n]]
        next_pop = list(elites)
        while len(next_pop) < population_size:
            parent = rng.choice(elites)
            next_pop.append(_mutate_instruction(prompt_model, parent, rng))
        population = next_pop

    optimized = dspy.Predict(dspy.Signature("input -> output", instructions=best_instr))
    optimized.tapo_selected_metrics = selected
    optimized.tapo_generation_scores = generation_scores
    optimized.tapo_best_fitness = best_fitness
    return optimized


class TapoOptimizer:
    """PromptOptimizer implementation: TAPO-inspired evolution (instruction-only)."""

    name = "tapo"

    def optimize(
        self,
        *,
        base_instructions: str,
        trainset: list,
        metric: Callable,
        task_lm,
        reflection_lm,
        cfg: Config,
    ) -> OptimizeResult:
        params = cfg.optimize.params
        population_size = int(params.get("population_size", 4))
        generations = int(params.get("generations", 3))
        seed = int(params.get("seed", 0))
        answer_type = cfg.task.answer_type
        selected = select_metrics_for_answer_type(answer_type)

        dspy.configure(lm=task_lm)
        signature = dspy.Signature("input -> output", instructions=base_instructions)
        student = dspy.Predict(signature)

        from evalloop import optimize as optimize_mod

        optimized_program = optimize_mod.run_tapo(
            student,
            trainset,
            _scalar_metric(metric),
            reflection_lm,
            task_lm,
            population_size,
            generations,
            seed,
            answer_type,
        )
        extra_log = {
            "population_size": population_size,
            "generations": generations,
            "seed": seed,
            "train_size": len(trainset),
            "selected_metrics": getattr(optimized_program, "tapo_selected_metrics", selected),
            "generation_scores": getattr(optimized_program, "tapo_generation_scores", []),
        }
        best_fit = getattr(optimized_program, "tapo_best_fitness", None)
        if best_fit is not None:
            extra_log["best_fitness"] = best_fit
        train_score = compute_train_score(trainset, metric, optimized_program)
        if train_score is not None:
            extra_log["train_score"] = train_score
        return OptimizeResult(
            optimized_instructions=optimized_program.signature.instructions,
            method=self.name,
            extra_log=extra_log,
        )
