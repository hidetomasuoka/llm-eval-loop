"""Few-shot demos.jsonl loading, formatting, and test-split leak checks (APO-16)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DEMOS_PLACEHOLDER = "{{demos}}"


@dataclass(frozen=True)
class DemoCase:
    input: str
    output: str
    id: str | None = None


class DemoError(RuntimeError):
    pass


def format_demos(demos: list[DemoCase]) -> str:
    """Render demos for embedding into a prompt template (pure / unit-testable)."""
    if not demos:
        return ""
    blocks = [f"Input: {d.input}\nOutput: {d.output}" for d in demos]
    return "\n\n".join(blocks) + "\n\n"


def shuffle_demos(demos: list[DemoCase], seed: int) -> list[DemoCase]:
    """Return a new list with demos shuffled reproducibly by ``seed`` (APO-19)."""
    items = list(demos)
    random.Random(seed).shuffle(items)
    return items


def load_demos_jsonl(path: Path) -> list[DemoCase]:
    """Load ``{"input", "output", optional "id"}`` rows from demos.jsonl."""
    demos: list[DemoCase] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DemoError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(row, dict):
                raise DemoError(f"{path}:{line_no}: each line must be a JSON object")
            if "input" not in row or "output" not in row:
                raise DemoError(f"{path}:{line_no}: requires 'input' and 'output' fields")
            if not isinstance(row["input"], str) or not isinstance(row["output"], str):
                raise DemoError(f"{path}:{line_no}: 'input' and 'output' must be strings")
            demo_id = row.get("id")
            if demo_id is not None and not isinstance(demo_id, str):
                raise DemoError(f"{path}:{line_no}: 'id' must be a string when provided")
            demos.append(
                DemoCase(
                    input=row["input"],
                    output=row["output"],
                    id=demo_id,
                )
            )
    if not demos:
        raise DemoError(f"{path} contains no demo rows")
    return demos


def assert_demos_do_not_leak_test(
    demos: list[DemoCase],
    *,
    test_ids: set[str],
    test_inputs: set[str],
) -> None:
    """Iron rule: demos must not include golden.jsonl test-split cases."""
    for demo in demos:
        if demo.id is not None and demo.id in test_ids:
            raise DemoError(
                f"demos.jsonl leaks test-split case id {demo.id!r}; "
                "few-shot demos must not include holdout cases"
            )
        if demo.input in test_inputs:
            raise DemoError(
                "demos.jsonl leaks a test-split input (exact string match); "
                "few-shot demos must not include holdout cases"
            )


def _example_field(ex, name: str):
    if hasattr(ex, "keys"):
        try:
            if name in ex.keys():
                return getattr(ex, name)
        except Exception:
            pass
    return getattr(ex, name, None)


def demos_from_dspy_program(
    program, *, train_input_to_id: dict[str, str]
) -> list[tuple[DemoCase, str]]:
    """Extract few-shot demos from a compiled dspy program (APO-17).

    Returns ``(DemoCase, origin)`` where origin is ``labeled`` or ``bootstrapped``.
    Raises ``DemoError`` if a demo input is not in the train-split map.
    """
    predictors = getattr(program, "predictors", None)
    if not callable(predictors):
        return []
    preds = predictors()
    if not preds:
        return []
    raw_demos = getattr(preds[0], "demos", None) or []
    extracted: list[tuple[DemoCase, str]] = []
    for ex in raw_demos:
        inp = _example_field(ex, "input")
        if inp is None:
            continue
        inp_s = str(inp)
        out = _example_field(ex, "output")
        origin = "bootstrapped"
        if out is None:
            out = _example_field(ex, "expected")
            origin = "labeled"
        if out is None:
            raise DemoError(f"optimized demo for input {inp_s!r} has neither output nor expected")
        case_id = train_input_to_id.get(inp_s)
        if case_id is None:
            raise DemoError(
                f"optimized demo input is not from the train split (leak or drift): {inp_s!r}"
            )
        extracted.append((DemoCase(input=inp_s, output=str(out), id=case_id), origin))
    return extracted


def save_demos_jsonl(
    path: Path,
    demos_with_origin: list[tuple[DemoCase, str]],
    *,
    provenance: dict,
) -> None:
    """Write demos.jsonl with per-row provenance meta (APO-17)."""
    if not demos_with_origin:
        raise DemoError("refusing to write empty demos.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for demo, origin in demos_with_origin:
            row = {
                "id": demo.id,
                "input": demo.input,
                "output": demo.output,
                "meta": {**provenance, "origin": origin},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def expand_demos_in_template(
    template: str,
    demos_path: Path,
    *,
    test_ids: set[str],
    test_inputs: set[str],
) -> tuple[str, int | None]:
    """Expand ``{{demos}}`` using demos.jsonl when the placeholder is present.

    Shared by ``evalloop build`` (promptfoo path) and ``evalloop optimize``
    (dspy training template) so both see the same rendered prompt.

    Returns ``(text, n_demos)`` when expanded, or ``(template, None)`` when
    the placeholder is absent (caller may still warn about an unused demos file).
    """
    if DEMOS_PLACEHOLDER not in template:
        return template, None
    if not demos_path.exists():
        raise DemoError(
            f"prompt contains {DEMOS_PLACEHOLDER} but {demos_path} is missing. "
            "Add demos.jsonl (see docs/DESIGN.md §5.6) or remove the placeholder."
        )
    demos = load_demos_jsonl(demos_path)
    assert_demos_do_not_leak_test(demos, test_ids=test_ids, test_inputs=test_inputs)
    return template.replace(DEMOS_PLACEHOLDER, format_demos(demos)), len(demos)
