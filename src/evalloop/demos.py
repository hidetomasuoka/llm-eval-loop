"""Few-shot demos.jsonl loading, formatting, and test-split leak checks (APO-16)."""

from __future__ import annotations

import json
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
            demos.append(
                DemoCase(
                    input=str(row["input"]),
                    output=str(row["output"]),
                    id=str(row["id"]) if row.get("id") is not None else None,
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
