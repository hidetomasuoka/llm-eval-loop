"""Demo-order sensitivity helpers (APO-19 / issue #78).

``evalloop build --shuffle-demos N`` writes N promptfoo variant configs whose
prompts differ only in the seed-shuffled order of few-shot demos. Operators then
``run`` / ``report`` each variant and ``compare`` the runs manually — this module
does not automate the eval loop.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from evalloop.demos import (
    DEMOS_PLACEHOLDER,
    DemoError,
    assert_demos_do_not_leak_test,
    format_demos,
    load_demos_jsonl,
    shuffle_demos,
)
from evalloop.paths import REPO_ROOT, TaskPaths
from evalloop.schemas import Config, load_golden_jsonl

_DEMOSHUFFLE_PROMPT_RE = re.compile(r"^demoshuffle_(\d+)\.txt$")
_DEMOSHUFFLE_VARIANT_RE = re.compile(r"^(.+)_demoshuffle_(\d+)\.yaml$")


class SensitivityError(RuntimeError):
    pass


def demoshuffle_variant_name(task: str, seed: int) -> str:
    return f"{task}_demoshuffle_{seed}"


def _reroot_file_refs(obj, prefix: str):
    if isinstance(obj, dict):
        return {k: _reroot_file_refs(v, prefix) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_reroot_file_refs(v, prefix) for v in obj]
    if isinstance(obj, str) and obj.startswith("file://"):
        return "file://" + prefix + obj[len("file://") :]
    return obj


def _load_holdout_from_build(paths: TaskPaths) -> tuple[set[str], set[str]]:
    """Return (case_ids, inputs) from the last build's tests_test.yaml.

    Mirrors optimize._load_holdout_from_build so demo leak checks cover the
    YAML holdout promptfoo actually evaluates (DESIGN §5.6).
    """
    if not paths.tests_test.exists():
        raise SensitivityError(
            f"{paths.tests_test} not found; run `evalloop build` before --shuffle-demos"
        )
    entries = yaml.safe_load(paths.tests_test.read_text(encoding="utf-8")) or []
    ids: set[str] = set()
    inputs: set[str] = set()
    for entry in entries:
        vars_ = entry.get("vars") or {}
        case_id = vars_.get("case_id")
        if case_id is not None:
            ids.add(str(case_id))
        inp = vars_.get("input")
        if inp is not None:
            inputs.add(str(inp))
    return ids, inputs


def _clear_stale_demoshuffle_artifacts(paths: TaskPaths, n: int) -> None:
    """Remove demoshuffle prompts/variants whose seed is outside 0..n-1."""
    keep_seeds = set(range(n))
    if paths.build_dir.is_dir():
        for path in paths.build_dir.glob("demoshuffle_*.txt"):
            m = _DEMOSHUFFLE_PROMPT_RE.match(path.name)
            if m and int(m.group(1)) not in keep_seeds:
                path.unlink(missing_ok=True)
                print(f"[build] removed stale demoshuffle prompt {path.name}")
    if paths.variants_dir.is_dir():
        prefix = f"{paths.task}_demoshuffle_"
        for path in paths.variants_dir.glob(f"{prefix}*.yaml"):
            m = _DEMOSHUFFLE_VARIANT_RE.match(path.name)
            if not m:
                continue
            task_name, seed_s = m.group(1), m.group(2)
            if task_name != paths.task:
                continue
            if int(seed_s) not in keep_seeds:
                path.unlink(missing_ok=True)
                print(f"[build] removed stale demoshuffle variant {path.name}")


def _variant_config_for_prompt(prompt_path: Path, paths: TaskPaths, *, seed: int) -> dict:
    if not paths.promptfoo_config.exists():
        raise SensitivityError(
            f"{paths.promptfoo_config} not found; run `evalloop build` before --shuffle-demos"
        )
    base_config = yaml.safe_load(paths.promptfoo_config.read_text(encoding="utf-8"))
    variant_config = _reroot_file_refs(base_config, prefix="../")
    rel = os.path.relpath(prompt_path, start=paths.variants_dir).replace(os.sep, "/")
    variant_config["prompts"] = [f"file://{rel}"]
    base_desc = base_config.get("description", "")
    variant_config["description"] = f"{base_desc} (demoshuffle seed={seed})".strip()
    return variant_config


def build_demoshuffle_variants(config: Config, paths: TaskPaths, n: int) -> list[str]:
    """Write N demoshuffle variants (seeds 0..N-1). Returns variant names."""
    if n < 1:
        raise SensitivityError("--shuffle-demos must be a positive integer")

    prompt_path = REPO_ROOT / config.task.prompt_file
    if not prompt_path.exists():
        raise SensitivityError(f"prompt file not found: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    if DEMOS_PLACEHOLDER not in template:
        raise SensitivityError(
            f"--shuffle-demos requires {DEMOS_PLACEHOLDER} in {config.task.prompt_file}; "
            "this task has no demos placeholder"
        )
    if not paths.demos.exists():
        raise SensitivityError(
            f"--shuffle-demos requires {paths.demos}; demos.jsonl is missing for this task"
        )

    cases = load_golden_jsonl(paths.golden)
    golden_test_cases = [c for c in cases if c.split == "test"]
    yaml_test_ids, yaml_test_inputs = _load_holdout_from_build(paths)
    demos_test_ids = yaml_test_ids | {c.id for c in golden_test_cases}
    demos_test_inputs = yaml_test_inputs | {c.input for c in golden_test_cases}
    try:
        demos = load_demos_jsonl(paths.demos)
        assert_demos_do_not_leak_test(
            demos,
            test_ids=demos_test_ids,
            test_inputs=demos_test_inputs,
        )
    except DemoError as e:
        raise SensitivityError(str(e)) from e

    if len(demos) < 2:
        raise SensitivityError(
            f"--shuffle-demos needs at least 2 demos to permute order; found {len(demos)}"
        )

    paths.build_dir.mkdir(parents=True, exist_ok=True)
    paths.variants_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_demoshuffle_artifacts(paths, n)

    names: list[str] = []
    for seed in range(n):
        shuffled = shuffle_demos(demos, seed)
        resolved = template.replace(DEMOS_PLACEHOLDER, format_demos(shuffled))
        resolved_path = paths.build_dir / f"demoshuffle_{seed}.txt"
        resolved_path.write_text(resolved, encoding="utf-8")

        variant_name = demoshuffle_variant_name(paths.task, seed)
        variant_path = paths.variants_dir / f"{variant_name}.yaml"
        variant_config = _variant_config_for_prompt(resolved_path, paths, seed=seed)
        variant_path.write_text(
            yaml.safe_dump(variant_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        names.append(variant_name)
        print(f"[build] wrote demoshuffle variant {variant_name} -> {variant_path}")

    print(
        "[build] demoshuffle next steps: "
        f"`evalloop run --variant {names[0]}` for each seed, then "
        "`evalloop compare --runs <run_ids>` to inspect order sensitivity"
    )
    return names
