"""Demo-order sensitivity helpers (APO-19 / issue #78).

``evalloop build --shuffle-demos N`` writes N promptfoo variant configs whose
prompts differ only in the seed-shuffled order of few-shot demos. Operators then
``run`` / ``report`` each variant and ``compare`` the runs manually — this module
does not automate the eval loop.
"""

from __future__ import annotations

import os
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
    test_cases = [c for c in cases if c.split == "test"]
    try:
        demos = load_demos_jsonl(paths.demos)
        assert_demos_do_not_leak_test(
            demos,
            test_ids={c.id for c in test_cases},
            test_inputs={c.input for c in test_cases},
        )
    except DemoError as e:
        raise SensitivityError(str(e)) from e

    if len(demos) < 2:
        raise SensitivityError(
            f"--shuffle-demos needs at least 2 demos to permute order; found {len(demos)}"
        )

    paths.build_dir.mkdir(parents=True, exist_ok=True)
    paths.variants_dir.mkdir(parents=True, exist_ok=True)

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
