"""Execute a process graph: promptfoo for LLM nodes, Python for the rest, echo+assert to grade."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from evalloop import build as build_mod
from evalloop import run as run_mod
from evalloop.paths import TaskPaths
from evalloop.process.graph import unique_successor, validate_graph
from evalloop.process.interp import assemble_end, render_template, route_if_else, stringify
from evalloop.process.schema import ProcessError, ProcessGraph, ProcessNode, load_process_graph
from evalloop.schemas import Config, _extract_result_rows, parse_promptfoo_output

EvalFn = Callable[..., object]


def load_validated_process(config: Config) -> ProcessGraph:
    if not config.task.process_file:
        raise ProcessError("task.process_file is not set")
    graph = load_process_graph(config.task.process_file)
    validate_graph(graph)
    return graph


def resolve_node_text(node: ProcessNode, task_dir: Path) -> str:
    if node.type == "llm":
        assert node.prompt_file is not None
        path = task_dir / node.prompt_file
        if not path.is_file():
            raise ProcessError(f"llm node {node.id!r}: prompt_file not found: {path}")
        return path.read_text(encoding="utf-8")
    if node.type == "template":
        if node.template is not None:
            return node.template
        assert node.template_file is not None
        path = task_dir / node.template_file
        if not path.is_file():
            raise ProcessError(f"template node {node.id!r}: template_file not found: {path}")
        return path.read_text(encoding="utf-8")
    raise ProcessError(f"node {node.id!r} has no text body")


def _provider_block(config: Config, alias: str) -> dict:
    model = config.model_by_alias(alias)
    provider_config: dict = {}
    if model.supports_sampling_params:
        provider_config["temperature"] = config.run.temperature
    provider_config["max_tokens"] = config.run.max_tokens
    return {"id": model.provider, "label": model.alias, "config": provider_config}


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _promptfoo_rows(output_path: Path) -> list:
    parsed = parse_promptfoo_output(output_path)
    return parsed.results


def _first_output_by_case(results, alias: str) -> dict[str, tuple[str, float, str | None]]:
    """Map case_id -> (output, cost, error) using the first repeat for this alias."""
    found: dict[str, tuple[str, float, str | None]] = {}
    for row in results:
        if row.alias != alias or not row.case_id or row.case_id in found:
            continue
        found[row.case_id] = (row.output or "", float(row.cost or 0.0), row.error)
    return found


def _run_llm_wave(
    *,
    config: Config,
    paths: TaskPaths,
    node: ProcessNode,
    alias: str,
    cases: list[dict],
    states: dict[str, dict[str, str]],
    run_dir: Path,
    repeat: int,
    no_cache: bool,
    timeout_s: int | None,
    eval_fn: EvalFn,
) -> dict[str, tuple[str, float, str | None]]:
    prompt_text = resolve_node_text(node, paths.task_dir)
    tests = []
    for case in cases:
        case_id = case["vars"]["case_id"]
        vars_ = {"case_id": case_id, **states[case_id]}
        tests.append({"description": case_id, "vars": vars_})
    cfg_path = run_dir / "promptfoo" / alias / f"{node.id}.yaml"
    out_path = run_dir / "promptfoo" / alias / f"{node.id}.output.json"
    payload = {
        "description": f"evalloop process node={node.id}",
        "providers": [_provider_block(config, alias)],
        "prompts": [prompt_text],
        "tests": tests,
    }
    _write_yaml(cfg_path, payload)
    proc = eval_fn(cfg_path, out_path, repeat=repeat, no_cache=no_cache, timeout_s=timeout_s)
    if not out_path.exists():
        err = getattr(proc, "stderr", "") or "promptfoo produced no output.json"
        return {c["vars"]["case_id"]: ("", 0.0, err) for c in cases}
    return _first_output_by_case(_promptfoo_rows(out_path), alias)


def _grade_echo(
    *,
    config: Config,
    paths: TaskPaths,
    alias: str,
    cases: list[dict],
    finals: dict[str, str],
    costs: dict[str, float],
    run_dir: Path,
    eval_fn: EvalFn,
    timeout_s: int | None,
) -> tuple[list[dict], int]:
    tests = []
    for case in cases:
        vars_ = dict(case["vars"])
        case_id = vars_["case_id"]
        vars_["output_raw"] = finals.get(case_id, "")
        tests.append({"description": case_id, "vars": vars_})
    default_test = build_mod._build_default_test(config, allow_same_judge=True, paths=paths)
    cfg_path = paths.promptfoo_dir / f"_process_echo_{run_dir.name}_{alias}.yaml"
    out_path = run_dir / "promptfoo" / alias / "echo.output.json"
    payload = {
        "description": f"evalloop process echo grade alias={alias}",
        "providers": [{"id": "echo", "label": alias}],
        "prompts": ["{{output_raw}}"],
        "defaultTest": default_test,
        "tests": tests,
    }
    _write_yaml(cfg_path, payload)
    proc = eval_fn(cfg_path, out_path, repeat=1, no_cache=True, timeout_s=timeout_s)
    exit_code = getattr(proc, "returncode", 0) or 0
    if not out_path.exists():
        return [], exit_code if exit_code else 1
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    rows, _warns = _extract_result_rows(raw)
    patched: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row = dict(row)
        vars_ = row.get("vars") or {}
        case_id = vars_.get("case_id")
        if case_id in costs:
            row["cost"] = costs[case_id]
        provider = row.get("provider")
        if isinstance(provider, dict):
            provider = dict(provider)
            provider["label"] = alias
            model = next((m for m in config.models if m.alias == alias), None)
            if model is not None:
                provider["id"] = model.provider
            row["provider"] = provider
        patched.append(row)
    return patched, exit_code


def execute_graph_for_alias(
    *,
    config: Config,
    paths: TaskPaths,
    graph: ProcessGraph,
    alias: str,
    cases: list[dict],
    run_dir: Path,
    repeat: int,
    no_cache: bool,
    timeout_s: int | None,
    eval_fn: EvalFn,
) -> tuple[dict[str, str], dict[str, float], list[dict], dict[str, str]]:
    """Walk every case through the DAG for one model.

    Returns (final_output_by_case, cost_by_case, trace_rows, last_node_by_case).
    """
    by_id = graph.node_map()
    start = graph.start()
    first = unique_successor(graph, start.id)
    states: dict[str, dict[str, str]] = {}
    cursor: dict[str, str] = {}
    for case in cases:
        vars_ = case["vars"]
        case_id = vars_["case_id"]
        states[case_id] = {"input": stringify(vars_.get("input", ""))}
        cursor[case_id] = first

    traces: list[dict] = []
    costs: dict[str, float] = defaultdict(float)
    last_transform: dict[str, str] = {c["vars"]["case_id"]: start.id for c in cases}
    max_steps = len(graph.nodes) + 2

    for _ in range(max_steps):
        pending = [cid for cid, nid in cursor.items() if by_id[nid].type != "end"]
        if not pending:
            break
        grouped: dict[str, list[str]] = defaultdict(list)
        for case_id in pending:
            grouped[cursor[case_id]].append(case_id)

        for node_id, case_ids in grouped.items():
            node = by_id[node_id]
            if node.type == "if-else":
                for case_id in case_ids:
                    nxt = route_if_else(node, states[case_id])
                    traces.append(
                        {
                            "case_id": case_id,
                            "alias": alias,
                            "node_id": node.id,
                            "type": node.type,
                            "output": nxt,
                        }
                    )
                    cursor[case_id] = nxt
                continue

            if node.type == "template":
                template = resolve_node_text(node, paths.task_dir)
                assert node.output is not None
                for case_id in case_ids:
                    rendered = render_template(template, states[case_id]).strip()
                    states[case_id][node.output] = rendered
                    last_transform[case_id] = node.id
                    traces.append(
                        {
                            "case_id": case_id,
                            "alias": alias,
                            "node_id": node.id,
                            "type": node.type,
                            "output": rendered,
                        }
                    )
                    cursor[case_id] = unique_successor(graph, node.id)
                continue

            if node.type != "llm":
                raise ProcessError(f"cannot execute node type {node.type!r} mid-graph ({node.id})")

            wave_cases = [c for c in cases if c["vars"]["case_id"] in set(case_ids)]
            outputs = _run_llm_wave(
                config=config,
                paths=paths,
                node=node,
                alias=alias,
                cases=wave_cases,
                states=states,
                run_dir=run_dir,
                repeat=repeat,
                no_cache=no_cache,
                timeout_s=timeout_s,
                eval_fn=eval_fn,
            )
            assert node.output is not None
            nxt = unique_successor(graph, node.id)
            for case_id in case_ids:
                text, cost, error = outputs.get(case_id, ("", 0.0, "missing llm result"))
                text = (text or "").strip()
                states[case_id][node.output] = text
                costs[case_id] += cost
                last_transform[case_id] = node.id
                traces.append(
                    {
                        "case_id": case_id,
                        "alias": alias,
                        "node_id": node.id,
                        "type": node.type,
                        "output": text,
                        "error": error,
                        "cost": cost,
                    }
                )
                cursor[case_id] = nxt if not error else graph.end().id

    finals: dict[str, str] = {}
    end = graph.end()
    for case in cases:
        case_id = case["vars"]["case_id"]
        if cursor.get(case_id) != end.id:
            raise ProcessError(f"case {case_id!r} did not reach end (stuck at {cursor.get(case_id)!r})")
        finals[case_id] = assemble_end(end, states[case_id])
        traces.append(
            {
                "case_id": case_id,
                "alias": alias,
                "node_id": end.id,
                "type": "end",
                "output": finals[case_id],
                "final": True,
            }
        )
    return finals, dict(costs), traces, last_transform


def run_process(
    config: Config,
    paths: TaskPaths,
    *,
    variant: str | None = None,
    repeat: int | None = None,
    limit: int | None = None,
    no_cache: bool = False,
    timeout_s: int | None = None,
    split: str = "test",
    eval_fn: EvalFn | None = None,
) -> run_mod.RunOutcome:
    if variant:
        raise run_mod.RunError("process tasks do not support --variant (graph-level optimize is out of scope)")
    if split not in run_mod.VALID_RUN_SPLITS:
        raise run_mod.RunError(f"split must be one of {run_mod.VALID_RUN_SPLITS}, got {split!r}")

    tests_path = paths.tests_dev if split == "dev" else paths.tests_test
    if not tests_path.exists():
        raise run_mod.RunError(f"{tests_path} not found; run `evalloop build --task {paths.task}` first")

    graph = load_validated_process(config)
    cases = yaml.safe_load(tests_path.read_text(encoding="utf-8")) or []
    if not isinstance(cases, list):
        raise run_mod.RunError(f"{tests_path} must be a list of promptfoo tests")
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise run_mod.RunError(f"{tests_path} has no cases to evaluate")

    effective_repeat = repeat if repeat is not None else config.run.repeat
    runner = eval_fn or run_mod.run_promptfoo_eval
    run_id = run_mod.new_run_id()
    run_dir = paths.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "output.json"
    meta_path = run_dir / "meta.json"
    trace_path = run_dir / "trace.jsonl"

    all_rows: list[dict] = []
    all_traces: list[dict] = []
    last_nodes: dict[tuple[str, str], str] = {}
    worst_exit = 0
    process_error: str | None = None
    try:
        for model in config.models:
            finals, costs, traces, last_transform = execute_graph_for_alias(
                config=config,
                paths=paths,
                graph=graph,
                alias=model.alias,
                cases=cases,
                run_dir=run_dir,
                repeat=effective_repeat,
                no_cache=no_cache,
                timeout_s=timeout_s,
                eval_fn=runner,
            )
            all_traces.extend(traces)
            for case_id, node_id in last_transform.items():
                last_nodes[(case_id, model.alias)] = node_id
            rows, echo_exit = _grade_echo(
                config=config,
                paths=paths,
                alias=model.alias,
                cases=cases,
                finals=finals,
                costs=costs,
                run_dir=run_dir,
                eval_fn=runner,
                timeout_s=timeout_s,
            )
            all_rows.extend(rows)
            if echo_exit:
                worst_exit = echo_exit
    except ProcessError as e:
        process_error = str(e)
        worst_exit = worst_exit or 1
        (run_dir / "process_error.txt").write_text(process_error, encoding="utf-8")

    with trace_path.open("w", encoding="utf-8") as f:
        for row in all_traces:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "last_nodes.json").write_text(
        json.dumps({f"{k[0]}\t{k[1]}": v for k, v in last_nodes.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    combined = {"evalId": run_id, "results": {"version": 3, "prompts": [], "results": all_rows}}
    output_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    actual_cost = run_mod._actual_cost_from_output(output_path)
    process_path = Path(config.task.process_file) if config.task.process_file else None
    grader_type = {"text": "llm-rubric", "label": "label-match", "json": "json-field-match"}[config.task.answer_type]
    agreement_rate = None
    if grader_type == "llm-rubric":
        from evalloop.calibrate import load_task_calibration

        snap = load_task_calibration(paths, judge_provider=config.judge.provider)
        if snap and snap.get("calibration_status"):
            calibration_status = str(snap["calibration_status"])
            agreement_rate = snap.get("agreement_rate")
        else:
            calibration_status = "uncalibrated"
    else:
        calibration_status = "not_applicable"
    grader = {"type": grader_type, "calibration_status": calibration_status}
    if grader_type == "llm-rubric":
        grader.update({"provider": config.judge.provider, "agreement_rate": agreement_rate})

    meta = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": paths.task,
        "task_name": config.task.name,
        "answer_type": config.task.answer_type,
        "kind": "process",
        "variant": None,
        "split": split,
        "promptfoo_config_path": run_mod._display_path(paths.promptfoo_config),
        "promptfoo_config_sha256": (
            run_mod.sha256_of_file(paths.promptfoo_config) if paths.promptfoo_config.exists() else None
        ),
        "process_file": run_mod._display_path(process_path) if process_path else None,
        "process_sha256": run_mod.sha256_of_file(process_path) if process_path and process_path.is_file() else None,
        "prompt_file": None,
        "prompt_sha256": None,
        "golden_sha256": run_mod.sha256_of_file(paths.golden) if paths.golden.exists() else None,
        "repeat": effective_repeat,
        "limit": limit,
        "no_cache": no_cache,
        "models": [{"alias": m.alias, "provider": m.provider, "tier": m.tier} for m in config.models],
        "actual_cost_usd": actual_cost,
        "grader": grader,
        "judge": {
            "provider": config.judge.provider,
            "calibration_status": calibration_status,
            "agreement_rate": agreement_rate,
        },
        "promptfoo_version": run_mod.get_promptfoo_version(),
        "node_version": run_mod.get_node_version(),
        "evalloop_command": (
            f"evalloop run --task {paths.task}"
            f"{' --split dev' if split == 'dev' else ''}"
            f" --repeat {effective_repeat}{f' --limit {limit}' if limit else ''}{' --no-cache' if no_cache else ''}"
        ),
        "promptfoo_exit_code": worst_exit,
        "promptfoo_stderr_tail": "",
        "trace_path": run_mod._display_path(trace_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    with paths.index.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "created_at": meta["created_at"],
                    "task": paths.task,
                    "task_name": config.task.name,
                    "variant": None,
                    "split": split,
                    "kind": "process",
                    "actual_cost_usd": actual_cost,
                    "promptfoo_exit_code": worst_exit,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    print(f"[run] run_id={run_id} (process)")
    print(f"[run] output -> {output_path}")
    print(f"[run] trace  -> {trace_path}")
    print(f"[run] meta   -> {meta_path}")
    print(f"[run] actual cost: ${actual_cost:.4f}")
    if process_error:
        raise run_mod.RunError(process_error)
    if worst_exit not in (0, 100):
        raise run_mod.RunError(f"process eval failed with exit code {worst_exit} (see {run_dir})")
    return run_mod.RunOutcome(run_id=run_id, output_path=output_path, meta_path=meta_path, meta=meta)


def write_node_promptfoo_configs(config: Config, paths: TaskPaths, graph: ProcessGraph) -> None:
    """Inspectable per-llm-node configs (providers + prompt). Tests are filled at run time."""
    paths.process_nodes_dir.mkdir(parents=True, exist_ok=True)
    providers = []
    for m in config.models:
        provider_config: dict = {}
        if m.supports_sampling_params:
            provider_config["temperature"] = config.run.temperature
        provider_config["max_tokens"] = config.run.max_tokens
        providers.append({"id": m.provider, "label": m.alias, "config": provider_config})
    for node in graph.llm_nodes():
        prompt_path = paths.task_dir / (node.prompt_file or "")
        payload = {
            "description": f"evalloop process node={node.id} (template; tests filled at run time)",
            "providers": providers,
            "prompts": [f"file://{build_mod.to_promptfoo_relpath(prompt_path, paths.process_nodes_dir)}"],
        }
        _write_yaml(paths.process_nodes_dir / f"{node.id}.yaml", payload)
