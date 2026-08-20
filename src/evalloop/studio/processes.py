"""LangChain-like process (chain) specs and a local DAG executor."""

from __future__ import annotations

import json
import re
from typing import Any

import yaml

from evalloop.studio.errors import StudioError
from evalloop.studio.expr import safe_eval
from evalloop.studio.ids import new_run_id, utcnow, validate_id
from evalloop.studio.render import lookup, render
from evalloop.studio.store import StudioStore, atomic_write_json, atomic_write_yaml

KNOWN_STEP_KINDS = {
    "prompt",
    "set",
    "transform",
    "retrieve",
    "classify",
    "predict",
    "llm",
    "tool",
    "branch",
    "switch",
    "map",
    "expr",
    "parse",
    "memory",
    "subprocess",
    "agent",
}

BUILTIN_TOOLS = {
    "utcnow": lambda _args, _state: utcnow(),
    "upper": lambda args, state: str(_arg(args, state, "value", 0)).upper(),
    "lower": lambda args, state: str(_arg(args, state, "value", 0)).lower(),
    "length": lambda args, state: len(_arg(args, state, "value", 0)),
    "json_dumps": lambda args, state: json.dumps(_arg(args, state, "value", 0), ensure_ascii=False),
}


def _arg(args: dict[str, Any], state: dict[str, Any], name: str, index: int) -> Any:
    if name in args:
        value = args[name]
        return render(value, state) if isinstance(value, str) else value
    values = args.get("args")
    if isinstance(values, list) and index < len(values):
        value = values[index]
        return render(value, state) if isinstance(value, str) else value
    if name in state:
        return state[name]
    raise StudioError(f"tool argument {name!r} is missing")


def load_process_spec(raw: dict[str, Any]) -> dict[str, Any]:
    if "id" not in raw or "steps" not in raw:
        raise StudioError("process spec requires id and steps")
    spec = dict(raw)
    spec["id"] = validate_id(str(spec["id"]), "process")
    spec.setdefault("name", spec["id"])
    spec.setdefault("description", "")
    spec.setdefault("inputs", [])
    spec.setdefault("outputs", [])
    steps = spec["steps"]
    if not isinstance(steps, list) or not steps:
        raise StudioError(f"process {spec['id']!r} must have a non-empty steps list")
    for i, step in enumerate(steps):
        _validate_step(step, f"steps[{i}]")
    return spec


def _validate_step(step: Any, where: str) -> None:
    if not isinstance(step, dict):
        raise StudioError(f"{where}: step must be a mapping")
    if "id" not in step or "kind" not in step:
        raise StudioError(f"{where}: step requires id and kind")
    validate_id(str(step["id"]), "step")
    kind = step["kind"]
    if kind not in KNOWN_STEP_KINDS:
        raise StudioError(f"{where}: unknown step kind {kind!r} (known: {sorted(KNOWN_STEP_KINDS)})")
    if kind == "switch":
        cases = step.get("cases") or {}
        if not isinstance(cases, dict) or not cases:
            raise StudioError(f"{where}: switch requires a cases mapping")
        for label, nested in cases.items():
            if not isinstance(nested, list):
                raise StudioError(f"{where}.cases.{label}: expected a list of steps")
            for j, nested_step in enumerate(nested):
                _validate_step(nested_step, f"{where}.cases.{label}[{j}]")
    if kind == "map":
        nested = step.get("steps") or []
        if not isinstance(nested, list) or not nested:
            raise StudioError(f"{where}: map requires a non-empty steps list")
        for j, nested_step in enumerate(nested):
            _validate_step(nested_step, f"{where}.steps[{j}]")
    if kind == "subprocess":
        if not step.get("process"):
            raise StudioError(f"{where}: subprocess requires process")
    if kind == "parse":
        fmt = str(step.get("format") or "json")
        if fmt not in {"json", "regex"}:
            raise StudioError(f"{where}: parse.format must be json or regex")
        if fmt == "regex" and not step.get("pattern"):
            raise StudioError(f"{where}: parse format=regex requires pattern")
    if kind == "agent":
        if not (step.get("tools") or step.get("knowledge") or step.get("model")):
            raise StudioError(f"{where}: agent requires tools, knowledge, or model")


def save_process(store: StudioStore, spec: dict[str, Any]) -> dict[str, Any]:
    spec = load_process_spec(spec)
    path = store.paths.process_file(spec["id"])
    atomic_write_yaml(path, spec)
    return store.put(
        "processes",
        spec["id"],
        {
            "kind": "process",
            "name": spec.get("name"),
            "description": spec.get("description", ""),
            "n_steps": len(spec["steps"]),
            "inputs": spec.get("inputs") or [],
            "outputs": spec.get("outputs") or [],
        },
    )


def load_process(store: StudioStore, process_id: str) -> dict[str, Any]:
    store.get("processes", process_id)
    path = store.paths.process_file(process_id)
    if not path.exists():
        raise StudioError(f"process {process_id!r} is missing {path}")
    return load_process_spec(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def run_process(
    store: StudioStore,
    process_id: str,
    inputs: dict[str, Any] | None = None,
    *,
    record: bool = True,
    _stack: tuple[str, ...] = (),
) -> dict[str, Any]:
    if process_id in _stack:
        raise StudioError(f"subprocess cycle: {' -> '.join([*_stack, process_id])}")
    spec = load_process(store, process_id)
    state: dict[str, Any] = dict(inputs or {})
    trace: list[dict[str, Any]] = []
    stack = (*_stack, process_id)
    _run_steps(store, spec["steps"], state, trace, stack)
    outputs: dict[str, Any] = {}
    declared = spec.get("outputs") or []
    if declared:
        for name in declared:
            key = name["name"] if isinstance(name, dict) else str(name)
            outputs[key] = lookup(state, key, default=state.get(key))
    else:
        outputs = {k: v for k, v in state.items() if not str(k).startswith("_")}
    result = {
        "process_id": process_id,
        "ok": True,
        "inputs": dict(inputs or {}),
        "outputs": outputs,
        "state": state,
        "trace": trace,
        "finished_at": utcnow(),
    }
    if record:
        run_id = new_run_id("proc").lower()
        result["run_id"] = run_id
        atomic_write_json(store.paths.run_file(run_id), result)
        store.put(
            "runs",
            run_id,
            {
                "kind": "process_run",
                "process_id": process_id,
                "ok": True,
                "output_keys": sorted(outputs),
            },
        )
    return result


def _run_steps(
    store: StudioStore,
    steps: list[dict[str, Any]],
    state: dict[str, Any],
    trace: list[dict[str, Any]],
    stack: tuple[str, ...] = (),
) -> None:
    for step in steps:
        value = _run_step(store, step, state, trace, stack)
        output_key = step.get("output") or step["id"]
        state[output_key] = value
        state[step["id"]] = value
        trace.append({"id": step["id"], "kind": step["kind"], "output": output_key, "value": _clip(value)})


def _clip(value: Any, limit: int = 400) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    if isinstance(value, list) and len(value) > 8:
        return value[:8] + [f"…({len(value) - 8} more)"]
    return value


def _run_step(
    store: StudioStore,
    step: dict[str, Any],
    state: dict[str, Any],
    trace: list[dict[str, Any]],
    stack: tuple[str, ...] = (),
) -> Any:
    kind = step["kind"]
    if kind == "prompt":
        template = step.get("template")
        if template is None:
            raise StudioError(f"step {step['id']!r}: prompt requires template")
        return render(str(template), state)
    if kind == "set":
        value = step.get("value")
        return render(value, state) if isinstance(value, str) else value
    if kind == "transform":
        return _transform(step, state)
    if kind == "retrieve":
        from evalloop.studio.knowledge import search

        knowledge_id = step.get("knowledge")
        if not knowledge_id:
            raise StudioError(f"step {step['id']!r}: retrieve requires knowledge")
        query = render(str(step.get("query") or "{{input}}"), state)
        k = int(step.get("k") or 3)
        hits = search(store, knowledge_id, query, k=k)
        return hits
    if kind in {"classify", "predict"}:
        from evalloop.studio import automl as automl_mod
        from evalloop.studio.models import require_trained

        model_id = step.get("model")
        if not model_id:
            raise StudioError(f"step {step['id']!r}: {kind} requires model")
        require_trained(store, model_id)
        row = _feature_row(step, state)
        result = automl_mod.predict_row(store, model_id, row)
        if kind == "classify":
            return result["prediction"]
        return result
    if kind == "llm":
        return _llm(step, state)
    if kind == "tool":
        name = step.get("name")
        if name not in BUILTIN_TOOLS:
            raise StudioError(f"step {step['id']!r}: unknown tool {name!r} (known: {sorted(BUILTIN_TOOLS)})")
        args = {k: v for k, v in step.items() if k not in {"id", "kind", "name", "output"}}
        return BUILTIN_TOOLS[name](args, state)
    if kind == "branch":
        on = render(str(step.get("on") or ""), state)
        cases = step.get("cases") or {}
        if on in cases:
            chosen = cases[on]
        elif str(on) in cases:
            chosen = cases[str(on)]
        else:
            chosen = step.get("default", "")
        return render(chosen, state) if isinstance(chosen, str) else chosen
    if kind == "switch":
        on = render(str(step.get("on") or ""), state)
        cases = step.get("cases") or {}
        nested = cases.get(on) or cases.get(str(on)) or cases.get("default") or []
        _run_steps(store, nested, state, trace, stack)
        output_key = step.get("output") or step["id"]
        return state.get(output_key, on)
    if kind == "map":
        over = step.get("over") or ""
        sequence = lookup(state, str(over).strip("{} "), default=state.get(over, []))
        if isinstance(over, str) and "{{" in over:
            rendered = render(over, state)
            try:
                sequence = json.loads(rendered)
            except json.JSONDecodeError:
                sequence = lookup(state, over.strip("{} "), default=[])
        if not isinstance(sequence, list):
            raise StudioError(f"step {step['id']!r}: map.over did not resolve to a list")
        alias = step.get("as") or "item"
        collect_from = step.get("collect") or (step["steps"][-1].get("output") or step["steps"][-1]["id"])
        collected = []
        for item in sequence:
            child_state = dict(state)
            child_state[alias] = item
            _run_steps(store, step["steps"], child_state, trace, stack)
            collected.append(child_state.get(collect_from))
        return collected
    if kind == "expr":
        expr = step.get("expr")
        if not expr:
            raise StudioError(f"step {step['id']!r}: expr requires expr")
        return safe_eval(str(expr), state)
    if kind == "parse":
        return _parse(step, state)
    if kind == "memory":
        history = state.get("history") or []
        return _format_history(history)
    if kind == "subprocess":
        child_id = str(step["process"])
        mapping = step.get("inputs") or {}
        child_inputs: dict[str, Any] = {}
        if isinstance(mapping, dict) and mapping:
            for key, template in mapping.items():
                child_inputs[key] = render(str(template), state) if isinstance(template, str) else template
        else:
            child_inputs = {k: v for k, v in state.items() if not str(k).startswith("_")}
        nested = run_process(store, child_id, child_inputs, record=False, _stack=stack)
        pick = step.get("pick")
        if pick:
            return nested["outputs"].get(pick)
        return nested["outputs"]
    if kind == "agent":
        return _agent(store, step, state)
    raise StudioError(f"unhandled step kind {kind!r}")


def _transform(step: dict[str, Any], state: dict[str, Any]) -> Any:
    value = lookup(state, str(step.get("field") or "input"), default=state.get("input", ""))
    op = step.get("op") or "identity"
    if op == "identity":
        return value
    if op == "lower":
        return str(value).lower()
    if op == "upper":
        return str(value).upper()
    if op == "strip":
        return str(value).strip()
    if op == "join":
        sep = str(step.get("sep") or "\n")
        if isinstance(value, list):
            return sep.join(
                (item.get("text") if isinstance(item, dict) and "text" in item else str(item)) for item in value
            )
        return str(value)
    if op == "json":
        return json.dumps(value, ensure_ascii=False)
    raise StudioError(f"unknown transform op {op!r}")


def _feature_row(step: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    mapping = step.get("features")
    if isinstance(mapping, dict) and mapping:
        row = {}
        for key, template in mapping.items():
            row[key] = render(str(template), state) if isinstance(template, str) else template
        return row
    field = step.get("input") or "input"
    value = lookup(state, str(field).strip("{} "), default=state.get(field, render(str(field), state)))
    feature_name = step.get("feature") or "input"
    return {feature_name: value}


def _llm(step: dict[str, Any], state: dict[str, Any]) -> str:
    """Local-only LLM step. Never calls a hosted provider (iron rule)."""
    provider = str(step.get("provider") or "template")
    prompt = render(str(step.get("prompt") or step.get("template") or ""), state)
    if provider == "echo":
        return prompt
    if provider == "template":
        template = step.get("template") or step.get("prompt") or ""
        return render(str(template), state)
    if provider == "passthrough":
        field = step.get("field") or "input"
        return str(lookup(state, field, default=""))
    raise StudioError(
        f"llm provider {provider!r} is not a local studio provider. "
        "Use echo/template/passthrough here; hosted models go through `evalloop run` (promptfoo)."
    )


def _parse(step: dict[str, Any], state: dict[str, Any]) -> Any:
    field = str(step.get("field") or "input")
    raw = lookup(state, field, default=state.get(field, ""))
    fmt = str(step.get("format") or "json")
    if fmt == "json":
        if isinstance(raw, (dict, list)):
            return raw
        try:
            return json.loads(str(raw))
        except json.JSONDecodeError as e:
            raise StudioError(f"step {step['id']!r}: JSON parse failed: {e}") from e
    match = re.search(str(step["pattern"]), str(raw))
    if not match:
        return None
    if match.groupdict():
        return match.groupdict()
    if match.groups():
        return list(match.groups())
    return match.group(0)


def _format_history(history: Any) -> str:
    if not isinstance(history, list) or not history:
        return ""
    lines = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role") or "user"
            content = item.get("content") or item.get("text") or ""
            lines.append(f"{role}: {content}")
        else:
            lines.append(str(item))
    return "\n".join(lines)


def _agent(store: StudioStore, step: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    from evalloop.studio import automl as automl_mod
    from evalloop.studio.knowledge import search
    from evalloop.studio.models import require_trained

    query = render(str(step.get("query") or "{{input}}"), state)
    tools = [str(t) for t in (step.get("tools") or ["retrieve"])]
    routes = step.get("routes") or {}
    label = None
    model_id = step.get("model")
    if model_id:
        require_trained(store, model_id)
        label = automl_mod.predict_row(store, model_id, {step.get("feature") or "input": query})["prediction"]
        action = routes.get(label) or routes.get(str(label)) or (label if label in tools else (step.get("default_tool") or tools[0]))
    else:
        action = tools[0]
    result: Any = None
    if action == "retrieve":
        knowledge_id = step.get("knowledge")
        if not knowledge_id:
            raise StudioError(f"step {step['id']!r}: agent retrieve requires knowledge")
        result = search(store, knowledge_id, query, k=int(step.get("k") or 3))
    elif action in BUILTIN_TOOLS:
        result = BUILTIN_TOOLS[action]({"value": query}, state)
    elif action in {"finish", "done"}:
        result = None
    else:
        raise StudioError(f"step {step['id']!r}: unknown agent action {action!r}")
    return {"label": label, "action": action, "query": query, "result": result}


def process_mermaid(spec: dict[str, Any]) -> str:
    """Render a process as a Mermaid flowchart (LangChain/Dify-style DAG view)."""
    lines = ["flowchart TD"]
    steps = spec.get("steps") or []
    if not steps:
        lines.append("  empty[empty process]")
        return "\n".join(lines)
    prev = None
    for step in steps:
        node_id = str(step["id"]).replace("-", "_")
        label = f"{step['id']} / {step['kind']}"
        lines.append(f'  {node_id}["{label}"]')
        if prev:
            lines.append(f"  {prev} --> {node_id}")
        if step["kind"] == "switch":
            for case, nested in (step.get("cases") or {}).items():
                case_slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(case)).strip("_") or "case"
                case_id = f"{node_id}_{case_slug}"
                lines.append(f'  {case_id}["{case}"]')
                lines.append(f"  {node_id} --> {case_id}")
                nested_prev = case_id
                for nested_step in nested:
                    nid = f"{node_id}_{str(nested_step['id']).replace('-', '_')}"
                    lines.append(f'  {nid}["{nested_step["id"]} / {nested_step["kind"]}"]')
                    lines.append(f"  {nested_prev} --> {nid}")
                    nested_prev = nid
        if step["kind"] == "subprocess":
            lines.append(f'  {node_id}_p["process {step.get("process")}"]')
            lines.append(f"  {node_id} --> {node_id}_p")
        prev = node_id
    return "\n".join(lines)


def run_process_on_dataset(
    store: StudioStore,
    process_id: str,
    dataset_id: str,
    *,
    input_field: str = "input",
    expected_field: str = "expected",
    limit: int | None = None,
) -> dict[str, Any]:
    from evalloop.studio.data import load_dataset

    _meta, rows = load_dataset(store, dataset_id)
    if limit is not None:
        rows = rows[:limit]
    spec = load_process(store, process_id)
    outputs_decl = spec.get("outputs") or []
    primary = None
    if outputs_decl:
        first = outputs_decl[0]
        primary = first["name"] if isinstance(first, dict) else str(first)
    results = []
    n_correct = 0
    n_scored = 0
    for row in rows:
        inputs = dict(row)
        if input_field in row and "input" not in inputs:
            inputs["input"] = row[input_field]
        elif input_field != "input":
            inputs["input"] = row.get(input_field, row.get("input"))
        run = run_process(store, process_id, inputs, record=False)
        predicted = run["outputs"].get(primary) if primary else None
        if predicted is None:
            predicted = run["outputs"].get("label") or run["outputs"].get("prediction") or run["outputs"].get("answer")
        item: dict[str, Any] = {"id": row.get("id"), "outputs": run["outputs"], "prediction": predicted}
        if expected_field in row and row[expected_field] is not None:
            n_scored += 1
            correct = str(predicted) == str(row[expected_field])
            item["expected"] = row[expected_field]
            item["correct"] = correct
            if correct:
                n_correct += 1
        results.append(item)
    summary = {
        "process_id": process_id,
        "dataset_id": dataset_id,
        "n": len(results),
        "n_scored": n_scored,
        "n_correct": n_correct,
        "accuracy": round(n_correct / n_scored, 4) if n_scored else None,
        "results": results,
        "finished_at": utcnow(),
    }
    run_id = new_run_id("batch").lower()
    summary["run_id"] = run_id
    atomic_write_json(store.paths.run_file(run_id), summary)
    store.put(
        "runs",
        run_id,
        {
            "kind": "batch_run",
            "process_id": process_id,
            "dataset_id": dataset_id,
            "n": len(results),
            "accuracy": summary["accuracy"],
        },
    )
    return summary
