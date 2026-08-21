import json
import types
from pathlib import Path

import pytest
import yaml

from evalloop import analyze as analyze_mod
from evalloop import build as build_mod
from evalloop import optimize as optimize_mod
from evalloop import paths as paths_mod
from evalloop import run as run_mod
from evalloop.process.graph import to_mermaid, validate_graph
from evalloop.process.interp import assemble_end, eval_when, render_template, route_if_else
from evalloop.process.schema import ProcessError, ProcessNode, load_process_graph
from evalloop.schemas import load_task
from tests.conftest import scaffold_task

REPO_ROOT = Path(__file__).resolve().parent.parent

CONTRACT_REPLY = "契約に関するご照会を受け付けました。担当者よりご連絡します。"


def _write_process_files(task_dir: Path, process_yaml: str, prompts: dict[str, str] | None = None) -> None:
    (task_dir / "process.yaml").write_text(process_yaml, encoding="utf-8")
    (task_dir / "prompts").mkdir(parents=True, exist_ok=True)
    for name, text in (prompts or {}).items():
        (task_dir / "prompts" / name).write_text(text, encoding="utf-8")


def _linear_process() -> str:
    return """\
process:
  version: 1
  nodes:
    - id: start
      type: start
    - id: classify
      type: llm
      prompt_file: prompts/classify.txt
      output: label
    - id: end
      type: end
      output: label
  edges:
    - {from: start, to: classify}
    - {from: classify, to: end}
"""


def _branching_process() -> str:
    return """\
process:
  version: 1
  nodes:
    - id: start
      type: start
    - id: classify
      type: llm
      prompt_file: prompts/classify.txt
      output: label
    - id: route
      type: if-else
      cases:
        - when: "label == '契約照会'"
          goto: contract_reply
      else: default_reply
    - id: contract_reply
      type: template
      template_file: prompts/contract_reply.txt
      output: reply
    - id: default_reply
      type: template
      template: "お問い合わせ（{{label}}）を受け付けました。"
      output: reply
    - id: end
      type: end
      assemble:
        label: label
        reply: reply
  edges:
    - {from: start, to: classify}
    - {from: classify, to: route}
    - {from: contract_reply, to: end}
    - {from: default_reply, to: end}
"""


def scaffold_process_task(isolated_root, name="proc1", branching=True, models=None):
    cfg, paths = scaffold_task(
        isolated_root,
        name=name,
        answer_type="json",
        models=models or ["qwen7b"],
        golden_rows=[
            {
                "id": "case-0001",
                "input": "契約書の解約条項について確認したいです。",
                "expected": {"label": "契約照会", "reply": CONTRACT_REPLY},
                "split": "train",
                "meta": {"category": "基本", "source": "self-made"},
            },
            {
                "id": "case-0101",
                "input": "契約書の解約条項について確認したいです。",
                "expected": {"label": "契約照会", "reply": CONTRACT_REPLY},
                "split": "test",
                "meta": {"category": "基本", "source": "self-made"},
            },
            {
                "id": "case-0102",
                "input": "ログインできずエラーになります。",
                "expected": {"label": "障害報告", "reply": "お問い合わせ（障害報告）を受け付けました。"},
                "split": "test",
                "meta": {"category": "基本", "source": "self-made"},
            },
        ],
    )
    yaml_text = _branching_process() if branching else _linear_process()
    _write_process_files(
        paths.task_dir,
        yaml_text,
        {
            "classify.txt": "分類せよ\n{{input}}\n",
            "contract_reply.txt": CONTRACT_REPLY,
        },
    )
    raw = yaml.safe_load(paths.task_config.read_text(encoding="utf-8"))
    raw["task"]["process_file"] = "process.yaml"
    paths.task_config.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return load_task(name, root=isolated_root)


# ---------------------------------------------------------------------------
# schema / graph
# ---------------------------------------------------------------------------


def test_load_and_validate_sample_process():
    graph = load_process_graph(REPO_ROOT / "tasks/sample-process/process.yaml")
    validate_graph(graph)
    assert [n.id for n in graph.nodes] == [
        "start",
        "classify",
        "route",
        "contract_reply",
        "default_reply",
        "end",
    ]
    assert graph.llm_nodes()[0].id == "classify"


def test_cycle_is_rejected(tmp_path):
    path = tmp_path / "process.yaml"
    path.write_text(
        """\
process:
  version: 1
  nodes:
    - {id: start, type: start}
    - {id: a, type: llm, prompt_file: p.txt, output: x}
    - {id: b, type: llm, prompt_file: p.txt, output: y}
    - {id: end, type: end, output: y}
  edges:
    - {from: start, to: a}
    - {from: a, to: b}
    - {from: b, to: a}
    - {from: b, to: end}
""",
        encoding="utf-8",
    )
    graph = load_process_graph(path)
    with pytest.raises(ProcessError, match="cycle"):
        validate_graph(graph)


def test_two_starts_rejected(tmp_path):
    path = tmp_path / "process.yaml"
    path.write_text(
        """\
process:
  version: 1
  nodes:
    - {id: start, type: start}
    - {id: other, type: start}
    - {id: end, type: end, output: input}
  edges:
    - {from: start, to: end}
""",
        encoding="utf-8",
    )
    with pytest.raises(ProcessError, match="exactly one start"):
        validate_graph(load_process_graph(path))


def test_if_else_cannot_be_in_edges(tmp_path):
    path = tmp_path / "process.yaml"
    path.write_text(
        """\
process:
  version: 1
  nodes:
    - {id: start, type: start}
    - id: route
      type: if-else
      cases:
        - {when: "x == 'a'", goto: end}
      else: end
    - {id: end, type: end, output: input}
  edges:
    - {from: start, to: route}
    - {from: route, to: end}
""",
        encoding="utf-8",
    )
    with pytest.raises(ProcessError, match="must not appear in edges"):
        validate_graph(load_process_graph(path))


def test_unknown_goto_rejected(tmp_path):
    path = tmp_path / "process.yaml"
    path.write_text(
        """\
process:
  version: 1
  nodes:
    - {id: start, type: start}
    - id: route
      type: if-else
      cases:
        - {when: "x == 'a'", goto: missing}
      else: end
    - {id: end, type: end, output: input}
  edges:
    - {from: start, to: route}
""",
        encoding="utf-8",
    )
    with pytest.raises(ProcessError, match="unknown node"):
        validate_graph(load_process_graph(path))


def test_mermaid_contains_branch_labels():
    graph = load_process_graph(REPO_ROOT / "tasks/sample-process/process.yaml")
    text = to_mermaid(graph)
    assert text.startswith("flowchart TD")
    assert "n_classify" in text
    assert "n_end" in text
    assert "else" in text
    assert "契約照会" in text


# ---------------------------------------------------------------------------
# interp
# ---------------------------------------------------------------------------


def test_eval_when_operators():
    state = {"label": "契約照会"}
    assert eval_when("label == '契約照会'", state) is True
    assert eval_when('label != "障害報告"', state) is True
    assert eval_when("label contains '契約'", state) is True
    assert eval_when("reply empty", state) is True
    with pytest.raises(ProcessError, match="unsupported"):
        eval_when("label > 1", state)


def test_render_and_assemble():
    assert render_template("hi {{label}}", {"label": "契約照会"}) == "hi 契約照会"
    node = ProcessNode(id="end", type="end", assemble={"label": "label", "reply": "reply"})
    out = assemble_end(node, {"label": "契約照会", "reply": "ok"})
    assert json.loads(out) == {"label": "契約照会", "reply": "ok"}


def test_route_if_else_uses_else():
    from evalloop.process.schema import IfCase

    node = ProcessNode(
        id="route",
        type="if-else",
        cases=(IfCase("label == '契約照会'", "contract_reply"),),
        else_goto="default_reply",
    )
    assert route_if_else(node, {"label": "契約照会"}) == "contract_reply"
    assert route_if_else(node, {"label": "その他"}) == "default_reply"


# ---------------------------------------------------------------------------
# load_task / init / optimize guard
# ---------------------------------------------------------------------------


def test_load_task_reads_process_file():
    cfg, paths = load_task("sample-process", root=REPO_ROOT)
    assert cfg.task.process_file
    assert Path(cfg.task.process_file).name == "process.yaml"
    assert paths.task == "sample-process"


def test_init_kind_process_scaffolds_graph(isolated_root):
    tp = paths_mod.init_task_workspace("p-init", root=isolated_root, answer_type="label", kind="process")
    assert (tp.task_dir / "process.yaml").exists()
    assert (tp.task_dir / "prompts" / "classify.txt").exists()
    assert not tp.prompt_file.exists()
    graph = load_process_graph(tp.task_dir / "process.yaml")
    validate_graph(graph)


def test_optimize_rejects_process_task(isolated_root):
    cfg, paths = scaffold_process_task(isolated_root)
    with pytest.raises(optimize_mod.OptimizeError, match="process tasks cannot be optimized"):
        optimize_mod.optimize(cfg, paths)


def test_build_shuffle_demos_rejected_for_process(isolated_root):
    cfg, paths = scaffold_process_task(isolated_root)
    with pytest.raises(build_mod.BuildError, match="shuffle-demos"):
        build_mod.build(cfg, paths, yes=True, shuffle_demos=2)


# ---------------------------------------------------------------------------
# build + run (promptfoo mocked)
# ---------------------------------------------------------------------------


def _label_for_input(text: str) -> str:
    if "契約" in text:
        return "契約照会"
    if "ログイン" in text or "クラッシュ" in text:
        return "障害報告"
    if "フィルタ" in text or "ダーク" in text:
        return "機能要望"
    return "その他"


def _fake_process_eval(config_path, output_path, **kwargs):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    desc = cfg.get("description") or ""
    tests = cfg.get("tests") or []
    provider = (cfg.get("providers") or [{}])[0]
    alias = provider.get("label") or "qwen7b"
    provider_id = provider.get("id") or "echo"
    rows = []
    for test in tests:
        vars_ = dict(test.get("vars") or {})
        if "echo grade" in desc:
            output = vars_.get("output_raw") or ""
            expected = vars_.get("expected")
            try:
                parsed = json.loads(output) if isinstance(output, str) else output
                passed = parsed == expected
            except json.JSONDecodeError:
                passed = False
            cost = 0.0
        else:
            output = _label_for_input(str(vars_.get("input") or ""))
            passed = True
            cost = 0.01
        rows.append(
            {
                "vars": vars_,
                "provider": {"id": provider_id, "label": alias},
                "response": {"output": output, "tokenUsage": {}},
                "gradingResult": {"pass": passed, "score": 1 if passed else 0, "reason": "ok"},
                "success": passed,
                "cost": cost,
            }
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def test_build_process_writes_node_configs(isolated_root):
    cfg, paths = scaffold_process_task(isolated_root)
    estimate = build_mod.build(cfg, paths, yes=True)
    assert paths.promptfoo_config.exists()
    built = yaml.safe_load(paths.promptfoo_config.read_text(encoding="utf-8"))
    assert built["evalloop_process"] is True
    assert (paths.process_nodes_dir / "classify.yaml").exists()
    assert paths.tests_test.exists()
    assert estimate.per_model_usd["qwen7b"] >= 0


def test_run_process_branches_and_grades(isolated_root, monkeypatch):
    cfg, paths = scaffold_process_task(isolated_root)
    build_mod.build(cfg, paths, yes=True)
    monkeypatch.setattr(run_mod, "get_promptfoo_version", lambda: "0.0.0-test")
    monkeypatch.setattr(run_mod, "get_node_version", lambda: "v22.22.0")
    from evalloop.process.execute import run_process

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", _fake_process_eval)
    outcome = run_process(cfg, paths, eval_fn=_fake_process_eval)
    assert outcome.meta["kind"] == "process"
    parsed = json.loads(outcome.output_path.read_text(encoding="utf-8"))
    rows = parsed["results"]["results"]
    by_case = {r["vars"]["case_id"]: r for r in rows}
    assert by_case["case-0101"]["success"] is True
    assert by_case["case-0102"]["success"] is True
    assert json.loads(by_case["case-0101"]["response"]["output"])["reply"] == CONTRACT_REPLY
    trace = (paths.runs_dir / outcome.run_id / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert any('"node_id": "route"' in line for line in trace)


def test_run_process_via_run_module(isolated_root, monkeypatch):
    cfg, paths = scaffold_process_task(isolated_root)
    build_mod.build(cfg, paths, yes=True)
    monkeypatch.setattr(run_mod, "get_promptfoo_version", lambda: "0.0.0-test")
    monkeypatch.setattr(run_mod, "get_node_version", lambda: "v22.22.0")
    monkeypatch.setattr(run_mod, "run_promptfoo_eval", _fake_process_eval)
    outcome = run_mod.run(cfg, paths)
    assert outcome.meta["kind"] == "process"


def test_failures_include_failed_node(isolated_root, monkeypatch):
    cfg, paths = scaffold_process_task(isolated_root)
    build_mod.build(cfg, paths, yes=True)
    monkeypatch.setattr(run_mod, "get_promptfoo_version", lambda: "0.0.0-test")
    monkeypatch.setattr(run_mod, "get_node_version", lambda: "v22.22.0")

    def wrong_label_eval(config_path, output_path, **kwargs):
        cfg_yaml = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        desc = cfg_yaml.get("description") or ""
        if "echo grade" not in desc:
            # always emit その他 so the contract case takes the default branch and fails
            tests = cfg_yaml.get("tests") or []
            rows = []
            for test in tests:
                vars_ = dict(test.get("vars") or {})
                rows.append(
                    {
                        "vars": vars_,
                        "provider": {"id": "ollama:chat:qwen2.5:7b", "label": "qwen7b"},
                        "response": {"output": "その他", "tokenUsage": {}},
                        "gradingResult": {"pass": True, "score": 1, "reason": ""},
                        "success": True,
                        "cost": 0.0,
                    }
                )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return _fake_process_eval(config_path, output_path, **kwargs)

    from evalloop.process.execute import run_process

    outcome = run_process(cfg, paths, eval_fn=wrong_label_eval)
    failures_path, _notes = analyze_mod.failures(outcome.run_id, paths)
    records = [json.loads(line) for line in failures_path.read_text(encoding="utf-8").splitlines() if line]
    assert records
    assert all("failed_node" in rec for rec in records)


def test_run_process_rejects_variant(isolated_root):
    cfg, paths = scaffold_process_task(isolated_root)
    with pytest.raises(run_mod.RunError, match="--variant"):
        run_mod.run(cfg, paths, variant="x")
