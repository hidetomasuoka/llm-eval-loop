import json

import pytest

from evalloop.schemas import SchemaError, parse_promptfoo_output

NESTED_SAMPLE = {
    "evalId": "eval-test-1",
    "results": {
        "version": 3,
        "prompts": [{"raw": "hello {{input}}", "label": "task"}],
        "results": [
            {
                "vars": {"case_id": "case-0001", "input": "x", "expected": "契約照会", "category": "基本"},
                "provider": {"id": "anthropic:messages:claude-haiku-4-5-20251001", "label": "haiku45"},
                "response": {"output": "契約照会", "tokenUsage": {"total": 10, "prompt": 8, "completion": 2}, "cached": False},
                "gradingResult": {"pass": True, "score": 1, "reason": "normalized output matches"},
                "success": True,
                "score": 1,
                "cost": 0.0001,
                "latencyMs": 250,
            },
            {
                "vars": {"case_id": "case-0002", "input": "y", "expected": "障害報告", "category": "基本"},
                "provider": {"id": "ollama:chat:qwen2.5:7b", "label": "qwen7b"},
                "response": {"output": "その他", "tokenUsage": {"total": 12}, "cached": True},
                "gradingResult": {"pass": False, "score": 0, "reason": "mismatch"},
                "success": False,
                "score": 0,
                "cost": 0.0,
                "latencyMs": 90,
            },
        ],
    },
}

FLAT_SAMPLE = {
    "evalId": "eval-test-2",
    "results": [
        {
            "vars": {"case_id": "case-0003", "expected": "その他", "category": "基本"},
            "provider": "haiku45",
            "response": {"output": "その他"},
            "gradingResult": {"pass": True, "score": 1, "reason": "ok"},
            "success": True,
            "cost": 0.00005,
            "latencyMs": 100,
        }
    ],
}


def test_parse_nested_layout(tmp_path):
    p = tmp_path / "output.json"
    p.write_text(json.dumps(NESTED_SAMPLE), encoding="utf-8")

    parsed = parse_promptfoo_output(p)

    assert parsed.eval_id == "eval-test-1"
    assert len(parsed.results) == 2
    r0 = parsed.results[0]
    assert r0.case_id == "case-0001"
    assert r0.alias == "haiku45"
    assert r0.passed is True
    assert r0.output == "契約照会"
    assert r0.cost == pytest.approx(0.0001)
    assert r0.latency_ms == 250
    assert r0.cached is False

    r1 = parsed.results[1]
    assert r1.alias == "qwen7b"
    assert r1.passed is False
    assert r1.cached is True
    assert parsed.warnings == []


def test_parse_flat_fallback_layout_warns_but_succeeds(tmp_path):
    p = tmp_path / "output.json"
    p.write_text(json.dumps(FLAT_SAMPLE), encoding="utf-8")

    parsed = parse_promptfoo_output(p)

    assert len(parsed.results) == 1
    assert parsed.results[0].alias == "haiku45"
    assert any("flat list" in w for w in parsed.warnings)


def test_parse_missing_results_array_returns_empty_with_warning(tmp_path):
    p = tmp_path / "output.json"
    p.write_text(json.dumps({"evalId": "x", "somethingElse": {}}), encoding="utf-8")

    parsed = parse_promptfoo_output(p)

    assert parsed.results == []
    assert len(parsed.warnings) == 1


def test_parse_missing_file_raises(tmp_path):
    with pytest.raises(SchemaError):
        parse_promptfoo_output(tmp_path / "does-not-exist.json")


def test_parse_row_missing_case_id_warns_but_keeps_row(tmp_path):
    sample = {
        "results": {
            "results": [
                {
                    "vars": {"expected": "その他"},
                    "provider": {"id": "p", "label": "alias"},
                    "response": {"output": "その他"},
                    "gradingResult": {"pass": True, "score": 1},
                }
            ]
        }
    }
    p = tmp_path / "output.json"
    p.write_text(json.dumps(sample), encoding="utf-8")

    parsed = parse_promptfoo_output(p)

    assert len(parsed.results) == 1
    assert parsed.results[0].case_id is None
    assert any("case_id" in w for w in parsed.warnings)


def test_token_usage_never_falls_back_to_judge_tokens(tmp_path):
    # gradingResult.tokensUsed is the llm-rubric JUDGE's consumption; when a
    # provider omits response.tokenUsage, the model-side usage must stay empty
    # instead of silently absorbing the judge's numbers (issue #85)
    sample = {
        "results": {
            "results": [
                {
                    "vars": {"case_id": "case-0001", "expected": "x", "category": "基本"},
                    "provider": {"id": "ollama:chat:qwen2.5:7b", "label": "qwen7b"},
                    "response": {"output": "x"},  # no tokenUsage
                    "gradingResult": {"pass": True, "score": 1, "tokensUsed": {"prompt": 100, "completion": 20}},
                    "success": True,
                }
            ]
        }
    }
    p = tmp_path / "output.json"
    p.write_text(json.dumps(sample, ensure_ascii=False), encoding="utf-8")

    parsed = parse_promptfoo_output(p)

    assert parsed.results[0].token_usage == {}


def test_repeat_index_counts_per_case_and_provider(tmp_path):
    # a multi-provider run interleaves providers for the same case; the repeat
    # counter must be per (case, alias) or one provider's repeats would spread
    # across indices and break report.py's per-repeat aggregation
    rows = []
    for _repeat in range(2):
        for label in ("haiku45", "qwen7b"):
            rows.append(
                {
                    "vars": {"case_id": "case-0001", "expected": "x", "category": "基本"},
                    "provider": {"id": label, "label": label},
                    "response": {"output": "x"},
                    "gradingResult": {"pass": True, "score": 1},
                    "success": True,
                }
            )
    p = tmp_path / "output.json"
    p.write_text(json.dumps({"results": {"results": rows}}, ensure_ascii=False), encoding="utf-8")

    parsed = parse_promptfoo_output(p)

    seq = [(r.alias, r.repeat_index) for r in parsed.results]
    assert seq == [("haiku45", 0), ("qwen7b", 0), ("haiku45", 1), ("qwen7b", 1)]
