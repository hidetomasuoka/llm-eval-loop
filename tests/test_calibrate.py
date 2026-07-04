import json

import pytest
import yaml

from evalloop import build as build_mod
from evalloop import calibrate as calibrate_mod
from evalloop import run as run_mod

REPO_ROOT = build_mod.REPO_ROOT


def _write_golden(path, cases):
    with path.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def _write_human_labels(path, labels):
    with path.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")


@pytest.fixture
def calibrate_env(tmp_path, monkeypatch):
    golden_path = tmp_path / "golden.jsonl"
    human_labels_path = tmp_path / "human_labels.jsonl"
    runs_dir = tmp_path / "runs"

    _write_golden(
        golden_path,
        [
            {"id": "case-0001", "input": "x", "expected": "契約照会", "split": "test", "meta": {"category": "基本", "source": "self-made"}},
            {"id": "case-0002", "input": "y", "expected": "障害報告", "split": "test", "meta": {"category": "基本", "source": "self-made"}},
            {"id": "case-0003", "input": "z", "expected": "機能要望", "split": "test", "meta": {"category": "基本", "source": "self-made"}},
        ],
    )

    monkeypatch.setattr(build_mod, "GOLDEN_PATH", golden_path)
    monkeypatch.setattr(calibrate_mod, "HUMAN_LABELS_PATH", human_labels_path)
    monkeypatch.setattr(run_mod, "RUNS_DIR", runs_dir)

    return {"golden_path": golden_path, "human_labels_path": human_labels_path, "runs_dir": runs_dir}


def _write_run_output(runs_dir, run_id, rows):
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "output.json").write_text(json.dumps({"results": {"results": rows}}), encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps({"judge": {"provider": "j", "calibration_status": "uncalibrated", "agreement_rate": None}}),
        encoding="utf-8",
    )


def _row(case_id, alias, passed):
    return {
        "vars": {"case_id": case_id, "expected": "x", "category": "基本"},
        "provider": {"id": "p", "label": alias},
        "response": {"output": "o"},
        "gradingResult": {"pass": passed, "score": 1 if passed else 0, "reason": "r"},
        "success": passed,
    }


def test_calibrate_run_id_mode_high_agreement(calibrate_env):
    _write_human_labels(
        calibrate_env["human_labels_path"],
        [
            {"case_id": "case-0001", "model_label": "haiku45", "output_raw": "契約照会", "human_verdict": "pass"},
            {"case_id": "case-0002", "model_label": "haiku45", "output_raw": "障害報告", "human_verdict": "pass"},
            {"case_id": "case-0003", "model_label": "haiku45", "output_raw": "その他", "human_verdict": "fail"},
        ],
    )
    _write_run_output(
        calibrate_env["runs_dir"],
        "run-1",
        [
            _row("case-0001", "haiku45", True),
            _row("case-0002", "haiku45", True),
            _row("case-0003", "haiku45", False),
        ],
    )

    result = calibrate_mod.calibrate(config_path=REPO_ROOT / "config.yaml", run_id="run-1")

    assert result.n_compared == 3
    assert result.agreement_rate == pytest.approx(1.0)
    assert result.status == "calibrated"

    meta = json.loads((calibrate_env["runs_dir"] / "run-1" / "meta.json").read_text(encoding="utf-8"))
    assert meta["judge"]["calibration_status"] == "calibrated"
    assert meta["judge"]["agreement_rate"] == pytest.approx(1.0)


def test_calibrate_run_id_mode_low_agreement_warns(calibrate_env):
    _write_human_labels(
        calibrate_env["human_labels_path"],
        [
            {"case_id": "case-0001", "model_label": "haiku45", "output_raw": "契約照会", "human_verdict": "pass"},
            {"case_id": "case-0002", "model_label": "haiku45", "output_raw": "障害報告", "human_verdict": "fail"},
            {"case_id": "case-0003", "model_label": "haiku45", "output_raw": "その他", "human_verdict": "fail"},
        ],
    )
    # judge disagrees with human on 2 of 3
    _write_run_output(
        calibrate_env["runs_dir"],
        "run-2",
        [
            _row("case-0001", "haiku45", True),
            _row("case-0002", "haiku45", True),
            _row("case-0003", "haiku45", True),
        ],
    )

    result = calibrate_mod.calibrate(config_path=REPO_ROOT / "config.yaml", run_id="run-2")

    assert result.agreement_rate == pytest.approx(1 / 3)
    assert result.status == "low_agreement"

    meta = json.loads((calibrate_env["runs_dir"] / "run-2" / "meta.json").read_text(encoding="utf-8"))
    assert meta["judge"]["calibration_status"] == "low_agreement"


def test_calibrate_skips_case_missing_from_golden(calibrate_env):
    _write_human_labels(
        calibrate_env["human_labels_path"],
        [{"case_id": "case-9999", "model_label": "haiku45", "output_raw": "x", "human_verdict": "pass"}],
    )
    _write_run_output(calibrate_env["runs_dir"], "run-3", [])

    result = calibrate_mod.calibrate(config_path=REPO_ROOT / "config.yaml", run_id="run-3")

    assert result.n_compared == 0
    assert result.n_skipped == 1
    assert result.status == "no_data"


def test_calibrate_skips_case_missing_judge_verdict(calibrate_env):
    _write_human_labels(
        calibrate_env["human_labels_path"],
        [
            {"case_id": "case-0001", "model_label": "haiku45", "output_raw": "契約照会", "human_verdict": "pass"},
            {"case_id": "case-0002", "model_label": "haiku45", "output_raw": "障害報告", "human_verdict": "pass"},
        ],
    )
    # only case-0001 has a matching judge verdict in the run
    _write_run_output(calibrate_env["runs_dir"], "run-4", [_row("case-0001", "haiku45", True)])

    result = calibrate_mod.calibrate(config_path=REPO_ROOT / "config.yaml", run_id="run-4")

    assert result.n_compared == 1
    assert result.n_skipped == 1


def test_calibrate_missing_run_raises(calibrate_env):
    _write_human_labels(
        calibrate_env["human_labels_path"],
        [{"case_id": "case-0001", "model_label": "haiku45", "output_raw": "x", "human_verdict": "pass"}],
    )
    with pytest.raises(calibrate_mod.CalibrateError):
        calibrate_mod.calibrate(config_path=REPO_ROOT / "config.yaml", run_id="does-not-exist")


def test_calibrate_empty_human_labels_raises(calibrate_env):
    calibrate_env["human_labels_path"].write_text("", encoding="utf-8")
    with pytest.raises(calibrate_mod.CalibrateError):
        calibrate_mod.calibrate(config_path=REPO_ROOT / "config.yaml", run_id="whatever")


def test_judge_verdicts_from_run_majority_vote_across_repeats(calibrate_env):
    _write_run_output(
        calibrate_env["runs_dir"],
        "run-5",
        [
            _row("case-0001", "haiku45", True),
            _row("case-0001", "haiku45", True),
            _row("case-0001", "haiku45", False),
        ],
    )
    verdicts = calibrate_mod._judge_verdicts_from_run("run-5")
    assert verdicts[("case-0001", "haiku45")] is True


def test_fresh_judge_config_uses_echo_provider_and_pinned_judge(calibrate_env, monkeypatch):
    """Structural check of the 'no run_id' path: build.py's iron-rule-#2 judge
    pin must flow into the throwaway echo-replay config, without ever calling
    a real API (run_promptfoo_eval is monkeypatched out).
    """
    _write_human_labels(
        calibrate_env["human_labels_path"],
        [{"case_id": "case-0001", "model_label": "haiku45", "output_raw": "契約照会", "human_verdict": "pass"}],
    )

    captured = {}

    def fake_eval(config_path, output_path, **kwargs):
        captured["config"] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "results": {
                        "results": [
                            {
                                "vars": {"case_id": "case-0001"},
                                "provider": {"id": "echo", "label": "echo"},
                                "response": {"output": "契約照会"},
                                "gradingResult": {"pass": True, "score": 1},
                                "success": True,
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

        class _P:
            returncode = 0
            stderr = ""

        return _P()

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)

    result = calibrate_mod.calibrate(config_path=REPO_ROOT / "config.yaml", run_id=None)

    assert captured["config"]["providers"] == [{"id": "echo", "label": "echo"}]
    assert captured["config"]["prompts"] == ["{{output_raw}}"]
    rubric_assert = captured["config"]["defaultTest"]["assert"][0]
    assert rubric_assert["type"] == "llm-rubric"
    assert rubric_assert["provider"] == "anthropic:messages:claude-sonnet-4-6"
    assert result.n_compared == 1
    assert result.agreement_rate == pytest.approx(1.0)
    # the throwaway config must not be left behind
    assert not list(build_mod.PROMPTFOO_DIR.glob("_calibrate_*.yaml"))
