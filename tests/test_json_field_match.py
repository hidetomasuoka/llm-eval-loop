"""Pins the two JSON deep-equality implementations to one shared case table:

    src/evalloop/asserts/json_field_match.js   (promptfoo's final grading)
    src/evalloop/optimize.py                   json_score_and_feedback (GEPA metric)

If they drift, GEPA optimizes against a different verdict than the one
promptfoo ultimately grades with -- same failure mode the label-normalization
fixture guards against.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from evalloop import build as build_mod
from evalloop.optimize import json_score_and_feedback

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_PATH = FIXTURES_DIR / "json_field_match_cases.json"
JS_RUNNER_PATH = FIXTURES_DIR / "js_json_field_match_runner.js"

CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_python_json_metric_matches_fixture(case):
    score, feedback = json_score_and_feedback(case["output"], case["expected"])
    assert (score == 1.0) is case["pass"], feedback


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_js_json_assert_matches_fixture():
    proc = subprocess.run(
        ["node", str(JS_RUNNER_PATH), str(FIXTURE_PATH), str(build_mod.JSON_FIELD_MATCH_JS)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert proc.returncode == 0, f"node runner failed:\n{proc.stderr}"
    js_results = json.loads(proc.stdout)
    expected = {c["name"]: c["pass"] for c in CASES}
    assert js_results == expected
