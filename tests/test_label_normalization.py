"""Pins the two label-normalization implementations to one shared case table:

    src/evalloop/asserts/label_match.js   normalizeLabel  (promptfoo's final grading)
    src/evalloop/optimize.py              _normalize_label (GEPA's in-process metric)

If they drift, GEPA optimizes the prompt against a different verdict than the
one promptfoo ultimately grades with, corrupting before/after comparisons.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from evalloop import build as build_mod
from evalloop.optimize import _normalize_label

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_PATH = FIXTURES_DIR / "label_normalization_cases.json"
JS_RUNNER_PATH = FIXTURES_DIR / "js_normalize_runner.js"

CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_python_normalization_matches_fixture(case):
    assert _normalize_label(case["input"]) == case["normalized"]


def test_python_normalization_non_string_returns_empty():
    assert _normalize_label(None) == ""
    assert _normalize_label(123) == ""
    assert _normalize_label(["解約可能"]) == ""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_js_normalization_matches_fixture():
    proc = subprocess.run(
        ["node", str(JS_RUNNER_PATH), str(FIXTURE_PATH), str(build_mod.LABEL_MATCH_JS)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert proc.returncode == 0, f"node runner failed:\n{proc.stderr}"
    js_results = json.loads(proc.stdout)
    expected = {c["name"]: c["normalized"] for c in CASES}
    assert js_results == expected
