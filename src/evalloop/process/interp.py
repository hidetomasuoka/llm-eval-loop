"""Local (non-LLM) process nodes: template render, if-else conditions, end assemble."""

from __future__ import annotations

import json
import re

from evalloop.process.schema import ProcessError, ProcessNode

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_EQ_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*==\s*(['\"])(.*?)\2$")
_NE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*!=\s*(['\"])(.*?)\2$")
_CONTAINS_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+contains\s+(['\"])(.*?)\2$")
_EMPTY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+empty$")


def stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def render_template(template: str, state: dict[str, str]) -> str:
    """Replace ``{{ var }}`` placeholders. Unknown vars become empty strings."""

    def repl(match: re.Match[str]) -> str:
        return state.get(match.group(1), "")

    return _PLACEHOLDER_RE.sub(repl, template)


def eval_when(expr: str, state: dict[str, str]) -> bool:
    """Tiny condition language: ``var == 'x'``, ``!=``, ``contains``, ``empty``."""
    text = expr.strip()
    match = _EQ_RE.match(text)
    if match:
        return state.get(match.group(1), "") == match.group(3)
    match = _NE_RE.match(text)
    if match:
        return state.get(match.group(1), "") != match.group(3)
    match = _CONTAINS_RE.match(text)
    if match:
        return match.group(3) in state.get(match.group(1), "")
    match = _EMPTY_RE.match(text)
    if match:
        return state.get(match.group(1), "") == ""
    raise ProcessError(
        f"unsupported if-else condition {expr!r}; allowed: var == 'lit', var != 'lit', var contains 'lit', var empty"
    )


def route_if_else(node: ProcessNode, state: dict[str, str]) -> str:
    for case in node.cases:
        if eval_when(case.when, state):
            return case.goto
    assert node.else_goto is not None
    return node.else_goto


def assemble_end(node: ProcessNode, state: dict[str, str]) -> str:
    if node.assemble is not None:
        obj = {key: state.get(var, "") for key, var in node.assemble.items()}
        return json.dumps(obj, ensure_ascii=False)
    assert node.output is not None
    return state.get(node.output, "")
