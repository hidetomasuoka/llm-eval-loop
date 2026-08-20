"""Restricted expression evaluator for process ``expr`` steps."""

from __future__ import annotations

import ast
from typing import Any

from evalloop.studio.errors import StudioError

_ALLOWED_FUNCS = {
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
    "sorted": sorted,
    "bool": bool,
}

_ALLOWED_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)
_ALLOWED_UNARY = (ast.UAdd, ast.USub, ast.Not)
_ALLOWED_CMP = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot)


def safe_eval(expr: str, state: dict[str, Any]) -> Any:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise StudioError(f"invalid expression {expr!r}: {e}") from e
    return _eval(tree.body, state)


def _eval(node: ast.AST, state: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in state:
            return state[node.id]
        if node.id in {"True", "true"}:
            return True
        if node.id in {"False", "false"}:
            return False
        if node.id in {"None", "none"}:
            return None
        if node.id in _ALLOWED_FUNCS:
            return _ALLOWED_FUNCS[node.id]
        raise StudioError(f"unknown name {node.id!r} in expression")
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        return _binop(node.op, _eval(node.left, state), _eval(node.right, state))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
        value = _eval(node.operand, state)
        if isinstance(node.op, ast.Not):
            return not value
        if isinstance(node.op, ast.USub):
            return -value
        return +value
    if isinstance(node, ast.BoolOp):
        values = [_eval(v, state) for v in node.values]
        if isinstance(node.op, ast.And):
            result: Any = True
            for item in values:
                result = result and item
                if not result:
                    return result
            return result
        result = False
        for item in values:
            result = result or item
            if result:
                return result
        return result
    if isinstance(node, ast.Compare):
        left = _eval(node.left, state)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            if not isinstance(op, _ALLOWED_CMP):
                raise StudioError("disallowed comparison")
            right = _eval(comparator, state)
            if not _compare(op, left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return _eval(node.body, state) if _eval(node.test, state) else _eval(node.orelse, state)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise StudioError("only a small allowlist of functions can be called in expressions")
        if node.keywords:
            raise StudioError("keyword arguments are not allowed in expressions")
        func = _ALLOWED_FUNCS[node.func.id]
        args = [_eval(arg, state) for arg in node.args]
        return func(*args)
    if isinstance(node, ast.Subscript):
        value = _eval(node.value, state)
        sl = node.slice
        key = _eval(sl, state)
        return value[key]
    if isinstance(node, ast.List):
        return [_eval(elt, state) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval(elt, state) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return {_eval(k, state): _eval(v, state) for k, v in zip(node.keys, node.values, strict=True) if k is not None}
    if isinstance(node, ast.JoinedStr):  # f-strings
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append(str(_eval(value.value, state)))
            else:
                raise StudioError("unsupported f-string fragment")
        return "".join(parts)
    raise StudioError(f"disallowed expression node {type(node).__name__}")


def _binop(op: ast.operator, left: Any, right: Any) -> Any:
    if isinstance(op, ast.Add):
        return left + right
    if isinstance(op, ast.Sub):
        return left - right
    if isinstance(op, ast.Mult):
        return left * right
    if isinstance(op, ast.Div):
        return left / right
    if isinstance(op, ast.FloorDiv):
        return left // right
    if isinstance(op, ast.Mod):
        return left % right
    if isinstance(op, ast.Pow):
        return left**right
    raise StudioError("disallowed operator")


def _compare(op: ast.cmpop, left: Any, right: Any) -> bool:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    if isinstance(op, ast.In):
        return left in right
    if isinstance(op, ast.NotIn):
        return left not in right
    if isinstance(op, ast.Is):
        return left is right
    if isinstance(op, ast.IsNot):
        return left is not right
    raise StudioError("disallowed comparison")
