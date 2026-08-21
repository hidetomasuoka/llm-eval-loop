"""Load and structurally validate ``process.yaml`` (Dify-inspired DAG)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

NODE_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
VALID_NODE_TYPES = ("start", "llm", "template", "if-else", "end")


class ProcessError(ValueError):
    """Invalid process.yaml or illegal graph execution state."""


@dataclass(frozen=True)
class IfCase:
    when: str
    goto: str


@dataclass(frozen=True)
class ProcessNode:
    id: str
    type: str
    prompt_file: str | None = None
    template_file: str | None = None
    template: str | None = None
    output: str | None = None
    cases: tuple[IfCase, ...] = ()
    else_goto: str | None = None
    assemble: dict[str, str] | None = None


@dataclass(frozen=True)
class ProcessEdge:
    frm: str
    to: str


@dataclass(frozen=True)
class ProcessGraph:
    version: int
    nodes: tuple[ProcessNode, ...]
    edges: tuple[ProcessEdge, ...]
    source_path: Path

    def node_map(self) -> dict[str, ProcessNode]:
        return {n.id: n for n in self.nodes}

    def start(self) -> ProcessNode:
        starts = [n for n in self.nodes if n.type == "start"]
        return starts[0]

    def end(self) -> ProcessNode:
        ends = [n for n in self.nodes if n.type == "end"]
        return ends[0]

    def llm_nodes(self) -> tuple[ProcessNode, ...]:
        return tuple(n for n in self.nodes if n.type == "llm")


def _require_id(raw_id: object, *, where: str) -> str:
    if not isinstance(raw_id, str) or not NODE_ID_RE.match(raw_id):
        raise ProcessError(f"{where}: node id must match {NODE_ID_RE.pattern}, got {raw_id!r}")
    return raw_id


def _require_var(name: object, *, where: str) -> str:
    if not isinstance(name, str) or not VAR_NAME_RE.match(name):
        raise ProcessError(f"{where}: variable name must match {VAR_NAME_RE.pattern}, got {name!r}")
    return name


def _parse_node(raw: object, index: int) -> ProcessNode:
    where = f"process.nodes[{index}]"
    if not isinstance(raw, dict):
        raise ProcessError(f"{where} must be a mapping")
    node_id = _require_id(raw.get("id"), where=where)
    node_type = raw.get("type")
    if node_type not in VALID_NODE_TYPES:
        raise ProcessError(f"{where} ({node_id}): type must be one of {VALID_NODE_TYPES}, got {node_type!r}")

    prompt_file = raw.get("prompt_file")
    template_file = raw.get("template_file")
    template = raw.get("template")
    output = raw.get("output")
    assemble = raw.get("assemble")
    cases_raw = raw.get("cases")
    else_goto = raw.get("else")

    if node_type == "start":
        extra = {k for k in raw if k not in {"id", "type"}}
        if extra:
            raise ProcessError(f"{where} ({node_id}): start nodes take no extra fields, got {sorted(extra)}")
        return ProcessNode(id=node_id, type=node_type)

    if node_type == "llm":
        if not isinstance(prompt_file, str) or not prompt_file.strip():
            raise ProcessError(f"{where} ({node_id}): llm nodes require prompt_file")
        return ProcessNode(
            id=node_id,
            type=node_type,
            prompt_file=prompt_file,
            output=_require_var(output, where=f"{where} ({node_id}).output"),
        )

    if node_type == "template":
        has_file = isinstance(template_file, str) and bool(template_file.strip())
        has_inline = isinstance(template, str) and bool(template)
        if has_file == has_inline:
            raise ProcessError(f"{where} ({node_id}): template nodes require exactly one of template_file or template")
        return ProcessNode(
            id=node_id,
            type=node_type,
            template_file=template_file if has_file else None,
            template=template if has_inline else None,
            output=_require_var(output, where=f"{where} ({node_id}).output"),
        )

    if node_type == "if-else":
        if not isinstance(cases_raw, list) or not cases_raw:
            raise ProcessError(f"{where} ({node_id}): if-else nodes require a non-empty cases list")
        cases: list[IfCase] = []
        for ci, case in enumerate(cases_raw):
            cwhere = f"{where} ({node_id}).cases[{ci}]"
            if not isinstance(case, dict):
                raise ProcessError(f"{cwhere} must be a mapping with when/goto")
            when = case.get("when")
            goto = case.get("goto")
            if not isinstance(when, str) or not when.strip():
                raise ProcessError(f"{cwhere}: when must be a non-empty string")
            cases.append(IfCase(when=when.strip(), goto=_require_id(goto, where=f"{cwhere}.goto")))
        if not isinstance(else_goto, str) or not else_goto.strip():
            raise ProcessError(f"{where} ({node_id}): if-else nodes require else: <node id>")
        return ProcessNode(
            id=node_id,
            type=node_type,
            cases=tuple(cases),
            else_goto=_require_id(else_goto, where=f"{where} ({node_id}).else"),
        )

    # end
    has_output = output is not None
    has_assemble = assemble is not None
    if has_output == has_assemble:
        raise ProcessError(f"{where} ({node_id}): end nodes require exactly one of output or assemble")
    parsed_assemble: dict[str, str] | None = None
    if has_assemble:
        if not isinstance(assemble, dict) or not assemble:
            raise ProcessError(f"{where} ({node_id}): assemble must be a non-empty mapping of output_key: var")
        parsed_assemble = {}
        for key, var in assemble.items():
            if not isinstance(key, str) or not VAR_NAME_RE.match(key):
                raise ProcessError(f"{where} ({node_id}).assemble: bad output key {key!r}")
            parsed_assemble[key] = _require_var(var, where=f"{where} ({node_id}).assemble.{key}")
        return ProcessNode(id=node_id, type=node_type, assemble=parsed_assemble)
    return ProcessNode(
        id=node_id,
        type=node_type,
        output=_require_var(output, where=f"{where} ({node_id}).output"),
    )


def _parse_edge(raw: object, index: int) -> ProcessEdge:
    where = f"process.edges[{index}]"
    if not isinstance(raw, dict):
        raise ProcessError(f"{where} must be a mapping with from/to")
    return ProcessEdge(
        frm=_require_id(raw.get("from"), where=f"{where}.from"),
        to=_require_id(raw.get("to"), where=f"{where}.to"),
    )


def load_process_graph(path: str | Path) -> ProcessGraph:
    """Parse process.yaml. Graph-topology checks live in ``validate_graph``."""
    path = Path(path)
    if not path.exists():
        raise ProcessError(f"process file not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ProcessError(f"{path}: top level must be a mapping")
    proc = raw.get("process", raw)
    if not isinstance(proc, dict):
        raise ProcessError(f"{path}: process: must be a mapping")

    version = proc.get("version", 1)
    if version != 1:
        raise ProcessError(f"{path}: unsupported process.version {version!r} (expected 1)")

    nodes_raw = proc.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise ProcessError(f"{path}: process.nodes must be a non-empty list")
    nodes = tuple(_parse_node(item, i) for i, item in enumerate(nodes_raw))
    ids = [n.id for n in nodes]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ProcessError(f"{path}: duplicate node id(s): {dupes}")

    edges_raw = proc.get("edges") or []
    if not isinstance(edges_raw, list):
        raise ProcessError(f"{path}: process.edges must be a list")
    edges = tuple(_parse_edge(item, i) for i, item in enumerate(edges_raw))
    return ProcessGraph(version=1, nodes=nodes, edges=edges, source_path=path)
