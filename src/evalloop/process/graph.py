"""DAG topology for process.yaml: successors, validation, mermaid export."""

from __future__ import annotations

from collections import deque

from evalloop.process.schema import ProcessError, ProcessGraph, ProcessNode


def successors(graph: ProcessGraph) -> dict[str, list[str]]:
    """Outgoing node ids. if-else routing uses cases/else, not edges."""
    succ: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    by_id = graph.node_map()
    for edge in graph.edges:
        if edge.frm not in by_id:
            raise ProcessError(f"{graph.source_path}: edge from unknown node {edge.frm!r}")
        if edge.to not in by_id:
            raise ProcessError(f"{graph.source_path}: edge to unknown node {edge.to!r}")
        if by_id[edge.frm].type == "if-else":
            raise ProcessError(
                f"{graph.source_path}: if-else node {edge.frm!r} must not appear in edges "
                "(use cases[].goto / else instead)"
            )
        succ[edge.frm].append(edge.to)
    for node in graph.nodes:
        if node.type != "if-else":
            continue
        for case in node.cases:
            if case.goto not in by_id:
                raise ProcessError(f"{graph.source_path}: {node.id} cases.goto unknown node {case.goto!r}")
            succ[node.id].append(case.goto)
        assert node.else_goto is not None
        if node.else_goto not in by_id:
            raise ProcessError(f"{graph.source_path}: {node.id} else unknown node {node.else_goto!r}")
        succ[node.id].append(node.else_goto)
    return succ


def predecessors(succ: dict[str, list[str]]) -> dict[str, list[str]]:
    pred: dict[str, list[str]] = {nid: [] for nid in succ}
    for frm, tos in succ.items():
        for to in tos:
            pred[to].append(frm)
    return pred


def unique_successor(graph: ProcessGraph, node_id: str) -> str:
    succs = successors(graph)[node_id]
    if len(succs) != 1:
        raise ProcessError(f"node {node_id!r} must have exactly one successor, got {succs}")
    return succs[0]


def validate_graph(graph: ProcessGraph) -> None:
    """Raise ProcessError unless the graph is a single-start/single-end DAG."""
    starts = [n for n in graph.nodes if n.type == "start"]
    ends = [n for n in graph.nodes if n.type == "end"]
    if len(starts) != 1:
        raise ProcessError(f"{graph.source_path}: need exactly one start node, got {[n.id for n in starts]}")
    if len(ends) != 1:
        raise ProcessError(f"{graph.source_path}: need exactly one end node, got {[n.id for n in ends]}")

    succ = successors(graph)
    pred = predecessors(succ)
    start = starts[0]
    end = ends[0]

    if pred[start.id]:
        raise ProcessError(f"{graph.source_path}: start node {start.id!r} must have no incoming edges")
    if succ[end.id]:
        raise ProcessError(f"{graph.source_path}: end node {end.id!r} must have no outgoing edges")
    if not pred[end.id]:
        raise ProcessError(f"{graph.source_path}: end node {end.id!r} is unreachable")

    for node in graph.nodes:
        outs = succ[node.id]
        if node.type in {"start", "llm", "template"} and len(outs) != 1:
            raise ProcessError(
                f"{graph.source_path}: {node.type} node {node.id!r} must have exactly one successor, got {outs}"
            )
        if node.type == "if-else" and len(outs) < 2:
            raise ProcessError(f"{graph.source_path}: if-else node {node.id!r} needs at least one case plus else")

    # Kahn cycle check
    indeg = {nid: len(pred[nid]) for nid in succ}
    queue = deque([nid for nid, d in indeg.items() if d == 0])
    seen = 0
    while queue:
        nid = queue.popleft()
        seen += 1
        for nxt in succ[nid]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if seen != len(succ):
        cyclic = sorted(nid for nid, d in indeg.items() if d > 0)
        raise ProcessError(f"{graph.source_path}: cycle detected involving nodes {cyclic}")

    reachable = _walk(start.id, succ)
    missing = sorted(nid for nid in succ if nid not in reachable)
    if missing:
        raise ProcessError(f"{graph.source_path}: node(s) unreachable from start: {missing}")

    rev = {nid: [] for nid in succ}
    for frm, tos in succ.items():
        for to in tos:
            rev[to].append(frm)
    can_end = _walk(end.id, rev)
    stranded = sorted(nid for nid in succ if nid not in can_end)
    if stranded:
        raise ProcessError(f"{graph.source_path}: node(s) cannot reach end: {stranded}")


def _walk(origin: str, adj: dict[str, list[str]]) -> set[str]:
    seen = {origin}
    queue = deque([origin])
    while queue:
        nid = queue.popleft()
        for nxt in adj.get(nid, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def to_mermaid(graph: ProcessGraph) -> str:
    """Render a flowchart. Node ids are prefixed (``end`` is reserved in mermaid)."""
    validate_graph(graph)
    succ = successors(graph)
    lines = ["flowchart TD"]
    for node in graph.nodes:
        lines.append(f"  {_mid(node.id)}{_mermaid_shape(node)}")
    for node in graph.nodes:
        if node.type == "if-else":
            for case in node.cases:
                label = case.when.replace('"', "'")
                lines.append(f'  {_mid(node.id)} -->|"{label}"| {_mid(case.goto)}')
            lines.append(f'  {_mid(node.id)} -->|"else"| {_mid(node.else_goto)}')
            continue
        for nxt in succ[node.id]:
            lines.append(f"  {_mid(node.id)} --> {_mid(nxt)}")
    return "\n".join(lines) + "\n"


def _mid(node_id: str) -> str:
    return f"n_{node_id}"


def _mermaid_shape(node: ProcessNode) -> str:
    title = node.type if node.type != "llm" else f"llm {node.id}"
    if node.type == "if-else":
        return f'{{"{node.id}"}}'
    if node.type == "end":
        return f'(["{node.id}"])'
    if node.type == "start":
        return f'(["{node.id}"])'
    return f'["{title}"]'
