"""Dify-inspired process graphs as an evaluation target.

A process task points ``task.process_file`` at a YAML DAG (start / llm /
template / if-else / end). LLM nodes still go through promptfoo; local nodes
run in Python; the final output is graded by the existing echo + assert path.
"""

from evalloop.process.graph import successors, to_mermaid, validate_graph
from evalloop.process.schema import (
    ProcessEdge,
    ProcessError,
    ProcessGraph,
    ProcessNode,
    load_process_graph,
)

__all__ = [
    "ProcessEdge",
    "ProcessError",
    "ProcessGraph",
    "ProcessNode",
    "load_process_graph",
    "successors",
    "to_mermaid",
    "validate_graph",
]
