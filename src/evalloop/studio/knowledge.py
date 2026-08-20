"""Local knowledge bases with TF-IDF retrieval (Dify knowledge analog)."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from evalloop.studio.errors import StudioError
from evalloop.studio.ids import validate_id
from evalloop.studio.store import StudioStore, atomic_write_yaml

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3040-\u30ff\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    parts = _TOKEN_RE.findall((text or "").lower())
    grams: list[str] = []
    for part in parts:
        if len(part) <= 2:
            grams.append(part)
            continue
        grams.append(part)
        grams.extend(part[i : i + 2] for i in range(len(part) - 1))
        if any("\u3040" <= ch <= "\u9fff" for ch in part) and len(part) >= 3:
            grams.extend(part[i : i + 3] for i in range(len(part) - 2))
    return grams


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {tok: (n / total) * idf.get(tok, 0.0) for tok, n in counts.items()}


def _dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(key, 0.0) for key, value in a.items())


def _norm(vec: dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in vec.values())) or 1e-12


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


def import_knowledge(
    store: StudioStore,
    knowledge_id: str,
    documents: list[dict[str, Any]],
    *,
    description: str = "",
) -> dict[str, Any]:
    validate_id(knowledge_id, "knowledge")
    docs: list[dict[str, Any]] = []
    for i, doc in enumerate(documents, start=1):
        if isinstance(doc, str):
            docs.append({"id": f"doc-{i:04d}", "title": "", "text": doc})
            continue
        text = doc.get("text") or doc.get("content") or ""
        if not str(text).strip():
            raise StudioError(f"knowledge {knowledge_id!r}: document {i} has empty text")
        docs.append(
            {
                "id": str(doc.get("id") or f"doc-{i:04d}"),
                "title": str(doc.get("title") or ""),
                "text": str(text),
                "meta": doc.get("meta") or {},
            }
        )
    knowledge_dir = store.paths.knowledge_dir(knowledge_id)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    docs_path = knowledge_dir / "docs.jsonl"
    with docs_path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    meta = {"kind": "knowledge", "description": description, "n_docs": len(docs)}
    atomic_write_yaml(knowledge_dir / "meta.yaml", meta)
    return store.put(
        "knowledge",
        knowledge_id,
        {"kind": "knowledge", "description": description, "n_docs": len(docs)},
    )


def load_documents(store: StudioStore, knowledge_id: str) -> list[dict[str, Any]]:
    store.get("knowledge", knowledge_id)
    path = store.paths.knowledge_dir(knowledge_id) / "docs.jsonl"
    if not path.exists():
        raise StudioError(f"knowledge {knowledge_id!r} is missing {path}")
    docs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def search(store: StudioStore, knowledge_id: str, query: str, k: int = 3) -> list[dict[str, Any]]:
    docs = load_documents(store, knowledge_id)
    tokenized = [tokenize(f"{doc.get('title', '')} {doc.get('text', '')}") for doc in docs]
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    n_docs = max(len(docs), 1)
    idf = {tok: math.log((n_docs + 1) / (count + 1)) + 1.0 for tok, count in df.items()}
    doc_vecs = [_tfidf_vector(tokens, idf) for tokens in tokenized]
    query_vec = _tfidf_vector(tokenize(query), idf)
    scored = []
    for doc, vec in zip(docs, doc_vecs, strict=True):
        scored.append({**doc, "score": round(cosine(query_vec, vec), 6)})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[: max(k, 0)]
