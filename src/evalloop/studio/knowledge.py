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


def _normalize_docs(documents: list[dict[str, Any] | str], knowledge_id: str, start: int = 1) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for i, doc in enumerate(documents, start=start):
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
    return docs


def _write_docs(store: StudioStore, knowledge_id: str, docs: list[dict[str, Any]], description: str) -> dict[str, Any]:
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


def import_knowledge(
    store: StudioStore,
    knowledge_id: str,
    documents: list[dict[str, Any] | str],
    *,
    description: str = "",
) -> dict[str, Any]:
    validate_id(knowledge_id, "knowledge")
    docs = _normalize_docs(documents, knowledge_id)
    return _write_docs(store, knowledge_id, docs, description)


def append_documents(
    store: StudioStore,
    knowledge_id: str,
    documents: list[dict[str, Any] | str],
) -> dict[str, Any]:
    existing = load_documents(store, knowledge_id)
    meta = store.get("knowledge", knowledge_id)
    extra = _normalize_docs(documents, knowledge_id, start=len(existing) + 1)
    return _write_docs(store, knowledge_id, existing + extra, str(meta.get("description") or ""))


def _split_markdown(text: str, fallback_title: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current_title = fallback_title
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            body = "\n".join(current).strip()
            if body:
                chunks.append({"title": current_title, "text": body})
            current_title = line[3:].strip() or fallback_title
            current = []
        else:
            current.append(line)
    body = "\n".join(current).strip()
    if body:
        chunks.append({"title": current_title, "text": body})
    if not chunks and text.strip():
        chunks.append({"title": fallback_title, "text": text.strip()})
    return chunks


def import_knowledge_from_files(
    store: StudioStore,
    knowledge_id: str,
    paths: list[Path],
    *,
    description: str = "",
    append: bool = False,
) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise StudioError(f"knowledge file not found: {path}")
        suffix = path.suffix.lower()
        raw = path.read_text(encoding="utf-8")
        if suffix == ".jsonl":
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    row = json.loads(line)
                    documents.append(row if isinstance(row, dict) else {"text": str(row)})
        elif suffix in {".md", ".txt", ".markdown"}:
            documents.extend(_split_markdown(raw, path.stem))
        else:
            raise StudioError(f"{path}: unsupported knowledge format (use .md / .txt / .jsonl)")
    if append and (store.paths.knowledge_dir(knowledge_id) / "docs.jsonl").exists():
        return append_documents(store, knowledge_id, documents)
    return import_knowledge(store, knowledge_id, documents, description=description or f"Imported from {len(paths)} file(s)")


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
