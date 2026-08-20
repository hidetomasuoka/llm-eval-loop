"""Dataset import, profiling, and train/test split (DataRobot-like data layer)."""

from __future__ import annotations

import csv
import io
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from evalloop.studio.errors import StudioError
from evalloop.studio.ids import validate_id
from evalloop.studio.store import StudioStore, atomic_write_json, atomic_write_yaml

MAX_SAMPLE_VALUES = 8
MAX_PROFILE_ROWS = 20000


def load_tabular(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path.read_text(encoding="utf-8"))
    if suffix == ".jsonl":
        return _load_jsonl(path.read_text(encoding="utf-8"))
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            columns = _union_columns(payload)
            return columns, [{c: row.get(c) for c in columns} for row in payload]
        raise StudioError(f"{path}: JSON dataset must be a list of objects")
    raise StudioError(f"{path}: unsupported dataset format (use .csv / .jsonl / .json)")


def load_tabular_text(text: str, fmt: str) -> tuple[list[str], list[dict[str, Any]]]:
    if fmt == "csv":
        return _load_csv(text)
    if fmt == "jsonl":
        return _load_jsonl(text)
    raise StudioError(f"unsupported inline format {fmt!r}")


def _load_csv(text: str) -> tuple[list[str], list[dict[str, Any]]]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise StudioError("CSV dataset has no header row")
    columns = [c for c in reader.fieldnames if c]
    rows = []
    for raw in reader:
        rows.append({c: _coerce_cell(raw.get(c, "")) for c in columns})
    return columns, rows


def _load_jsonl(text: str) -> tuple[list[str], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise StudioError(f"jsonl line {lineno}: {e}") from e
        if not isinstance(row, dict):
            raise StudioError(f"jsonl line {lineno}: expected an object")
        records.append(row)
    if not records:
        raise StudioError("jsonl dataset is empty")
    columns = _union_columns(records)
    return columns, [{c: row.get(c) for c in columns} for row in records]


def _union_columns(rows: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def _coerce_cell(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in text or "e" in lowered:
            return float(text)
        return int(text)
    except ValueError:
        return text


def infer_column_type(values: list[Any]) -> str:
    present = [v for v in values if v is not None and v != ""]
    if not present:
        return "empty"
    if all(isinstance(v, bool) for v in present):
        return "boolean"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        return "numeric"
    if all(isinstance(v, (dict, list)) for v in present):
        return "json"
    nunique = len({_norm_key(v) for v in present})
    avg_len = sum(len(str(v)) for v in present) / len(present)
    if avg_len >= 40 or nunique > max(20, len(present) * 0.5):
        return "text"
    return "categorical"


def _norm_key(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def profile_rows(columns: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample = rows[:MAX_PROFILE_ROWS]
    col_profiles = []
    for col in columns:
        values = [row.get(col) for row in sample]
        present = [v for v in values if v is not None and v != ""]
        missing = len(values) - len(present)
        kind = infer_column_type(values)
        counts = Counter(_norm_key(v) for v in present)
        entry: dict[str, Any] = {
            "name": col,
            "type": kind,
            "missing": missing,
            "missing_rate": round(missing / len(values), 4) if values else 0.0,
            "nunique": len(counts),
        }
        if kind == "numeric" and present:
            nums = [float(v) for v in present]
            entry["min"] = min(nums)
            entry["max"] = max(nums)
            entry["mean"] = round(sum(nums) / len(nums), 6)
        else:
            entry["top_values"] = [{"value": k, "count": n} for k, n in counts.most_common(MAX_SAMPLE_VALUES)]
        col_profiles.append(entry)
    return {
        "n_rows": len(rows),
        "n_columns": len(columns),
        "columns": col_profiles,
        "quality": dataset_quality(columns, sample, target=_guess_target(columns)),
    }


def dataset_quality(columns: list[str], rows: list[dict[str, Any]], target: str | None = None) -> dict[str, Any]:
    """DataRobot-like data quality flags (leakage, constants, missingness)."""
    flags: list[dict[str, Any]] = []
    n = len(rows) or 1
    target_vals = [_norm_key(row.get(target)) for row in rows] if target and target in columns else None
    for col in columns:
        if col == target:
            continue
        values = [row.get(col) for row in rows]
        present = [v for v in values if v is not None and v != ""]
        missing_rate = 1.0 - (len(present) / n)
        if missing_rate >= 0.4:
            flags.append({"code": "high_missing", "column": col, "missing_rate": round(missing_rate, 4)})
        nunique = len({_norm_key(v) for v in present})
        if present and nunique == 1:
            flags.append({"code": "constant", "column": col, "value": _norm_key(present[0])})
        if target_vals and present and len(present) == len(rows):
            feat_vals = [_norm_key(v) for v in values]
            if feat_vals == target_vals:
                flags.append({"code": "target_leakage", "column": col, "detail": "identical to target"})
            elif nunique == len({t for t in target_vals}) == len(rows):
                flags.append({"code": "id_like", "column": col, "detail": "unique per row; may leak identity"})
    return {"n_flags": len(flags), "flags": flags}


def split_rows(
    rows: list[dict[str, Any]],
    target: str | None = None,
    test_ratio: float = 0.25,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if any(row.get("split") in {"train", "test"} for row in rows):
        train = [r for r in rows if r.get("split") == "train"]
        test = [r for r in rows if r.get("split") == "test"]
        if train and test:
            return train, test
    if not 0.0 < test_ratio < 1.0:
        raise StudioError("test_ratio must be between 0 and 1 exclusive")
    indexed = list(enumerate(rows))
    rng = random.Random(seed)
    if target:
        buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for item in indexed:
            key = _norm_key(item[1].get(target))
            buckets.setdefault(key, []).append(item)
        train: list[dict[str, Any]] = []
        test: list[dict[str, Any]] = []
        for group in buckets.values():
            rng.shuffle(group)
            n_test = max(1, round(len(group) * test_ratio)) if len(group) > 1 else 0
            if n_test >= len(group):
                n_test = len(group) - 1
            test.extend(row for _, row in group[:n_test])
            train.extend(row for _, row in group[n_test:])
        if not test and train:
            test.append(train.pop())
        return train, test
    rng.shuffle(indexed)
    n_test = max(1, round(len(indexed) * test_ratio)) if len(indexed) > 1 else 0
    test = [row for _, row in indexed[:n_test]]
    train = [row for _, row in indexed[n_test:]]
    return train, test


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def import_dataset(
    store: StudioStore,
    dataset_id: str,
    source: Path | None = None,
    *,
    rows: list[dict[str, Any]] | None = None,
    columns: list[str] | None = None,
    origin: str = "file",
    description: str = "",
) -> dict[str, Any]:
    validate_id(dataset_id, "dataset")
    if rows is None:
        if source is None:
            raise StudioError("import_dataset requires source= or rows=")
        columns, rows = load_tabular(Path(source))
    elif columns is None:
        columns = _union_columns(rows)
    if not rows:
        raise StudioError("dataset is empty")
    profile = profile_rows(columns, rows)
    dataset_dir = store.paths.dataset_dir(dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_path = dataset_dir / "data.jsonl"
    write_jsonl(data_path, rows)
    meta = {
        "kind": "dataset",
        "description": description,
        "origin": origin,
        "source": str(source) if source is not None else None,
        "n_rows": len(rows),
        "columns": columns,
        "path": str(data_path),
    }
    atomic_write_yaml(dataset_dir / "meta.yaml", meta)
    atomic_write_json(dataset_dir / "profile.json", profile)
    return store.put(
        "datasets",
        dataset_id,
        {
            "kind": "dataset",
            "description": description,
            "origin": origin,
            "n_rows": len(rows),
            "n_columns": len(columns),
            "columns": columns,
            "target_hint": _guess_target(columns),
        },
    )


def _guess_target(columns: list[str]) -> str | None:
    for name in ("expected", "label", "target", "y", "churn", "class"):
        if name in columns:
            return name
    return columns[-1] if columns else None


def load_dataset(store: StudioStore, dataset_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = store.get("datasets", dataset_id)
    data_path = store.paths.dataset_dir(dataset_id) / "data.jsonl"
    if not data_path.exists():
        raise StudioError(f"dataset {dataset_id!r} is missing {data_path}")
    return meta, read_jsonl(data_path)


def load_profile(store: StudioStore, dataset_id: str) -> dict[str, Any]:
    store.get("datasets", dataset_id)
    path = store.paths.dataset_dir(dataset_id) / "profile.json"
    if not path.exists():
        _, rows = load_dataset(store, dataset_id)
        columns = _union_columns(rows)
        profile = profile_rows(columns, rows)
        atomic_write_json(path, profile)
        return profile
    return json.loads(path.read_text(encoding="utf-8"))


def import_from_task(store: StudioStore, task_name: str, dataset_id: str | None = None, repo_root: Path | None = None) -> dict[str, Any]:
    from evalloop.paths import REPO_ROOT, for_task
    from evalloop.schemas import load_golden_jsonl

    root = repo_root or REPO_ROOT
    paths = for_task(task_name, root)
    cases = load_golden_jsonl(paths.golden)
    rows = []
    for case in cases:
        expected = case.expected
        if isinstance(expected, (dict, list)):
            expected = json.dumps(expected, ensure_ascii=False)
        rows.append(
            {
                "id": case.id,
                "input": case.input,
                "expected": expected,
                "split": case.split,
                "category": case.category,
                "difficulty": case.difficulty,
                "source": case.source,
            }
        )
    entity_id = dataset_id or task_name
    return import_dataset(
        store,
        entity_id,
        rows=rows,
        columns=["id", "input", "expected", "split", "category", "difficulty", "source"],
        origin=f"task:{task_name}",
        description=f"Imported from evalloop task {task_name}",
    )
