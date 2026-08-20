"""In-process AutoML (DataRobot analog): profile → candidates → leaderboard → deploy.

No extra ML libraries. Algorithms are small, JSON-serializable, and deterministic
given a seed. Python still does not call LLM providers; this trains tabular/text
classifiers and regressors on studio datasets.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from evalloop.studio.data import infer_column_type, load_dataset, split_rows
from evalloop.studio.errors import StudioError
from evalloop.studio.ids import new_run_id, validate_id
from evalloop.studio.knowledge import tokenize
from evalloop.studio.store import StudioStore, atomic_write_json, atomic_write_yaml

MAX_TEXT_FEATURES = 80
MAX_NUMERIC_SPLITS = 16
MAX_TREE_DEPTH = 6


def _model_id(dataset_id: str, algorithm: str) -> str:
    return f"{dataset_id}-{algorithm.replace('_', '-')}"[:40]


def detect_problem(values: list[Any]) -> str:
    present = [v for v in values if v is not None and v != ""]
    if not present:
        raise StudioError("target column is empty")
    kind = infer_column_type(present)
    if kind == "numeric" and len({_key(v) for v in present}) > min(12, max(4, len(present) // 4)):
        return "regression"
    return "classification"


def _key(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def prepare_frame(
    rows: list[dict[str, Any]],
    target: str,
    feature_columns: list[str] | None = None,
) -> tuple[list[str], dict[str, str], list[dict[str, Any]], list[Any], str]:
    if target not in (rows[0] if rows else {}):
        # still allow if some rows have it
        if not any(target in row for row in rows):
            raise StudioError(f"target column {target!r} not in dataset")
    skip = {target, "split", "id"}
    columns = feature_columns or [c for c in _columns(rows) if c not in skip]
    if not columns:
        raise StudioError("no feature columns left after excluding target/split/id")
    types = {c: infer_column_type([row.get(c) for row in rows]) for c in columns}
    y = [row.get(target) for row in rows]
    problem = detect_problem(y)
    return columns, types, rows, y, problem


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def _text_vocab(rows: list[dict[str, Any]], columns: list[str], types: dict[str, str]) -> list[str]:
    df: Counter[str] = Counter()
    text_cols = [c for c in columns if types[c] == "text"]
    if not text_cols:
        return []
    for row in rows:
        bag: set[str] = set()
        for col in text_cols:
            for tok in tokenize(str(row.get(col) or "")):
                bag.add(f"{col}={tok}")
        df.update(bag)
    return [tok for tok, _ in df.most_common(MAX_TEXT_FEATURES)]


def encode_row(
    row: dict[str, Any],
    columns: list[str],
    types: dict[str, str],
    vocab: list[str],
    cat_levels: dict[str, list[str]],
    num_stats: dict[str, tuple[float, float]],
) -> dict[str, float]:
    features: dict[str, float] = {}
    for col in columns:
        value = row.get(col)
        kind = types[col]
        if kind == "numeric":
            mean, scale = num_stats[col]
            number = mean if _is_missing(value) else float(value)
            features[f"num:{col}"] = (number - mean) / scale
        elif kind == "text":
            tokens = set(f"{col}={tok}" for tok in tokenize(str(value or "")))
            for gram in vocab:
                if gram.startswith(f"{col}=") and gram in tokens:
                    features[f"txt:{gram}"] = 1.0
        else:
            key = _key(value) if not _is_missing(value) else "__missing__"
            levels = cat_levels.get(col, [])
            if key not in levels:
                key = "__other__"
            features[f"cat:{col}={key}"] = 1.0
    return features


def _num_stats(rows: list[dict[str, Any]], columns: list[str], types: dict[str, str]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for col in columns:
        if types[col] != "numeric":
            continue
        nums = [float(row[col]) for row in rows if not _is_missing(row.get(col))]
        if not nums:
            stats[col] = (0.0, 1.0)
            continue
        mean = sum(nums) / len(nums)
        var = sum((x - mean) ** 2 for x in nums) / len(nums)
        stats[col] = (mean, math.sqrt(var) or 1.0)
    return stats


def _cat_levels(rows: list[dict[str, Any]], columns: list[str], types: dict[str, str]) -> dict[str, list[str]]:
    levels: dict[str, list[str]] = {}
    for col in columns:
        if types[col] in {"numeric", "text"}:
            continue
        counts = Counter(_key(row.get(col)) if not _is_missing(row.get(col)) else "__missing__" for row in rows)
        levels[col] = [k for k, _ in counts.most_common(24)]
    return levels


# ---------------------------------------------------------------------------
# algorithms
# ---------------------------------------------------------------------------


def _majority_fit(y: list[Any], problem: str) -> dict[str, Any]:
    if problem == "regression":
        nums = [float(v) for v in y if not _is_missing(v)]
        value = sum(nums) / len(nums) if nums else 0.0
        return {"algorithm": "majority", "problem": problem, "value": value}
    counts = Counter(_key(v) for v in y if not _is_missing(v))
    label = counts.most_common(1)[0][0]
    total = sum(counts.values()) or 1
    probs = {k: n / total for k, n in counts.items()}
    return {"algorithm": "majority", "problem": problem, "label": label, "probs": probs}


def _majority_predict(artifact: dict[str, Any], _features: dict[str, float]) -> tuple[Any, dict[str, float] | None]:
    if artifact["problem"] == "regression":
        return artifact["value"], None
    return artifact["label"], dict(artifact["probs"])


def _nb_fit(
    encoded: list[dict[str, float]],
    y: list[Any],
) -> dict[str, Any]:
    labels = [_key(v) for v in y]
    class_counts = Counter(labels)
    token_counts: dict[str, Counter[str]] = {label: Counter() for label in class_counts}
    token_totals: dict[str, float] = {label: 0.0 for label in class_counts}
    for feats, label in zip(encoded, labels, strict=True):
        for name, value in feats.items():
            if value <= 0:
                continue
            token_counts[label][name] += value
            token_totals[label] += value
    vocab = sorted({name for feats in encoded for name in feats})
    return {
        "algorithm": "naive_bayes",
        "problem": "classification",
        "class_counts": dict(class_counts),
        "token_counts": {label: dict(counter) for label, counter in token_counts.items()},
        "token_totals": token_totals,
        "vocab": vocab,
        "n": len(labels),
    }


def _nb_predict(artifact: dict[str, Any], features: dict[str, float]) -> tuple[Any, dict[str, float]]:
    vocab_size = max(len(artifact["vocab"]), 1)
    scores: dict[str, float] = {}
    for label, count in artifact["class_counts"].items():
        logp = math.log(count / artifact["n"])
        total = artifact["token_totals"].get(label, 0.0) + vocab_size
        counts = artifact["token_counts"].get(label) or {}
        for name, value in features.items():
            if value <= 0:
                continue
            logp += value * math.log((counts.get(name, 0.0) + 1.0) / total)
        scores[label] = logp
    m = max(scores.values())
    exp = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exp.values()) or 1.0
    probs = {k: v / z for k, v in exp.items()}
    label = max(probs, key=probs.get)
    return label, probs


def _gini(labels: list[str]) -> float:
    n = len(labels) or 1
    return 1.0 - sum((c / n) ** 2 for c in Counter(labels).values())


def _mse(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _tree_fit(
    encoded: list[dict[str, float]],
    y: list[Any],
    problem: str,
    max_depth: int = MAX_TREE_DEPTH,
    min_samples: int = 2,
) -> dict[str, Any]:
    labels = [_key(v) for v in y] if problem == "classification" else [float(v) for v in y]
    feature_names = sorted({name for feats in encoded for name in feats})

    def leaf(indexes: list[int]) -> dict[str, Any]:
        subset_y = [labels[i] for i in indexes]
        if problem == "regression":
            value = sum(subset_y) / len(subset_y)
            return {"leaf": True, "value": value}
        counts = Counter(subset_y)
        total = sum(counts.values()) or 1
        return {
            "leaf": True,
            "label": counts.most_common(1)[0][0],
            "probs": {k: n / total for k, n in counts.items()},
        }

    def split(indexes: list[int], depth: int) -> dict[str, Any]:
        if depth >= max_depth or len(indexes) < min_samples * 2:
            return leaf(indexes)
        current_y = [labels[i] for i in indexes]
        if problem == "classification" and len(set(current_y)) == 1:
            return leaf(indexes)
        parent_score = _gini(current_y) if problem == "classification" else _mse(current_y)
        best: tuple[float, str, float, list[int], list[int]] | None = None
        for name in feature_names:
            values = sorted({encoded[i].get(name, 0.0) for i in indexes})
            if len(values) <= 1:
                continue
            cuts = values if len(values) <= MAX_NUMERIC_SPLITS else [
                values[int(i * (len(values) - 1) / (MAX_NUMERIC_SPLITS - 1))] for i in range(MAX_NUMERIC_SPLITS)
            ]
            for threshold in cuts:
                left = [i for i in indexes if encoded[i].get(name, 0.0) <= threshold]
                right = [i for i in indexes if encoded[i].get(name, 0.0) > threshold]
                if len(left) < min_samples or len(right) < min_samples:
                    continue
                if problem == "classification":
                    score = (len(left) * _gini([labels[i] for i in left]) + len(right) * _gini([labels[i] for i in right])) / len(
                        indexes
                    )
                else:
                    score = (len(left) * _mse([labels[i] for i in left]) + len(right) * _mse([labels[i] for i in right])) / len(
                        indexes
                    )
                gain = parent_score - score
                if best is None or gain > best[0]:
                    best = (gain, name, threshold, left, right)
        if best is None or best[0] <= 1e-12:
            return leaf(indexes)
        _, name, threshold, left, right = best
        return {
            "leaf": False,
            "feature": name,
            "threshold": threshold,
            "left": split(left, depth + 1),
            "right": split(right, depth + 1),
        }

    return {
        "algorithm": "decision_tree",
        "problem": problem,
        "tree": split(list(range(len(encoded))), 0),
    }


def _tree_walk(node: dict[str, Any], features: dict[str, float]) -> dict[str, Any]:
    while not node.get("leaf"):
        value = features.get(node["feature"], 0.0)
        node = node["left"] if value <= node["threshold"] else node["right"]
    return node


def _tree_predict(artifact: dict[str, Any], features: dict[str, float]) -> tuple[Any, dict[str, float] | None]:
    leaf = _tree_walk(artifact["tree"], features)
    if artifact["problem"] == "regression":
        return leaf["value"], None
    return leaf["label"], dict(leaf.get("probs") or {leaf["label"]: 1.0})


def _logreg_fit(encoded: list[dict[str, float]], y: list[Any], steps: int = 80, lr: float = 0.4) -> dict[str, Any]:
    classes = sorted({_key(v) for v in y})
    feature_names = sorted({name for feats in encoded for name in feats})
    index = {name: i for i, name in enumerate(feature_names)}
    n_features = len(feature_names)
    n_classes = len(classes)
    weights = [[0.0] * n_features for _ in range(n_classes)]
    bias = [0.0] * n_classes
    y_idx = [classes.index(_key(v)) for v in y]
    for _ in range(steps):
        grad_w = [[0.0] * n_features for _ in range(n_classes)]
        grad_b = [0.0] * n_classes
        for feats, yi in zip(encoded, y_idx, strict=True):
            logits = []
            for c in range(n_classes):
                s = bias[c]
                for name, value in feats.items():
                    s += weights[c][index[name]] * value
                logits.append(s)
            m = max(logits)
            exps = [math.exp(v - m) for v in logits]
            z = sum(exps) or 1.0
            probs = [e / z for e in exps]
            for c in range(n_classes):
                err = probs[c] - (1.0 if c == yi else 0.0)
                grad_b[c] += err
                for name, value in feats.items():
                    grad_w[c][index[name]] += err * value
        n = max(len(encoded), 1)
        for c in range(n_classes):
            bias[c] -= lr * grad_b[c] / n
            for j in range(n_features):
                weights[c][j] -= lr * (grad_w[c][j] / n + 0.01 * weights[c][j])
    return {
        "algorithm": "logreg",
        "problem": "classification",
        "classes": classes,
        "feature_names": feature_names,
        "weights": weights,
        "bias": bias,
    }


def _logreg_predict(artifact: dict[str, Any], features: dict[str, float]) -> tuple[Any, dict[str, float]]:
    index = {name: i for i, name in enumerate(artifact["feature_names"])}
    logits = []
    for c, cls in enumerate(artifact["classes"]):
        s = artifact["bias"][c]
        for name, value in features.items():
            j = index.get(name)
            if j is not None:
                s += artifact["weights"][c][j] * value
        logits.append((cls, s))
    m = max(v for _, v in logits)
    exp = [(cls, math.exp(v - m)) for cls, v in logits]
    z = sum(v for _, v in exp) or 1.0
    probs = {cls: v / z for cls, v in exp}
    label = max(probs, key=probs.get)
    return label, probs


def _linreg_fit(encoded: list[dict[str, float]], y: list[Any], steps: int = 120, lr: float = 0.05) -> dict[str, Any]:
    feature_names = sorted({name for feats in encoded for name in feats})
    index = {name: i for i, name in enumerate(feature_names)}
    weights = [0.0] * len(feature_names)
    bias = 0.0
    targets = [float(v) for v in y]
    for _ in range(steps):
        grad_w = [0.0] * len(feature_names)
        grad_b = 0.0
        for feats, target in zip(encoded, targets, strict=True):
            pred = bias + sum(weights[index[name]] * value for name, value in feats.items() if name in index)
            err = pred - target
            grad_b += err
            for name, value in feats.items():
                if name in index:
                    grad_w[index[name]] += err * value
        n = max(len(encoded), 1)
        bias -= lr * grad_b / n
        for j, g in enumerate(grad_w):
            weights[j] -= lr * (g / n + 0.01 * weights[j])
    return {
        "algorithm": "linreg",
        "problem": "regression",
        "feature_names": feature_names,
        "weights": weights,
        "bias": bias,
    }


def _linreg_predict(artifact: dict[str, Any], features: dict[str, float]) -> tuple[Any, None]:
    index = {name: i for i, name in enumerate(artifact["feature_names"])}
    pred = artifact["bias"]
    for name, value in features.items():
        j = index.get(name)
        if j is not None:
            pred += artifact["weights"][j] * value
    return pred, None


def _knn_fit(encoded: list[dict[str, float]], y: list[Any], problem: str, k: int = 3) -> dict[str, Any]:
    return {
        "algorithm": "knn",
        "problem": problem,
        "k": k,
        "x": encoded,
        "y": [_key(v) if problem == "classification" else float(v) for v in y],
    }


def _knn_predict(artifact: dict[str, Any], features: dict[str, float]) -> tuple[Any, dict[str, float] | None]:
    def dist(other: dict[str, float]) -> float:
        keys = set(features) | set(other)
        return math.sqrt(sum((features.get(k, 0.0) - other.get(k, 0.0)) ** 2 for k in keys))

    ranked = sorted(range(len(artifact["x"])), key=lambda i: dist(artifact["x"][i]))[: artifact["k"]]
    neighbors = [artifact["y"][i] for i in ranked]
    if artifact["problem"] == "regression":
        return sum(neighbors) / len(neighbors), None
    counts = Counter(neighbors)
    total = sum(counts.values()) or 1
    probs = {k: n / total for k, n in counts.items()}
    return counts.most_common(1)[0][0], probs


PREDICTORS = {
    "majority": _majority_predict,
    "naive_bayes": _nb_predict,
    "decision_tree": _tree_predict,
    "logreg": _logreg_predict,
    "linreg": _linreg_predict,
    "knn": _knn_predict,
}


def predict_encoded(model: dict[str, Any], features: dict[str, float]) -> tuple[Any, dict[str, float] | None]:
    algo = model["algorithm"]
    if algo not in PREDICTORS:
        raise StudioError(f"unknown algorithm {algo!r}")
    return PREDICTORS[algo](model, features)


def classification_metrics(y_true: list[Any], y_pred: list[Any]) -> dict[str, Any]:
    t = [_key(v) for v in y_true]
    p = [_key(v) for v in y_pred]
    n = len(t) or 1
    acc = sum(a == b for a, b in zip(t, p, strict=True)) / n
    labels = sorted(set(t) | set(p))
    f1s = []
    per_class = {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(t, p, strict=True))
        fp = sum(a != label and b == label for a, b in zip(t, p, strict=True))
        fn = sum(a == label and b != label for a, b in zip(t, p, strict=True))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        per_class[label] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "n": len(t),
        "per_class": per_class,
    }


def regression_metrics(y_true: list[Any], y_pred: list[Any]) -> dict[str, Any]:
    t = [float(v) for v in y_true]
    p = [float(v) for v in y_pred]
    n = len(t) or 1
    mae = sum(abs(a - b) for a, b in zip(t, p, strict=True)) / n
    mse = sum((a - b) ** 2 for a, b in zip(t, p, strict=True)) / n
    mean = sum(t) / n
    var = sum((a - mean) ** 2 for a in t) / n
    r2 = 1.0 - mse / var if var > 1e-12 else 0.0
    return {"mae": round(mae, 4), "rmse": round(math.sqrt(mse), 4), "r2": round(r2, 4), "n": n}


def _score_row(problem: str, metrics: dict[str, Any]) -> float:
    if problem == "classification":
        return float(metrics["accuracy"])
    return -float(metrics["mae"])


def train_job(
    store: StudioStore,
    dataset_id: str,
    target: str,
    *,
    job_id: str | None = None,
    test_ratio: float = 0.25,
    seed: int = 0,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    ds_meta, rows = load_dataset(store, dataset_id)
    columns, types, rows, _y_all, problem = prepare_frame(rows, target, feature_columns)
    train_rows, test_rows = split_rows(rows, target=target, test_ratio=test_ratio, seed=seed)
    if not train_rows or not test_rows:
        raise StudioError("need both train and test rows to run AutoML")
    vocab = _text_vocab(train_rows, columns, types)
    cat_levels = _cat_levels(train_rows, columns, types)
    num_stats = _num_stats(train_rows, columns, types)
    encoder = {
        "columns": columns,
        "types": types,
        "vocab": vocab,
        "cat_levels": cat_levels,
        "num_stats": {k: list(v) for k, v in num_stats.items()},
        "target": target,
        "problem": problem,
    }

    def encode_all(subset: list[dict[str, Any]]) -> list[dict[str, float]]:
        return [encode_row(row, columns, types, vocab, cat_levels, num_stats) for row in subset]

    x_train = encode_all(train_rows)
    x_test = encode_all(test_rows)
    y_train = [row.get(target) for row in train_rows]
    y_test = [row.get(target) for row in test_rows]

    candidates: list[tuple[str, dict[str, Any]]] = [("majority", _majority_fit(y_train, problem))]
    if problem == "classification":
        candidates.extend(
            [
                ("naive_bayes", _nb_fit(x_train, y_train)),
                ("decision_tree", _tree_fit(x_train, y_train, problem)),
                ("logreg", _logreg_fit(x_train, y_train)),
                ("knn", _knn_fit(x_train, y_train, problem, k=3)),
            ]
        )
    else:
        candidates.extend(
            [
                ("decision_tree", _tree_fit(x_train, y_train, problem)),
                ("linreg", _linreg_fit(x_train, y_train)),
                ("knn", _knn_fit(x_train, y_train, problem, k=3)),
            ]
        )

    leaderboard = []
    fitted: dict[str, dict[str, Any]] = {}
    for name, artifact in candidates:
        fitted[name] = artifact
        preds = [predict_encoded(artifact, feats)[0] for feats in x_test]
        metrics = classification_metrics(y_test, preds) if problem == "classification" else regression_metrics(y_test, preds)
        train_preds = [predict_encoded(artifact, feats)[0] for feats in x_train]
        train_metrics = (
            classification_metrics(y_train, train_preds) if problem == "classification" else regression_metrics(y_train, train_preds)
        )
        row = {
            "model": name,
            "problem": problem,
            "holdout": metrics,
            "train": train_metrics,
            "score": _score_row(problem, metrics),
        }
        leaderboard.append(row)
    leaderboard.sort(key=lambda r: r["score"], reverse=True)
    winner_name = leaderboard[0]["model"]
    job_id = validate_id((job_id or new_run_id("job")).lower()[:40], "job")
    winner_model_id = _model_id(dataset_id, winner_name)
    winner_artifact = {
        "encoder": encoder,
        "model": fitted[winner_name],
        "dataset_id": dataset_id,
        "job_id": job_id,
    }
    job_dir = store.paths.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(job_dir / "leaderboard.json", leaderboard)
    atomic_write_json(job_dir / "winner.json", winner_artifact)
    meta = {
        "kind": "train_job",
        "dataset_id": dataset_id,
        "dataset_origin": ds_meta.get("origin"),
        "target": target,
        "problem": problem,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "winner": winner_name,
        "winner_model_id": winner_model_id,
        "leaderboard": leaderboard,
    }
    atomic_write_yaml(job_dir / "meta.yaml", meta)
    store.put("jobs", job_id, {k: v for k, v in meta.items() if k != "leaderboard"})
    register_trained_model(
        store,
        winner_model_id,
        winner_artifact,
        description=f"AutoML winner ({winner_name}) for {dataset_id}.{target}",
        job_id=job_id,
        metrics=leaderboard[0]["holdout"],
    )
    # keep non-winners too so the leaderboard is actually deployable
    for name, artifact in fitted.items():
        if name == winner_name:
            continue
        mid = _model_id(dataset_id, name)
        register_trained_model(
            store,
            mid,
            {"encoder": encoder, "model": artifact, "dataset_id": dataset_id, "job_id": job_id},
            description=f"AutoML candidate ({name}) for {dataset_id}.{target}",
            job_id=job_id,
            metrics=next(r["holdout"] for r in leaderboard if r["model"] == name),
            deployed=False,
        )
    return store.get("jobs", job_id)


def register_trained_model(
    store: StudioStore,
    model_id: str,
    artifact: dict[str, Any],
    *,
    description: str,
    job_id: str,
    metrics: dict[str, Any],
    deployed: bool = True,
) -> dict[str, Any]:
    validate_id(model_id, "model")
    model_dir = store.paths.model_dir(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(model_dir / "artifact.json", artifact)
    meta = {
        "kind": "trained",
        "algorithm": artifact["model"]["algorithm"],
        "problem": artifact["model"]["problem"],
        "dataset_id": artifact.get("dataset_id"),
        "job_id": job_id,
        "description": description,
        "metrics": metrics,
        "deployed": deployed,
    }
    atomic_write_yaml(model_dir / "meta.yaml", meta)
    return store.put("models", model_id, meta)


def load_trained_model(store: StudioStore, model_id: str) -> dict[str, Any]:
    meta = store.get("models", model_id)
    if meta.get("kind") != "trained":
        raise StudioError(f"model {model_id!r} is {meta.get('kind')!r}, not a trained AutoML model")
    path = store.paths.model_dir(model_id) / "artifact.json"
    if not path.exists():
        raise StudioError(f"model {model_id!r} is missing artifact.json")
    return json.loads(path.read_text(encoding="utf-8"))


def json_read(path: Any) -> dict[str, Any]:
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict_row(store: StudioStore, model_id: str, row: dict[str, Any]) -> dict[str, Any]:
    artifact = load_trained_model(store, model_id)
    encoder = artifact["encoder"]
    num_stats = {k: (float(v[0]), float(v[1])) for k, v in encoder["num_stats"].items()}
    features = encode_row(
        row,
        encoder["columns"],
        encoder["types"],
        encoder["vocab"],
        encoder["cat_levels"],
        num_stats,
    )
    pred, probs = predict_encoded(artifact["model"], features)
    result: dict[str, Any] = {"model_id": model_id, "prediction": pred}
    if probs is not None:
        result["probabilities"] = {k: round(v, 6) for k, v in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)}
    return result


def predict_dataset(store: StudioStore, model_id: str, dataset_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    _meta, rows = load_dataset(store, dataset_id)
    if limit is not None:
        rows = rows[:limit]
    artifact = load_trained_model(store, model_id)
    target = artifact["encoder"]["target"]
    out = []
    for row in rows:
        result = predict_row(store, model_id, row)
        if target in row:
            result["expected"] = row[target]
            result["correct"] = _key(result["prediction"]) == _key(row[target])
        out.append(result)
    return out
