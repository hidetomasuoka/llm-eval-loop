"""cuad100 golden.jsonl を chenghao/cuad_qa ミラーから決定的に再生成するスクリプト。

データポリシー（issue #47）により golden.jsonl 本体は git 管理外だが、
このスクリプトと manifest.json（case id → データセットレコード id の対応表。
契約書本文は含まない）を追跡することで、誰でも同一バイト列を再生成できる。

使い方（datasets はプロジェクト依存に含めていないため --with で注入する）:

    uv run --with datasets python tasks/cuad100/scripts/build_golden.py build
    uv run --with datasets python tasks/cuad100/scripts/build_golden.py build --check
    uv run --with datasets python tasks/cuad100/scripts/build_golden.py select-extras

- `build`: manifest.json の各エントリをデータセットから引いて golden.jsonl を書き出す。
  `--check` は書き出さずに既存ファイルとのバイト一致を検証する
- `select-extras`: 2026-07-19 の拡張分（train ネガティブ10 + dev 40）を seed 固定で
  選定し manifest.json に追記する一度きりの手順。既に extras があれば拒否する
  （`--force` で再選定）。選定結果は manifest が正であり、このコマンドを回し直す
  必要は通常ない

レコード仕様（PROVENANCE.md / docs/DESIGN.md §5.1 参照）:
- input  = "[{title}]\\n\\n{context}\\n\\nSource: {source}\\n\\n[質問]\\n{question}"
- expected = answers.text を "; " 結合。ネガティブケースは「該当条項なし」
  （optimizers/metrics.py の NO_CLAUSE_ANSWER と正規化一致する文字列）

ネガティブケースについて: chenghao/cuad_qa ミラーは回答可能ペアのみを含む
（answers 空のレコードは 0 件）。そのため select-extras は公式 CUAD
（theatticusproject/cuad-qa の parquet 変換ブランチ、answers 空 = gold「該当なし」）で
(契約, カテゴリ) が本当に未該当であることを照合してから選定する。build 時の契約書
本文はミラーの同一契約コンテキスト（契約内で全レコード同一であることを確認済み）を
使うため、再生成にはミラーだけがあればよい。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
GOLDEN_PATH = TASK_DIR / "golden.jsonl"
MANIFEST_PATH = SCRIPT_DIR / "manifest.json"

DATASET_NAME = "chenghao/cuad_qa"
# 公式CUAD（未該当ペアを含む）。script形式のため parquet 変換ブランチを直接読む
OFFICIAL_DATA_FILES = {
    "train": "hf://datasets/theatticusproject/cuad-qa@refs/convert/parquet/default/train/*.parquet",
    "test": "hf://datasets/theatticusproject/cuad-qa@refs/convert/parquet/default/test/*.parquet",
}
ANSWER_SEP = "; "
NO_CLAUSE_ANSWER = "該当条項なし"
SOURCE_CREDIT = "CUAD v1 (The Atticus Project, CC BY 4.0) via chenghao/cuad_qa mirror on Hugging Face"

# 拡張分（case-0101..）の選定パラメータ。既存100件には適用されない。
SELECT_SEED = 42
MAX_CONTEXT_CHARS = 150_000  # 実行時間対策: 極端に長い契約書は新規選定から除外
N_TRAIN_NEGATIVES = 10
N_DEV_POSITIVES = 32
N_DEV_NEGATIVES = 8


def load_dataset_index():
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME)
    by_id = {}
    by_title = {}
    for split in ds:
        for rec in ds[split]:
            by_id[rec["id"]] = rec
            # コンテキストは契約内で全レコード同一（検証済み）。決定性のため id 最小の代表を保持
            cur = by_title.get(rec["title"])
            if cur is None or rec["id"] < cur["id"]:
                by_title[rec["title"]] = rec
    return by_id, by_title


def make_case(case_id: str, split: str, title: str, context: str, source: str, category: str, expected: str) -> dict:
    input_text = f"[{title}]\n\n{context}\n\nSource: {source}\n\n[質問]\n{category}"
    return {
        "id": case_id,
        "input": input_text,
        "expected": expected,
        "split": split,
        "meta": {
            "category": category,
            "difficulty": "normal",
            "source": SOURCE_CREDIT,
        },
    }


def record_to_case(case_id: str, split: str, rec: dict) -> dict:
    texts = rec["answers"]["text"]
    expected = ANSWER_SEP.join(texts) if texts else NO_CLAUSE_ANSWER
    return make_case(case_id, split, rec["title"], rec["context"], rec["source"], rec["question"], expected)


def build(check: bool) -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    by_id, by_title = load_dataset_index()
    lines = []
    for entry in manifest["cases"]:
        if entry.get("negative"):
            rep = by_title.get(entry["title"])
            if rep is None:
                print(f"ERROR: contract not found in mirror: {entry['title']}", file=sys.stderr)
                return 1
            case = make_case(
                entry["case"],
                entry["split"],
                entry["title"],
                rep["context"],
                rep["source"],
                entry["category"],
                NO_CLAUSE_ANSWER,
            )
        else:
            rec = by_id.get(entry["dataset_id"])
            if rec is None:
                print(f"ERROR: dataset record not found: {entry['dataset_id']}", file=sys.stderr)
                return 1
            case = record_to_case(entry["case"], entry["split"], rec)
        lines.append(json.dumps(case, ensure_ascii=False))
    content = "\n".join(lines) + "\n"

    if check:
        current = GOLDEN_PATH.read_text(encoding="utf-8") if GOLDEN_PATH.exists() else ""
        if current == content:
            print(f"OK: golden.jsonl is byte-identical to manifest rebuild ({len(lines)} cases)")
            return 0
        cur_lines = current.splitlines()
        for i, line in enumerate(lines):
            if i >= len(cur_lines) or cur_lines[i] != line:
                print(f"MISMATCH at line {i + 1} (case {manifest['cases'][i]['case']})")
                return 1
        print(f"MISMATCH: existing file has {len(cur_lines)} lines, rebuild has {len(lines)}")
        return 1

    GOLDEN_PATH.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    splits = Counter(e["split"] for e in manifest["cases"])
    print(f"wrote {GOLDEN_PATH} ({len(lines)} cases, splits={dict(splits)})")
    print(f"sha256: {digest}")
    return 0


def _quota_by_composition(counts: Counter, total: int) -> dict[str, int]:
    """カテゴリ構成 counts に比例した合計 total の割当（決定的、大きい端数優先）。"""
    grand = sum(counts.values())
    frac = {c: total * n / grand for c, n in counts.items()}
    quota = {c: int(f) for c, f in frac.items()}
    remainder = total - sum(quota.values())
    order = sorted(counts, key=lambda c: (-(frac[c] - quota[c]), c))
    for c in order[:remainder]:
        quota[c] += 1
    return {c: q for c, q in quota.items() if q > 0}


def _pick(pool_by_cat, quota, used_titles, rng, want_negative):
    """カテゴリ割当に従い、未使用契約から1契約1件で決定的に選ぶ。"""
    picked = []
    shortfall = 0
    for cat in sorted(quota):
        need = quota[cat]
        for rec in pool_by_cat.get(cat, []):
            if need == 0:
                break
            if rec["title"] in used_titles:
                continue
            picked.append(rec)
            used_titles.add(rec["title"])
            need -= 1
        shortfall += need
    if shortfall:
        # 候補不足カテゴリの分は残り全候補から補充（rng順、契約重複なし）
        rest = [r for cat in sorted(pool_by_cat) for r in pool_by_cat[cat]]
        rng.shuffle(rest)
        for rec in rest:
            if shortfall == 0:
                break
            if rec["title"] in used_titles:
                continue
            picked.append(rec)
            used_titles.add(rec["title"])
            shortfall -= 1
    if shortfall:
        raise RuntimeError(f"candidate pool exhausted (negative={want_negative})")
    return picked


def select_extras(force: bool) -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    existing = [e for e in manifest["cases"] if int(e["case"].split("-")[1]) <= 100]
    extras = [e for e in manifest["cases"] if int(e["case"].split("-")[1]) > 100]
    if extras and not force:
        print(f"extras already present ({len(extras)} cases); use --force to reselect")
        return 1

    from datasets import load_dataset

    by_id, by_title = load_dataset_index()
    existing_recs = [by_id[e["dataset_id"]] for e in existing]
    existing_ids = {e["dataset_id"] for e in existing}
    used_titles = {r["title"] for r in existing_recs}

    train_cats = Counter(r["question"] for e, r in zip(existing, existing_recs) if e["split"] == "train")
    test_cats = Counter(r["question"] for e, r in zip(existing, existing_recs) if e["split"] == "test")
    known_cats = set(train_cats) | set(test_cats)

    rng = random.Random(SELECT_SEED)

    # ポジティブ候補: ミラー（回答可能ペアのみ）から
    pos_by_cat: dict[str, list] = defaultdict(list)
    for rec_id in sorted(by_id):
        rec = by_id[rec_id]
        if rec_id in existing_ids or rec["question"] not in known_cats:
            continue
        if len(rec["context"]) > MAX_CONTEXT_CHARS:
            continue
        pos_by_cat[rec["question"]].append(rec)

    # ネガティブ候補: 公式CUADの answers 空ペア（gold「該当なし」）を照合して採用。
    # ミラーに同じ (契約, カテゴリ) が存在しないことも整合性チェックとして要求する
    mirror_pairs = {(r["title"], r["question"]) for r in by_id.values()}
    official = load_dataset("parquet", data_files=OFFICIAL_DATA_FILES)
    neg_by_cat: dict[str, list] = defaultdict(list)
    seen_neg_pairs = set()
    official_empty = []
    for sp in official:
        for rec in official[sp]:
            if not rec["answers"]["text"]:
                official_empty.append(rec)
    for rec in sorted(official_empty, key=lambda r: r["id"]):
        title = rec["title"]
        category = rec["id"][len(title) + 2 : rec["id"].rfind("_")]
        if category not in known_cats or title not in by_title:
            continue
        if (title, category) in mirror_pairs or (title, category) in seen_neg_pairs:
            continue
        if len(by_title[title]["context"]) > MAX_CONTEXT_CHARS:
            continue
        seen_neg_pairs.add((title, category))
        neg_by_cat[category].append({"title": title, "question": category})
    for cat_pool in (pos_by_cat, neg_by_cat):
        for recs in cat_pool.values():
            rng.shuffle(recs)

    train_neg = _pick(neg_by_cat, _quota_by_composition(train_cats, N_TRAIN_NEGATIVES), used_titles, rng, True)
    dev_pos = _pick(pos_by_cat, _quota_by_composition(test_cats, N_DEV_POSITIVES), used_titles, rng, False)
    dev_neg = _pick(neg_by_cat, _quota_by_composition(test_cats, N_DEV_NEGATIVES), used_titles, rng, True)

    new_entries = []
    next_num = 101
    for split, recs in (("train", train_neg), ("dev", dev_pos), ("dev", dev_neg)):
        for rec in recs:
            if "id" in rec:
                entry = {"case": f"case-{next_num:04d}", "dataset_id": rec["id"], "split": split}
            else:
                entry = {
                    "case": f"case-{next_num:04d}",
                    "split": split,
                    "negative": True,
                    "title": rec["title"],
                    "category": rec["question"],
                }
            new_entries.append(entry)
            next_num += 1

    manifest["cases"] = existing + new_entries
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(
        f"selected extras: train_neg={len(train_neg)} dev_pos={len(dev_pos)} "
        f"dev_neg={len(dev_neg)} -> manifest total {len(manifest['cases'])}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser("build", help="manifest.json から golden.jsonl を再生成")
    p_build.add_argument("--check", action="store_true", help="書き出さずバイト一致を検証")
    p_sel = sub.add_parser("select-extras", help="拡張分（case-0101..）を選定し manifest に追記")
    p_sel.add_argument("--force", action="store_true", help="既存 extras を破棄して再選定")
    args = parser.parse_args()
    if args.cmd == "build":
        return build(check=args.check)
    return select_extras(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
