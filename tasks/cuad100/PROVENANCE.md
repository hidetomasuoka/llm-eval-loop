# cuad100 — データ出自と再取得手順

このタスクのデータ（`golden.jsonl` / `human_labels.jsonl`）は **git 管理外**
（issue #47 のデータポリシー: タスクデータは既定でコミット禁止）。

## 出典

- **CUAD v1** (Contract Understanding Atticus Dataset) — The Atticus Project 発行、
  **CC BY 4.0**。https://www.atticusprojectai.org/cuad
- 取得元は Hugging Face の `chenghao/cuad_qa` ミラー
- 本タスクはそこから抽出した **100件サブセット**（train 20 / test 80）。
  契約書抜粋＋条項カテゴリの質問に対し、該当箇所を原文どおり抜き出す抽出型QA

## ファイル指紋（検証用）

- `golden.jsonl` sha256: `b8eb63ce2c005667934f7cbd6fc14f203e56bee3327659a797df13ff68ad93be`
  （2026-07-05 時点、100件）
- `human_labels.jsonl`: 意図的に空（CUAD-100への人手レビュー未実施）

## 再取得

1. 過去にこのリポジトリを clone していた場合: git 履歴に旧パス `data/golden.jsonl`
   として残っている（`git show daa2b15:data/golden.jsonl` 以降のコミット）
2. 新規に作り直す場合: `chenghao/cuad_qa` から抽出（抽出スクリプトは未整備 —
   スクリプト化は後続課題。上記 sha256 と一致すれば同一データ）
