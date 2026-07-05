# cuad100 — データ出自と再取得手順

このタスクのデータ（`golden.jsonl` / `human_labels.jsonl`）は **git 管理外**（issue #47 の
データポリシー: タスクデータは既定でコミット禁止）。.gitignore の
`tasks/*/golden.jsonl` / `tasks/*/human_labels.jsonl` パターンで除外されている
（`sample-inquiry` だけが否定パターンで opt-in）。

## 出典

- **CUAD v1** (Contract Understanding Atticus Dataset) — The Atticus Project 発行。
  - 配布ページ: https://www.atticusprojectai.org/cuad
  - **License**: CC BY 4.0（原著作者クレジットを保持すれば再配布・改変・商用利用可）
- 取得元は Hugging Face の `chenghao/cuad_qa` ミラー（Hugging Face Hub 上の
  `datasets` 互換フォーマット）。本リポジトリは CUAD の公式配布物ではなく、
  同ミラー経由で取得した QA ペアを元に作成した派生サブセット。
- **Retrieved**: 2026-07-04

## タスク概要

CUAD の **100件サブセット**（train 20 / test 80）。契約書抜粋＋条項カテゴリの
質問に対し、該当箇所を原文どおり抜き出す抽出型QA。
`task.yaml` は `answer_type: text` ＋ `llm-rubric` ジャッジで評価する。

## サンプリング方法

- `chenghao/cuad_qa` から **41種の条項カテゴリ** を横断的にカバーするよう
  100件を選択。カテゴリ一覧は CUAD v1 の 41 個のラベルそのままで、
  本タスクのサンプルに含まれるのは以下の 41 カテゴリ（昇順）:
  Affiliate License-Licensee, Affiliate License-Licensor, Agreement Date,
  Anti-Assignment, Audit Rights, Cap On Liability, Change Of Control,
  Competitive Restriction Exception, Covenant Not To Sue, Document Name,
  Effective Date, Exclusivity, Expiration Date, Governing Law, Insurance,
  Ip Ownership Assignment, Irrevocable Or Perpetual License, Joint Ip
  Ownership, License Grant, Liquidated Damages, Minimum Commitment,
  Most Favored Nation, No-Solicit Of Customers, No-Solicit Of Employees,
  Non-Compete, Non-Disparagement, Non-Transferable License, Notice Period
  To Terminate Renewal, Parties, Post-Termination Services, Price
  Restrictions, Renewal Term, Revenue/Profit Sharing, Rofr/Rofo/Rofn,
  Source Code Escrow, Termination For Convenience, Third Party Beneficiary,
  Uncapped Liability, Unlimited/All-You-Can-Eat-License, Volume
  Restriction, Warranty Duration
- 各カテゴリ 1〜3 件を含む（カテゴリ別件数は均等でない）。
- **注意**: train 20件がカバーするのは 17 カテゴリのみ。test 80件は 40 カテゴリを
  カバーし、train と test でカテゴリ構成は完全には一致しない
  （`Parties` は train にのみ存在; `Source Code Escrow` は test にのみ存在し件数最少1件）。
  層化抽出ではなく簡易サンプリングなので、カテゴリ別精度評価には適さない。
- `id` は `case-0001` .. `case-0100`（100件連番）。
- 抽出スクリプトは未整備（後続課題）。

## フィールド構成

各レコードのフィールド:

- `id` — `case-NNNN` 形式の連番
- `input` — `[契約書名]` 行、`[契約書抜粋]` 本文、`[質問]` 行の3段落構成。
  本文は長すぎる場合 `...(前略)...` / `...(後略)...` で中略し、CUAD の `Source:`
  行（例: `Source: PCQUOTE COM INC, S-1/A, 7/21/1999`）はそのまま残す
- `expected` — 該当条項の原文抜粋。複数ある場合はセミコロン(`;`)で区切って
  全て列挙。該当なしは未観測（本タスクの100件はすべて該当あり）
- `split` — `train` / `test`
- `meta.category` — 上記41種のいずれか
- `meta.difficulty` — 全件 `normal`（CUAD には難易度軸がないため固定値）
- `meta.source` — `"CUAD v1 (The Atticus Project, CC BY 4.0) via
  chenghao/cuad_qa mirror on Hugging Face"`。`task.yaml` の
  `blog.allowed_sources` と一致する値

## ファイル指紋（検証用）

- `golden.jsonl`:
  - sha256: `cc67ad2abd8ad31b0e96dbd0b320a1a01faf1733cf8bf4d28b9c2246bfb69d80`
  - サイズ: 357245 bytes, 100行（末尾改行あり）
  - 検証: `shasum -a 256 tasks/cuad100/golden.jsonl`
- `human_labels.jsonl`: 意図的に空（CUAD-100への人手レビュー未実施）

## 再取得

`chenghao/cuad_qa` から抽出（抽出スクリプトは未整備 — スクリプト化は後続課題。
上記 sha256 と一致すれば同一データ）。

```bash
python -c "from datasets import load_dataset; ds=load_dataset('chenghao/cuad_qa')"
```

その後、上記サンプリング方法・フィールド構成に従って `golden.jsonl` を構築。
CC BY 4.0 のため出典クレジットを保持すれば再配布可だが、本リポジトリの
データポリシーに従い git にはコミットしない。

## ライセンス上の注意

CUAD は CC BY 4.0 で、`meta.source` にクレジットを明記済み。契約書抜粋をそのまま
`golden.jsonl` に格納しているのは再現性確保のため。ブログ記事化の際は
`task.yaml` の `blog.allowed_sources` が `meta.source` の値をホワイトリスト
検査するので、許可された出典文字列以外はビルドガードで弾かれる。