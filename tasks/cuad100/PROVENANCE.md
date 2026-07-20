# cuad100 — データ出自と再取得手順

このタスクのデータ（`golden.jsonl` / `human_labels.jsonl`）は **git 管理外**（issue #47 の
データポリシー: タスクデータは既定でコミット禁止）。.gitignore の
`tasks/*/golden.jsonl` / `tasks/*/human_labels.jsonl` パターンで除外されている
（`sample-inquiry` だけが否定パターンで opt-in）。

再生成は `scripts/build_golden.py` + `scripts/manifest.json`（git 追跡。case id →
データセットレコード id の対応表のみで契約書本文は含まない）で**決定的に**行える。

## 出典

- **CUAD v1** (Contract Understanding Atticus Dataset) — The Atticus Project 発行。
  - 配布ページ: https://www.atticusprojectai.org/cuad
  - **License**: CC BY 4.0（原著作者クレジットを保持すれば再配布・改変・商用利用可）
- 取得元は Hugging Face の `chenghao/cuad_qa` ミラー（回答可能な QA ペアのみを含む）。
  本リポジトリは CUAD の公式配布物ではなく、同ミラー経由で取得した QA ペアを元に
  作成した派生サブセット。
- ネガティブケース（後述）の gold 照合には公式 `theatticusproject/cuad-qa`
  （parquet 変換ブランチ。answers 空 = gold「該当なし」を含む）を使用。
- **Retrieved**: 2026-07-04（初版100件）/ 2026-07-19（拡張50件）

## タスク概要

CUAD の **150件サブセット**（train 50 / dev 40 / test 60）。契約書抜粋＋条項カテゴリの
質問に対し、該当箇所を原文どおり抜き出す抽出型QA。該当条項が無いケース
（ネガティブ、正解は「該当条項なし」）を train に 10件・dev に 8件含む（test は
初版からの互換性維持のため全件ポジティブのまま）。
`task.yaml` は `answer_type: text` ＋ `llm-rubric` ジャッジで評価する。

**dev split の目的**（docs/DESIGN.md §5.1）: `evalloop optimize` の自動評価と McNemar
出荷ゲートは dev のみで行い、test は promoted な variant の最終確認1回のために温存する。
ベースラインは `evalloop run --task cuad100 --split dev` で作る。
n=40 の McNemar は不一致ペア数に依存し、10pt 級の差の検出は難しい —
出荷ゲートは「大きな改善のみ通す」保守的な判定であることに注意。

## 構成の履歴

| 版 | 構成 | 備考 |
|---|---|---|
| 初版 (2026-07-04) | train 20 / test 80、本文は中略あり | 旧 sha256 `cc67ad2a…` |
| 再配分版 | train 40 / test 60（100件・全文コンテキスト） | GEPA 実験期の再カット |
| 現行 (2026-07-19) | **train 50 / dev 40 / test 60**（150件） | 既存100件は据え置き、`case-0101..0150` を追加 |

## サンプリング方法

### 既存100件（case-0001..0100）

`chenghao/cuad_qa` から 41 種の条項カテゴリを横断するよう選択（当初は手作業サンプリング）。
現在は `scripts/manifest.json` に全件のデータセットレコード id が固定されており、
`build` で当時のファイルとバイト一致で再生成できることを検証済み。
層化抽出ではないため、カテゴリ別精度評価には適さない。

- カテゴリ被覆: train 40件で37カテゴリ / test 60件で33カテゴリ（合わせて41カテゴリ全被覆）。
  契約書タイトルは 51 種（同一契約を複数カテゴリで使用、train/test で契約重複あり）

### 拡張50件（case-0101..0150、2026-07-19）

`scripts/build_golden.py select-extras`（seed 42、決定的）で選定し manifest に固定:

- **train ネガティブ +10**（case-0101..0110）: train のカテゴリ構成に比例配分
- **dev ポジティブ +32**（case-0111..0142）: test 60件のカテゴリ構成に比例配分（28カテゴリ）
- **dev ネガティブ +8**（case-0143..0150）: test のカテゴリ構成に比例配分
- 制約: 新規50件は契約書タイトルがすべて相異なり、既存100件の契約とも重複しない。
  実行時間対策としてコンテキスト 150,000 文字超の契約は選定から除外
- **ネガティブの gold 担保**: 公式 CUAD（`theatticusproject/cuad-qa`）で当該
  (契約, カテゴリ) の answers が空（=アノテータが「該当条項なし」と判定）であること、
  かつミラーに同ペアの回答可能レコードが存在しないことの両方を機械的に検証して採用

## フィールド構成

各レコードのフィールド:

- `id` — `case-NNNN` 形式の連番（case-0001..0150）
- `input` — `[契約書タイトル]` 行、契約書全文コンテキスト、`Source: cuad` 行、
  `[質問]` 行＋カテゴリ名。中略はしない（初版にあった `...(前略)...` 方式は
  再配分版で全文コンテキストに置き換え済み）
- `expected` — 該当条項の原文抜粋。複数ある場合はセミコロン+空白（`; `）で区切って
  全て列挙。ネガティブケースは **`該当条項なし`**（`src/evalloop/optimizers/metrics.py`
  の `NO_CLAUSE_ANSWER` と正規化一致する文字列）
- `split` — `train` / `dev` / `test`
- `meta.category` — CUAD v1 の 41 カテゴリのいずれか
- `meta.difficulty` — 全件 `normal`（CUAD には難易度軸がないため固定値）
- `meta.source` — `"CUAD v1 (The Atticus Project, CC BY 4.0) via
  chenghao/cuad_qa mirror on Hugging Face"`。`task.yaml` の
  `blog.allowed_sources` と一致する値

## ファイル指紋（検証用）

- `golden.jsonl`（現行 150件版）:
  - sha256: `5babae35268d8854bf63dcf273919a14889ece5d359bae26fa09ee3065224a72`
  - サイズ: 8952121 bytes, 150行（末尾改行あり）
  - 検証: `shasum -a 256 tasks/cuad100/golden.jsonl`
- `human_labels.jsonl`: 当初は意図的に空（人手レビュー未実施）。issue #100 対応時に
  **gold-oracle プロキシ**（test 10件 × gold/`expected`=pass + 明確な誤答=fail、計20件）を
  ローカルへ作成し、`ollama:chat:glm-5.2:cloud` と `ollama:chat:deepseek-v4-pro:cloud`
  で `evalloop calibrate`（agreement 100%）済み。本物の人手ラベルではない（git 管理外）

## 再取得

```bash
# manifest.json から golden.jsonl を決定的に再生成（初回は HF からデータセットを取得）
uv run --with datasets python tasks/cuad100/scripts/build_golden.py build
# 手元のファイルとのバイト一致検証だけ行う場合
uv run --with datasets python tasks/cuad100/scripts/build_golden.py build --check
```

`datasets` はプロジェクト依存に含めていない（データ再生成時のみ必要）ため
`uv run --with datasets` で注入する。生成後、上記 sha256 と一致することを確認。
CC BY 4.0 のため出典クレジットを保持すれば再配布可だが、本リポジトリの
データポリシーに従い git にはコミットしない。

## ライセンス上の注意

CUAD は CC BY 4.0 で、`meta.source` にクレジットを明記済み。契約書抜粋をそのまま
`golden.jsonl` に格納しているのは再現性確保のため。ブログ記事化の際は
`task.yaml` の `blog.allowed_sources` が `meta.source` の値をホワイトリスト
検査するので、許可された出典文字列以外はビルドガードで弾かれる。
