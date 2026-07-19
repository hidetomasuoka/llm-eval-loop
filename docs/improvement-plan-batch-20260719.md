# llm-eval-loop 改善バッチ実装プラン

## Context

ヒアリングの結果、`docs/improvement-plan-gepa-cuad100.md`（2026-07-06）の未消化項目＋コード品質改善をまとめて実施することになった。cuad100 の GEPA 実験の振り返りで判明した構造的欠陥 — 「訓練データで自己採点する候補選抜」「出荷ゲート不在で test を毎回消費」「paired 設計なのに独立2標本の検定」「言い換えでも点が入る proxy metric」— を塞ぎ、あわせて 1,373 行に膨らんだ `optimize.py` を責務別に分割する。

**完了済み**（ブランチ `feat/improvement-plan-batch` にコミット済み、ca81420）:
- 私的ファイル `tasks/cuad100/AWS_Support_減免申請.md` を `~/Documents/` へ移動
- `ruff format` 全体適用 + CI に `ruff format --check` 追加（保留 TODO 解消）

以降のステップはすべて同ブランチに1ステップ=1コミットで積む。

## Step 1: optimize.py の責務別分割（純粋な移動、挙動変更なし）

`optimize.py`(1,373行) を以下に分割。既存の APO-03 リファクタ（metrics を optimizers/ へ移して optimize.py で再エクスポート）と同じパターンを踏襲する。

| 新モジュール | 移動する内容（現 optimize.py の行） |
|---|---|
| `src/evalloop/dspy_lm.py` | provider→dspy LM 変換・temperature・reflection registry（92-137）、`SearchCostSummary` と LM history コスト集計（188-257） |
| `src/evalloop/optimize_cost.py` | `OptimizeCostEstimate`・`_rollout_factor`・`estimate_optimize_cost`・定数（140-186, 260-298） |
| `src/evalloop/variants.py` | variant config 生成（301-330）、slug/summary（333-479）、`_append_optimized_index` |
| `src/evalloop/compare.py` | fmt ヘルパー・tradeoff 定数・index/log ローダー・`_compare_pair`・`_compare_matrix`・`compare`（961-1321） |
| `optimize.py`（残す） | `optimize()` orchestration、generalization gate、`OptimizeOutcome`、`_find_latest_base_run` |

**後方互換**: `optimize.py` 冒頭で移動した全シンボル（テストが参照する `_make_variant_slug`、`_compare_report_filename` 等のプライベート名含む）を `# noqa: F401` 付きで再エクスポート。テストの monkeypatch 対象は `run_gepa`/`run_miprov2`/`run_copro`/`run_tapo` のみ（確認済み）で、これらは元々 optimize.py 名前空間の慣習なので影響なし。`blog.py:40-44` の `_compare_matrix`/`_method_for_variant`/`_COMPARE_MULTI_DISCLAIMER` は `evalloop.compare` からの直接 import に書き換える。

**検証**: 既存 358 テストが無修正で green であること。

## Step 2: McNemar exact 検定（計画 #2）

- 新規 `src/evalloop/stats.py`:
  - `mcnemar_exact_p(b, c) -> float | None`: 両側 exact binomial（`math.comb`、scipy 不要）。b+c==0 → None
  - `paired_transition(results_a, results_b, alias) -> (b, c, n_paired)`: `CaseResult` を (alias, case_id) でペアリング。repeat>1 のケースは多数決（同数は fail）で1判定に潰す。片方にしか無い case は除外
- `compare.py` の `_compare_pair` に列追加: `b/c`（改善/悪化ケース数）と `mcnemar_p`。フッター注記に「同一ケース集合の paired 検定。Wilson CI 非重複より検出力が高い」を追記
- テスト: `tests/test_stats.py` 新規（既知値ピン: b=12,c=4 → p≈0.077 等）+ `test_compare_multi.py` に列の存在確認を追加

## Step 3: verbatim 検証 + WE/PE 分岐 feedback（計画 #3）

対象: `src/evalloop/optimizers/metrics.py` の `text_score_and_feedback`（219-255行）

- シグネチャを `text_score_and_feedback(output, expected, source: str | None = None)` に拡張（`source` = ケースの原文書）。`_score_fn_for` の返す関数も `(output, expected, source=None)` に統一し、`optimize.py` の metric closure（752-754行）から `gold.input` を渡す。`calibrate.py:283` は label/json のみ import しており影響なし
- **verbatim 検証**: 正規化（空白圧縮・小文字化）後、各出力スパンが `source` の連続部分文字列かを判定。非 verbatim スパンがあればスコアを `min(score, 0.5)` にキャップし、「原文から一字一句そのまま引用せよ。要約・言い換えは final judge で fail になる」という専用 feedback を返す。`source=None` 時はスキップ（後方互換）
- **WE/PE 分岐**: `_span_set_score` の結果で分岐。score < 0.2（別条項選択 = WE 相当）→「見出し語の一致ではなくカテゴリの法的概念で条項を特定せよ」、0.2 ≤ score < 1.0（部分抽出 = PE 相当）→ 既存の「COMPLETE clause を抜け」文言。閾値は定数 `WE_OVERLAP_THRESHOLD = 0.2`
- テスト: `test_optimize.py`（または新規 `test_metrics_text.py`）に verbatim キャップ・WE/PE 分岐・source なし後方互換のケース追加

## Step 4: GEPA に valset を渡す（計画 #6）

対象: `src/evalloop/optimizers/gepa.py`

- `run_gepa()` に `valset=None` キーワードを追加し `GEPA.compile(student, trainset=..., valset=...)` へ渡す（dspy 3.2.1 の compile シグネチャで valset 対応確認済み）
- `GepaOptimizer.optimize()`: miprov2 の `split_train_val`（`optimizers/miprov2.py:50`、val_ratio 8:2・seed 固定）を再利用して train を分割。`params.val_ratio`（default 0.2）/ `params.seed` を尊重。trainset < 2 件なら valset なしにフォールバック + 警告
- `train_score` は train_part のみで算出（miprov2 の 0ab839b と同じ扱い）。`extra_log` に `train_size` / `val_size` / `val_ratio` を記録
- テスト: `test_optimize.py` の fake `run_gepa` に valset 引数を受けるよう追記し、分割・フォールバック・ログをピン

## Step 5: dev split + 出荷ゲート（計画 #4）

**スキーマ/ビルド**:
- `schemas.py:24` `VALID_SPLITS` に `"dev"` を追加。`assert_split_disjoint` を train/dev/test の3対で呼ぶ（build.py・optimize.py 双方）
- `paths.py` に `tests_dev`（`build/tests_dev.yaml`）と `promptfoo_config_dev`（`promptfoo/promptfooconfig.dev.yaml`）を追加
- `build.py`: dev ケースが存在するとき `tests_dev.yaml` と dev 用 promptfoo config（tests 参照先だけ差し替え）を追加出力。dev なしタスクは現状と完全に同一出力

**run**:
- `run.py run()` と CLI `evalloop run` に `--split dev|test`（default test）を追加。dev 指定で dev config を解決、`meta.json` に `"split"` を記録（既存 run は欠損 = test 扱い）
- variant 実行: `variants.py build_variant_config()` に split 引数を追加し、dev のときは tests 参照を `tests_dev.yaml` に向けた variant yaml を別名（`{name}.dev.yaml`）で生成

**出荷ゲート（optimize.py）**:
- dev ケースが存在するタスクでは、optimize の自動 run を **dev のみ** に変更（test は消費しない）。dev なしタスクは従来どおり test 実行 + 「dev split 追加を推奨」警告
- ゲート判定: 最新の base **dev** run（`_find_latest_base_run` を split 対応に拡張）との per-case ペアで `mcnemar_exact_p` を計算し、`delta > 0 かつ p < 0.05` のときだけ `promoted=True`。base dev run が無い場合は自動では回さず（コスト方針に従い勝手に API を叩かない）、`evalloop run --split dev` の実行を促すメッセージを出して `promoted=None`
- `OptimizeOutcome`・`optimize_log.json`・`optimized/index.jsonl` エントリに `promoted` / `gate_p_value` / `gate_split` を追加。コンソールに合否を明示（既存 generalization gate 表示の隣）
- テスト: `test_generalization_gate.py` 拡張 or 新規 `test_shipping_gate.py`（promoted 判定・base dev 不在・dev なしフォールバックの3系統）

**注**: cuad100 の実データを train/dev/test に切り直すのはデータ作業（計画 #5 と一体）なので本バッチ対象外。仕組みだけ入れる。

## Step 6: ドキュメント同期

- `README.md` / `README.ja.md`（日本語が正、両方同時更新）: `--split dev`、promoted ゲート、compare の McNemar 列を追記
- `docs/DESIGN.md`: split 3分割と出荷ゲートの設計根拠を追記
- `docs/improvement-plan-gepa-cuad100.md`: 消化済み項目に完了マークを追記

## 検証

1. 各ステップ後: `uv run ruff check . && uv run ruff format --check . && uv run pytest`（現在 358 passed、ステップごとに増加）
2. Step 1 直後はテスト**無修正**で green であることを分割の正しさの証拠とする
3. Step 5 後の実機確認（API 不要の範囲）: `uv run evalloop build --task sample-inquiry` が従来と同一の成果物を出すこと（dev なしタスクの後方互換）
4. 最後に `git log --oneline` でステップ単位のコミット履歴を確認。push / PR 作成は別途ユーザーに確認

## コミット構成（1ステップ=1コミット、済み分含め計7）

1. ✅ ruff format 全体適用 + CI 強制（ca81420）
2. optimize.py 分割（dspy_lm / optimize_cost / variants / compare）
3. stats.py + compare に McNemar
4. metrics に verbatim 検証 + WE/PE feedback
5. GEPA valset
6. dev split + 出荷ゲート
7. ドキュメント同期
