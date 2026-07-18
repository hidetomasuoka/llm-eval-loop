# llm-eval-loop v2 — 設計ドキュメント（promptfooハイブリッド構成）

> **このドキュメントについて**: これは実装時にClaude Codeへ渡した元の設計仕様書。
> 全マイルストーン（M1〜M5）は実装済み。今後の変更もこの設計方針（責務分担・鉄の掟）に
> 従うこと。実際の使い方は [README.md](../README.md) を参照。
> promptfoo / dspy のAPI詳細は公式ドキュメント（promptfoo.dev / dspy.ai）を正とする。
> 実装時に確認した値（Node.jsバージョン要件など）は本ドキュメントにも反映済み。

> **⚠ 2026-07-05 追記 — 複数タスクワークスペース構成（issue #47）**:
> 本書のパス規約（`data/golden.jsonl`・`prompts/base/`・`promptfoo/promptfooconfig.yaml`・
> `results/runs/` 等の単一タスク前提の固定パス）は **`tasks/<name>/` ワークスペース構成に
> 置き換えられた**。現行の正は以下のとおり（詳細設計は issue #47、使い方は README）:
> - 1タスク = `tasks/<name>/`（task.yaml / golden.jsonl / prompts/ / human_labels /
>   taxonomy / notes / optimized/ / PROVENANCE.md）。生成物はタスク別サブツリー
>   （`data/build/<t>/`・`promptfoo/<t>/`・`results/<t>/`・`blog/<t>/`）
> - ルート `config.yaml` はモデルregistry＋runデフォルト＋default_taskのみ。タスク選択は
>   `--task` / `EVALLOOP_TASK` / default_task。パス解決は `src/evalloop/paths.py` の
>   `TaskPaths` に一元化。タスクの新規作成は `evalloop task init NAME`
> - **データポリシー**: タスクのデータ（golden/human_labels/notes/taxonomy）は既定で
>   gitignore。追跡は task.yaml / prompts/ / PROVENANCE.md のみ（合成データの
>   sample-inquiry だけオプトイン追跡）
> - 鉄の掟（責務分担・split分離・append-only台帳・公開ガード等）は**タスク単位でそのまま
>   有効**。本書の設計根拠・データ仕様（セクション5のレコード形式）・鉄の掟の記述は
>   引き続き正であり、置き換わったのはパスレイアウトと設定ファイルの分割のみ。

---

## 1. 目的

個人開発のLLM評価ハーネス。1つのタスクについて、**小型ローカルモデル〜フロンティアモデル（Claude Fable級）まで同一条件で精度を測り**、以下のループを回す。

```
前処理(データセット) → 実行×判定(promptfoo) → 失敗分析(Python) → 改善(GEPA) → 再評価 → ブログ公開用エクスポート
```

答える問いは1つ:

> **「どのモデルが必要精度を満たすか。それはいくら（コスト）で達成できるか」**

最終成果物は2つ:

1. 「モデル × 精度 × コスト × レイテンシ」マトリクスと「失敗カテゴリ × モデル」ピボット
2. **ブログ記事にそのまま貼れる図表・実験条件・記事ドラフト一式**（セクション9）

## 2. アーキテクチャ方針（責務分担）

実行と判定はpromptfooに任せ、Pythonは薄いグルーレイヤーに徹する。

| 責務 | 担当 | 理由 |
|---|---|---|
| マルチモデル実行・並列制御 | **promptfoo** | providers定義だけで済む |
| 決定的判定（ラベル一致・JSON検証） | **promptfoo** (javascript / is-json assert) | 設定のみ、コード最小 |
| LLMジャッジ | **promptfoo** (llm-rubric + provider固定) | {reason, score, pass} が構造化で返る |
| キャッシュ | **promptfoo** 内蔵ディスクキャッシュ | 実装不要 |
| 反復実行（非決定性対策） | **promptfoo** `--repeat N` | 実装不要 |
| 結果の目視確認 | **promptfoo** web viewer (`promptfoo view`) | ブログ用スクショにも使える |
| データセット管理・split分離・tests生成 | **Python (evalloop)** | 鉄の掟の担保はこちらで持つ |
| ジャッジ校正（人手ラベルとの一致率） | **Python** | promptfooに機能なし |
| 失敗分析（タクソノミー・ピボット） | **Python** | promptfooのoutput.jsonをパース |
| GEPA最適化 | **Python (dspy)** | promptfooに機能なし |
| run台帳・比較・ブログ出力 | **Python** | 〃 |

`calibrate`（judge再判定）と `cluster`（タクソノミー案生成）もこの原則に従い、
**モデル呼び出しは常にpromptfoo経由**で行う（`echo` providerで既存出力をリプレイしたり、
使い捨てconfigで1回のevalを流したりする形で実装している。`src/evalloop/calibrate.py`,
`src/evalloop/analyze.py` 参照）。Pythonが直接LLM APIを叩くことはない。

> **補足**: promptfooは2026年3月にOpenAIに買収されたが、現行ライセンス（MIT）のOSSとして継続することが公式に表明されている（出典: [OpenAI to acquire Promptfoo](https://openai.com/index/openai-to-acquire-promptfoo/)、[Promptfoo is joining OpenAI](https://www.promptfoo.dev/blog/promptfoo-joining-openai/)）。ローカル実行ならデータは手元に残る。**`promptfoo share` はクラウドアップロードなので本プロジェクトでは使用禁止**（セクション9の公開ガード参照）。

## 3. 技術スタック

- **promptfoo**: `npx promptfoo@<固定バージョン>` で実行（グローバルインストール不要）。Node.js が前提。
  バージョンは `src/evalloop/run.py` の `PROMPTFOO_VERSION` で一元的に固定する（サプライチェーン
  露出と再現性ドリフト対策。`@latest` は使わない）。
  promptfoo 0.121.17 の実行時チェックで確認済みの必須レンジは **`^20.20.0` または `>=22.22.0`**
  （21.x、および22.0.0〜22.21.xは非対応。`node --version` が範囲外だとpromptfoo自体がハードエラーで起動を拒否する。
  これは実装時に実機で確認した値であり、`src/evalloop/cli.py` の `_node_version_ok()` もこの正確なレンジでチェックしている）
- **Python 3.11+ / uv**: `dspy`, `pandas`, `matplotlib`, `typer`, `pyyaml`, `rich`, `pytest`
- **Ollama**: ローカル小型モデル用
- 日本語図表のため matplotlib は起動時に **Hiragino Sans / Noto Sans CJK JP / Yu Gothic** 等の
  CJK対応フォントを探索し、`font.family` に設定する（`src/evalloop/blog.py` の `find_cjk_font()`）。
  フォント未検出時は警告を出し、図のラベルを英語にフォールバックする（□文字＝豆腐の混入防止）

## 4. ディレクトリ構成

```
llm-eval-loop/
├── README.md
├── docs/DESIGN.md               # 本ファイル
├── pyproject.toml
├── config.yaml                  # evalloop側マスター設定（セクション6）
├── data/
│   ├── golden.jsonl             # 評価データセット（5.1）
│   ├── human_labels.jsonl       # ジャッジ校正用（5.3）
│   ├── notes.csv                # オープンコーディング用（5.4）
│   ├── taxonomy.yaml            # 失敗タクソノミー（5.5）
│   ├── build/                   # ★evalloop build の自動生成物。手編集禁止（git非追跡）
│   │   ├── tests_test.yaml      # promptfoo用テスト（split=test のみ）
│   │   └── tests_train.yaml     # GEPA用（promptfooのevalには絶対渡さない）
│   └── sample/                  # 動作確認用ダミー一式（セクション10）
├── promptfoo/
│   ├── promptfooconfig.yaml     # buildがテンプレートから生成（git非追跡）
│   └── variants/                # 最適化プロンプト版config（optimizeが生成、git非追跡）
├── prompts/
│   ├── base/task.txt            # {{input}} プレースホルダを含むベースプロンプト
│   ├── base/judge_rubric.txt
│   └── optimized/{alias}/{ts}/task.txt   # GEPA出力（上書き禁止）
├── results/                     # run成果物はすべてgit非追跡（README「生成物ポリシー」参照）
│   ├── runs/{run_id}/
│   │   ├── output.json          # promptfoo -o の生出力
│   │   └── meta.json            # config snapshot・コスト集計・校正状態
│   ├── index.jsonl              # 全run台帳（追記のみ、マシンローカル）
│   └── reports/                 # Markdownレポート
├── blog/                        # ブログ用エクスポート（セクション9、git非追跡）
├── src/evalloop/
│   ├── cli.py                   # typerエントリポイント
│   ├── build.py                 # golden → promptfoo tests/config 生成
│   ├── run.py                   # promptfoo evalのラッパー（run_id管理）
│   ├── report.py
│   ├── calibrate.py
│   ├── analyze.py               # failures / cluster / pivot
│   ├── optimize.py              # dspy GEPA
│   ├── blog.py                  # 図表・記事ドラフト生成
│   ├── asserts/label_match.js   # promptfoo用 正規化つきラベル一致assert
│   ├── asserts/json_field_match.js
│   └── schemas.py
└── tests/
```

## 5. データ仕様

### 5.1 golden.jsonl（評価データセット・唯一のソース）

```json
{"id": "case-0001", "input": "（タスクへの入力テキスト）", "expected": "契約照会", "split": "test", "meta": {"category": "基本", "difficulty": "easy", "source": "self-made"}}
```

| フィールド | 型 | 必須 | 説明 |
|---|---|:---:|---|
| id | str | ○ | 一意。`case-` プレフィックス |
| input | str | ○ | プロンプトの `{{input}}` に入る |
| expected | str \| dict | ○ | answer_typeに応じた期待値 |
| split | "train" \| "test" | ○ | **後から変更しない** |
| meta.category | str | ○ | ピボット軸になる事前分類 |
| meta.difficulty | str | - | easy / normal / hard |
| meta.source | str | ○ | **"self-made" またはライセンス明記。ブログ公開ガードが参照する** |

### 5.2 promptfoo output.json（実行×判定結果）

promptfooの `-o` 出力をそのまま保存。Python側は `results[]`（実際には `results.results[]` の
ネスト構造。バージョンにより flat な `results[]` の場合もあるため両対応）の各要素から
`vars`（case_id, expected, category を含める）、`response`（output, cost, latencyMs,
tokenUsage, cached）、`gradingResult`（pass, score, reason, componentResults）を読む。
**パーサは promptfoo のバージョン差異に備えて `schemas.py` の `parse_promptfoo_output()` に
分離し、キー欠落は警告して続行する**（例外を投げない）。

### 5.3 human_labels.jsonl（ジャッジ校正用）

```json
{"case_id": "case-0001", "model_label": "haiku45", "output_raw": "...", "human_verdict": "pass"}
```

**(case_id, model_label) が複合主キー**。同一caseを複数モデルの出力についてそれぞれラベルする
前提であり（ジャッジ校正実験は最低2モデル×同一ケース群を推奨）、calibrateはジャッジ判定を
この複合キーでマッチバックする。同一 (case_id, model_label) の重複行はロード時にエラーになる。

### 5.4 notes.csv / 5.5 taxonomy.yaml

notes.csv は `case_id, model, input_head, output_head, expected, note`。
taxonomy.yaml は `categories`（id/name/definition のリスト）と `assignments`（case_id→category id の辞書）。

### 5.6 demos.jsonl（任意・few-shot例）

```json
{"id": "case-0001", "input": "（例の入力）", "output": "契約照会"}
```

| フィールド | 型 | 必須 | 説明 |
|---|---|:---:|---|
| input | str | ○ | デモ入力 |
| output | str | ○ | デモ出力（ラベル / テキスト / JSON文字列） |
| id | str | - | あれば golden の test id と照合してリーク検査 |

- 既定で gitignore（`tasks/*/demos.jsonl`）。出所は必ず `PROVENANCE.md` に記載
- `prompts/task.txt` に `{{demos}}` があるときだけ `evalloop build` が
  `Input: ...\nOutput: ...\n\n` 形式で展開し、`data/build/<task>/prompt.resolved.txt` を生成して
  promptfoo から参照する
- `evalloop optimize` も同じ `{{demos}}` を展開してから dspy に渡す（build 後の promptfoo 評価と訓練テンプレを一致させる）
- `{{demos}}` あり・ファイルなし → build/optimize エラー / ファイルあり・プレースホルダなし → 警告
- **MIPROv2 few-shot 探索**（`optimize.method: miprov2` かつ `params.max_bootstrapped_demos` /
  `max_labeled_demos` が正）: 探索中は `{{demos}}` を空にして instruction に焼き込まない。
  完了後に train split 由来の demos を `optimized/<alias>/<variant>/demos.jsonl` へ保存し
  （行ごと `origin: labeled|bootstrapped` と出所メタ）、variant の `task.txt` に再展開する。
  プロンプトに `{{demos}}` が無いと preflight エラー
- **鉄の掟**: demo の `id` または `input` が、現在の golden test split と直近 build の
  `tests_test.yaml` holdout の**和集合**と重複したらエラー（リーク防止。golden を更新しても rebuild 前の YAML を見落さない）

## 6. config.yaml 仕様（evalloopマスター設定）

実際の値は [config.yaml](../config.yaml) を参照（コメント付き）。構造は以下の通り。

```yaml
task:
  name: sample-inquiry-classification
  answer_type: label            # label | json | text
  prompt_file: prompts/base/task.txt
  labels: ["契約照会", "障害報告", "機能要望", "その他"]
  json_schema_file: null        # answer_type=jsonのとき必須
models:                         # provider IDの表記は実装時にpromptfoo公式Docsで確認済み
  - provider: ollama:chat:qwen2.5:7b
    alias: qwen7b
    tier: local
    price_in_per_mtok: 0.0
    price_out_per_mtok: 0.0
  # ... (haiku45 / sonnet46 / opus48 / fable5)
run:
  repeat: 1                     # 安定測定フェーズでは3
  temperature: 0.0
  max_tokens: 1024
  cost_warn_usd: 3.0            # 実行前概算がこれを超えたら確認プロンプト
judge:
  provider: anthropic:messages:claude-sonnet-4-6   # llm-rubricのgrader。必ず明示する
  threshold: 0.8
  agreement_threshold: 0.85
  rubric_file: prompts/base/judge_rubric.txt
optimize:
  target_alias: qwen7b
  reflection_provider: anthropic/claude-opus-4-8   # dspy側の表記（litellm形式）
  auto: light
blog:
  jpy_per_usd: 150              # コストの円換算表示用（0で非表示）
  slug_prefix: llm-eval
```

> `models[].provider` はpromptfoo表記（例: `anthropic:messages:claude-...`）、
> `optimize.reflection_provider` はdspy/litellm表記（例: `anthropic/claude-...`）と
> **書式が異なる**点に注意。`optimize.target_alias` で指定したモデルは
> `src/evalloop/optimize.py` の `promptfoo_provider_to_dspy_lm()` で自動変換される
> （対応済み: `anthropic:messages:` , `ollama:chat:`。他のprefixを追加する場合は
> このマッピングにケースを追加すること）。

## 7. CLI仕様

エントリポイントは `evalloop`（`uv run evalloop ...`）。全コマンド実装済み。

| コマンド | 入力 | 出力 | 説明 |
|---|---|---|---|
| `doctor` | config.yaml | 標準出力 | Node/promptfoo/Ollama/APIキーの疎通確認。全providerに1件だけ極小evalを流す |
| `build` | golden.jsonl, config.yaml | data/build/*, promptfoo/promptfooconfig.yaml | tests生成＋config生成＋**実行前コスト概算表示** |
| `run [--variant NAME] [--repeat N] [--limit N] [--no-cache]` | build成果物 | results/runs/{run_id}/ | `npx promptfoo@<固定バージョン> eval -c ... -o ...` をsubprocess実行し、meta.json・index.jsonlを記録 |
| `view` | - | - | `npx promptfoo@<固定バージョン> view` のパススルー |
| `report RUN_ID` | output.json | reports/{run_id}.md | マトリクスレポート |
| `calibrate [--run-id ID]` | human_labels.jsonl | 標準出力＋meta更新 | ジャッジと人手ラベルの一致率。閾値未満なら警告 |
| `failures RUN_ID` | output.json | results/runs/{run_id}/failures.jsonl, data/notes.csv | 失敗抽出＋メモ用テンプレ生成（追記・冪等） |
| `cluster [--notes data/notes.csv]` | notes.csv | data/taxonomy.draft.yaml | LLM(promptfoo経由)でタクソノミー案生成（既存taxonomy.yamlは上書きしない） |
| `pivot RUN_ID` | output.json + taxonomy.yaml | reports/pivot_{run_id}.md | 失敗カテゴリ×モデルのクロス集計（unassigned行あり） |
| `optimize` | golden(train) | prompts/optimized/..., promptfoo/variants/... | GEPA実行→最適化プロンプト保存→variant config生成→自動run/report/compare |
| `compare --runs A,B` | 2つのrun | reports/compare_A_B.md | before/after比較（精度差・コスト差） |
| `blog --runs A[,B] [--slug NAME]` | run(s) | blog/{date}_{slug}/ | セクション9の一式を生成 |

## 8. 実装仕様の要点

### 8.1 build.py

1. golden.jsonl を読み、split別に promptfoo tests ファイルを生成（YAML）。各テストは `vars: {case_id, input, expected, category}` を持つ
2. answer_typeに応じて `defaultTest.assert` を構成:
   - **label**: `javascript` assert（`asserts/label_match.js`）。出力を正規化（前後空白・全角半角・末尾句点除去）し、`context.vars.expected` と一致、またはラベルリスト中の1つだけが出力に含まれる場合にpass
   - **json**: `is-json`（json_schema_file指定）＋ `javascript`（`asserts/json_field_match.js`）でフィールド比較
   - **text**: `llm-rubric`。`provider` に judge.provider を**必ず明示**（環境変数依存のデフォルトグレーダー禁止）、`threshold` 設定。
     rubricは**ファイルの中身を読み込んでインライン文字列として`value`に埋め込む**（`file://...`参照ではない）。
     実機検証で判明: promptfoo 0.121.17では`llm-rubric`の`value`が`file://`参照だと
     `{{input}}`/`{{expected}}`がNunjucksテンプレート処理されず、リテラルな`{{input}}`文字列の
     ままジャッジに渡ってしまう（インラインの`value`文字列なら置換される）。`calibrate.py`の
     echo-replay用ルーブリックも同様にインライン化している
3. promptfooconfig.yaml をテンプレートから生成: providers（temperature/max_tokens含む、`label`にaliasを設定）、prompts（file://参照）、tests（**tests_test.yaml のみ**）
4. 実行前コスト概算 = Σ(モデル単価 × 推定トークン × ケース数 × repeat) を表示。`cost_warn_usd` 超過時は y/n 確認（`--yes` でスキップ可）

### 8.2 run.py

- run_id = `YYYYMMDD-HHMMSS-{4桁hex}`。subprocess で promptfoo eval を実行し、output.json を run ディレクトリへ
- meta.json に: 実際に使用したpromptfoo configのpath/sha256、同configから一意に解決した実効プロンプトファイルのpath/sha256（inline・複数promptはnull）、repeat、実測コスト合計（output.jsonから集計）、実効grader（text=`llm-rubric`、label=`label-match`、json=`json-field-match`）と校正状態、promptfooバージョン。label/jsonの校正状態は `not_applicable` とし、旧runの読み取りでは既存の `judge` objectへfallbackする
- index.jsonl へ追記（**追記のみ、削除機能は作らない**）。promptfoo自体が完全に失敗して output.json が生成されなかった場合でも、meta.json/index.jsonlには記録してから例外を送出する（孤立したrun_idディレクトリを残さないため）
- promptfooのキャッシュはデフォルト有効のまま使う（`--no-cache` はフラグで明示したときのみ）。`--share`は常に無効（`--no-share`固定）
- `npx` の実行は `shutil.which("npx")` で解決する（Windowsでは`npx`が`npx.cmd`のため、素の`subprocess.run(["npx",...])`は失敗する）

### 8.3 calibrate.py

human_labels.jsonl の各ケースについて、`--run-id` があれば既存runのgradingResultと照合、
なければpromptfooの`echo` providerで各`output_raw`をリプレイしてllm-rubricで再判定し、
一致率を算出。`agreement_threshold` 未満なら**以降のreportに `⚠ uncalibrated/low-agreement judge` を必ず表示**。

### 8.4 analyze.py（failures / cluster / pivot）

`failures`は promptfoo output.json から失敗行を抽出し、`data/notes.csv` に追記する
（既存の`case_id,model`組は上書きしない＝手書きnoteを壊さない）。`cluster` はLLM
（judge.provider、promptfoo経由）でカテゴリ案＋割当案を生成し `taxonomy.draft.yaml` に出力、
人間がマージ。`pivot` は未割当失敗を `unassigned` 行に集計。

### 8.5 optimize.py（GEPA）

1. dspy 3.2.1 の `dspy.Signature("input -> output", instructions=...)` + `dspy.Predict` を使用。
   学習データは **golden.jsonl の split=="train" のみ**。test IDとの積集合が空であることを
   `assert_split_disjoint()` でassert、違反は即異常終了
2. metric（`label_score_and_feedback`）は `asserts/label_match.js` と同じ正規化ロジックの
   Python移植版。GEPAは候補ロールアウトごとにmetricを呼ぶため、promptfoo経由にはできない
   （in-processである必要がある）。**スコア＋テキストフィードバック**（`dspy.Prediction(score=, feedback=)`）を返す
3. `reflection_lm` は `optimize.reflection_provider`（litellm形式、大型モデル）
4. 出力: `prompts/optimized/{alias}/{ts}/task.txt` ＋ `optimize_log.json`。続けて
   `promptfoo/variants/{alias}_{ts}.yaml`（プロンプトだけ差し替えたconfig。
   `promptfoo/variants/`は`promptfoo/`の1階層下なので、`file://`参照はすべて
   `../`を1つ多く挿入して再root化している）を生成
5. 自動で `run --variant` → `report` → `compare`（index.jsonlにある直近のベースrunがあれば
   最適化前後を比較。大型モデルも同じrunに含まれるため表に自然に含まれる）まで実行
6. **現状 `answer_type=="label"` のみ対応**（metricがlabel_match.js相当のみ移植済みのため）。
   json/textタイプでoptimizeを呼ぶと明示的にOptimizeErrorになる — 対応する場合は
   json用・text用のmetricをoptimize.pyに追加すること

## 9. ブログ出力仕様（blog.py）★

`evalloop blog --runs A[,B]` で `blog/{YYYYMMDD}_{slug}/` に以下を生成する。**「図表とデータはコピペで記事に貼れる」状態がゴール。**

### 9.1 図（PNG 150dpi ＋ 同名SVG）

| ファイル | 内容 |
|---|---|
| `fig01_accuracy_by_model.png` | モデル別精度の棒グラフ。--runsが2つのときはrunごとにグループ化した棒。tier順に並べる |
| `fig02_cost_vs_accuracy.png` | 散布図。x=1ケースあたり実測コストUSD（対数軸）、y=精度。点ラベル=alias。--runsが2つ（最適化前後）のときは同一モデルを矢印で結ぶ |
| `fig03_failure_heatmap.png` | 失敗カテゴリ×モデルのヒートマップ（`data/taxonomy.yaml`未定義時はスキップし、その旨を標準出力に表示） |

- 日本語ラベル必須のため、起動時にCJKフォントを探索して`font.family`に設定する。未検出時は警告して英語ラベルにフォールバック（豆腐を出さない）
- 配色はモデルのtierで色相を揃える

### 9.2 テキスト

| ファイル | 内容 |
|---|---|
| `tables.md` | run毎のサマリ表（モデル×tier×精度×総コスト×p50レイテンシ×キャッシュ率）。**そのまま貼れるMarkdown** |
| `conditions.md` | 再現性ブロック: 実験日 / モデルID一覧（provider ID表記そのまま）/ repeat / temperature / プロンプトsha256先頭8桁 / ジャッジmodelと校正一致率 / 総コスト（USDと円換算）/ promptfoo・dspyバージョン / 再現コマンド列 |
| `article_draft.md` | 記事スケルトン: タイトル案 → 背景 → 手法 → 結果（図表への相対パス参照と数値を自動埋め込み）→ 考察（TODO）→ 限界と注意 → 再現手順。**文章はプレースホルダで、数値と図参照だけ自動生成** |

### 9.3 公開ガード（blog実行時に自動チェック、fail時は生成中断）

1. golden.jsonl の全ケースで `meta.source == "self-made"`（またはconfigで許可したライセンス値）であること。違反IDを列挙して中断
2. 生成物は一旦一時ディレクトリに書き出し、APIキーパターン（`sk-`, `AKIA` 等）・ホームディレクトリの絶対パスをgrepしてから`blog/`へコピーする。検出時は`blog/`に一切書き込まず中断（部分生成物を残さない）
3. article_draft.md の冒頭に `<!-- 公開前に固有情報がないか目視確認 -->` を必ず挿入
4. `promptfoo share` は使わない（README冒頭と `doctor` の出力にも注意書きを出す。`run.py`は常に`--no-share`を付与する）

## 10. サンプルタスク

- タスク: 「日本語の問い合わせ文を `契約照会 / 障害報告 / 機能要望 / その他` に分類」、answer_type=label
- `data/sample/golden.jsonl` にダミー20件（train 8 / test 12、meta.category は `基本` / `曖昧` / `複合`、**meta.source は全件 "self-made"**）
- `data/sample/human_labels.jsonl` にダミー10件
- `prompts/base/task.txt`（{{input}}入り）と `judge_rubric.txt` の初期版
- config.yaml 初期値はこのサンプルを指していた

> **現状**: `data/golden.jsonl` / `config.yaml` / `prompts/base/*.txt` は実地検証のため
> CUAD-100タスク（README.md参照）に差し替え済み。`data/sample/` 配下のファイルはこの
> サンプルタスクの原本として変更されていない。サンプルタスクに戻すには
> `data/sample/golden.jsonl` → `data/golden.jsonl`、`data/sample/human_labels.jsonl` →
> `data/human_labels.jsonl` にコピーし、`config.yaml`の`task.*`を本セクション冒頭の値に
> 戻せばよい。

## 11. 鉄の掟（違反する実装は不可）

1. **split分離はファイル分離で担保する**: tests_train.yaml を eval 用configが参照したらビルドエラー（`build.py`の`_assert_config_never_references_train`）。optimizeはtrainのみ読み、train/test IDの交差を`assert_split_disjoint`でassert
2. **llm-rubricのgraderは必ずproviderを明示**し、被評価モデルと同一なら停止（`--allow-same-judge` でのみ回避可）。環境キー依存の暗黙デフォルトグレーダーは禁止
3. **結果は追記のみ**。run_id台帳で管理し、上書き・削除機能は作らない。promptfoo自体が失敗した場合も含めて必ず記録する
4. **キャッシュ有効がデフォルト**。無効化は明示フラグのみ
5. **コストの二段ガード**: 実行前概算で確認プロンプト、実行後は実測をmeta/indexに記録。概算に使う単価表(config)は使用時点の公式価格に更新する
6. **未校正／低一致率ジャッジのスコアには必ず警告表示**
7. **ブログ出力は公開ガード（9.3）を通過しないと生成されない**。`promptfoo share`はどこからも呼ばない

## 12. マイルストーン（全て実装済み）

| マイルストーン | 内容 | 状態 |
|---|---|---|
| M1: 最小ループ | `doctor` / `build` / `run` / `report` | ✅ 実装済み・テスト済み |
| M2: ジャッジ整備 | llm-rubric構成・`calibrate` | ✅ 実装済み・テスト済み |
| M3: 失敗分析 | `failures` / `cluster` / `pivot` | ✅ 実装済み・テスト済み |
| M4: 改善ループ | `optimize`（dspy GEPA）/ `compare` | ✅ 実装済み・テスト済み（`answer_type=label`のみ） |
| M5: ブログ出力 | `blog` + 公開ガード | ✅ 実装済み・テスト済み |

各モジュールのユニットテストは `tests/` にあり、`pytest` で全件確認できる
（label正規化・split分離assert・output.jsonパーサ・公開ガードを含む）。

**実装時の既知の制約**（README.mdの「既知の制約」「Windows実地検証」も参照）:
- 初期実装はNode.js・APIキー・Ollamaの無いサンドボックス環境で行い、`npx promptfoo eval` と
  dspy GEPA の実API呼び出し部分はsubprocess/API境界をモック化したユニットテストのみで検証した
- その後、実機（Windows 11 + Node.js v22.23.1 + Ollama）で `doctor`/`build`/`run`/`report`/`blog`
  を実際に動かして検証し、モックだけでは見つからなかった複数の実装バグ（npx解決・cp932
  デコード・llm-rubricのfile://テンプレート未展開など）を発見・修正した。詳細は
  README.md「Windows実地検証で見つかった問題と修正」を参照
- `optimize` は `answer_type=="label"` のみ対応（metricの都合上）。現在アクティブなタスクは
  `answer_type=text`（CUAD-100）のため、`optimize`自体は`data/sample/`のラベルタスクに
  戻すか、text用metricの追加実装をしないと使えない

## 13. セットアップ

```bash
node --version        # ^20.20.0 または >=22.22.0 であることを確認（21.x, 22.0-22.21は不可）
uv sync
ollama pull qwen2.5:7b
export ANTHROPIC_API_KEY=...     # 必須（実行・ジャッジ・GEPA reflection）
export OPENAI_API_KEY=...        # 任意
export GEMINI_API_KEY=...        # 任意
uv run evalloop doctor           # 最初に必ず疎通確認
```

> **注意**: config.yaml のprovider ID・単価はあくまで例。promptfoo側の命名規則やモデル提供状況・価格は変わるため、`doctor` が通らないIDは使わず、単価は使用時点の公式価格に更新すること。

## 14. 実装メモ

- 例外はケース単位で握りつぶさず promptfoo に任せ、Python側はoutput.jsonの `error` を集計に反映
- 日本語のためファイルI/OはすべてUTF-8明示
- 乱数を使う箇所はseed固定（`optimize.py`のGEPA呼び出しは`seed=0`固定）
- 本タスク（sample）差し替え時に触るのは `config.yaml` / `data/golden.jsonl` / `prompts/base/*.txt` の3点だけで済む構造にしている
- 将来Langfuse等へ移行する場合に備え、output.jsonのパースは `schemas.py` に閉じ込めている
