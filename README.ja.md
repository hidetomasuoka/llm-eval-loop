# llm-eval-loop

[![CI](https://github.com/hidetomasuoka/llm-eval-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/hidetomasuoka/llm-eval-loop/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

[English](README.md) | **日本語**

> このREADMEは英語版 [README.md](README.md) と内容を同期して維持する。差分が生じた場合は日本語版を正とする。

LLM評価ハーネス。1つのタスクについて、ローカル小型モデルからフロンティアモデルまで
同一条件で評価し、次の問いに答える。

> **どのモデルが必要精度を満たすか。それはいくら（コスト）で達成できるか。**

実行と判定は [promptfoo](https://www.promptfoo.dev/) に任せ、Python（`evalloop`）は
データセット管理・ジャッジ校正・失敗分析・[dspy](https://dspy.ai/) GEPAによるプロンプト最適化・
ブログ公開用エクスポートを担当する薄いグルーレイヤーとして実装されている。

個人プロジェクトとして開発している実験的なツールだが、git clone してそのまま使える状態を
維持している（for mac / for windows の実機検証状況は後述）。issue / PR は歓迎 — 対応はベストエフォート。
期待値と開発手順は [CONTRIBUTING.md](CONTRIBUTING.md) を参照。

設計方針・データ仕様・鉄の掟など詳細な設計根拠は [docs/DESIGN.md](docs/DESIGN.md) を参照。

> **ポリシー**: このプロジェクトは `promptfoo share`（クラウドアップロード）を一切使用しない。
> ローカル結果の閲覧には `evalloop view` を使う。

## 必要環境

- **Node.js**: `^20.20.0` または `>=22.22.0`（21.x、22.0.0〜22.21.xは非対応。promptfoo自体が
  この範囲外だと起動を拒否する。まず `node --version` で確認すること）
- **promptfoo**: バージョンは `src/evalloop/run.py` の `PROMPTFOO_VERSION` で固定されており、
  `npx promptfoo@<固定バージョン>` として実行される（`@latest` は使わない —
  サプライチェーン露出と結果の再現性ドリフト対策）。更新するときは固定値を上げて
  `evalloop doctor` と `run --limit` のスモークを通してからコミットする
- **Python 3.11+** と [uv](https://docs.astral.sh/uv/)
- **Ollama**（ローカル小型モデルを使う場合。`ollama pull qwen2.5:7b` など）
- `ANTHROPIC_API_KEY`（必須。実行・LLMジャッジ・GEPAのreflectionに使用）。
  `OPENAI_API_KEY` / `GEMINI_API_KEY` は使うモデルに応じて任意

## セットアップ

```bash
node --version                   # 上記レンジを満たすことを確認
uv sync
ollama pull qwen2.5:7b           # ローカルモデルを使う場合
export ANTHROPIC_API_KEY=...
uv run evalloop doctor           # Node/promptfoo/Ollama/APIキーの疎通確認。必ず最初に実行する
```

`doctor` は各providerに1件だけ極小のevalを流して疎通確認する（実際にAPIを呼ぶ）。

## タスクは自己完結のワークスペース

評価タスクはそれぞれ `tasks/<name>/` ディレクトリに自己完結して置かれる（issue #47）:
`task.yaml`（何をどう測るか）、`golden.jsonl`（データセット）、`prompts/task.txt`
（textタスクは `judge_rubric.txt` も）、および人手キュレーション物
（`human_labels.jsonl` / `taxonomy.yaml` / `notes.csv`）。生成物はタスク別サブツリー
（`data/build/<task>/`、`promptfoo/<task>/`、`results/<task>/`、`blog/<task>/`）に出る。
タスクの選択は各コマンドの `--task NAME`、環境変数 `EVALLOOP_TASK`、またはグローバル
`config.yaml` の `default_task`（モデルregistry・単価もここで一元管理）。
`evalloop task list` で一覧できる。

**データポリシー: タスクのデータは既定で gitignore。** 追跡されるのは `task.yaml` /
`prompts/` / `PROVENANCE.md`（出典と再取得手順）だけなので、私有データセットも安全に
ぶら下げられる。合成データの `sample-inquiry` デモタスクだけが例外的にオプトイン追跡
されており、default_task と CI スモークのデータ源になっている。

## クイックスタート

fresh clone のまま、同梱の `sample-inquiry` タスク（合成の問い合わせ4分類24件、
answer_type=label・決定的assert採点なのでLLMジャッジ不要。ローカルOllamaモデルなら
APIキーも不要）で動く:

```bash
uv run evalloop build --models qwen7b          # sample-inquiry が default_task
uv run evalloop run --limit 10                 # 先頭10件だけ試し打ち
uv run evalloop report <表示されたrun_id>       # モデル×精度×コストのMarkdown表を生成
uv run evalloop view                           # promptfooのローカルビューアで結果を見る
```

`--models qwen7b` を外せば registry の全モデルで評価する（`ANTHROPIC_API_KEY` が必要）。

CUAD-100 契約条項抽出タスク（answer_type=text、llm-rubricジャッジ）は `tasks/cuad100/` に
定義済みだがデータは同梱されない — 取得手順は [tasks/cuad100/PROVENANCE.md](tasks/cuad100/PROVENANCE.md)
を参照（CUAD v1・CC BY 4.0）。取得後:

```bash
uv run evalloop build --task cuad100 --allow-same-judge
uv run evalloop run --task cuad100 --limit 10
```

> `--allow-same-judge` が必要なのは、`tasks/cuad100/task.yaml` がjudge（sonnet46と同一
> provider）を評価対象5モデルの中に含めているため（sonnet46の行だけ自己採点になる既知の
> トレードオフ）。judgeを評価対象外のモデルにすれば不要になる。

問題なければ `--limit` を外してフルセットで実行し、失敗分析・改善ループ・ブログ出力に進める。

```bash
uv run evalloop run                                        # フルセットで実行
uv run evalloop failures <run_id>                          # 失敗ケースを抽出、data/notes.csvにメモ欄を追加
#   -> data/notes.csv の note 列に人手で失敗理由を書き込む
uv run evalloop cluster                                    # LLMがカテゴリ案をdata/taxonomy.draft.yamlに提案
#   -> 内容を確認し、data/taxonomy.yaml として保存（draftは自動では上書きしない）
uv run evalloop pivot <run_id>                              # 失敗カテゴリ×モデルのクロス集計
uv run evalloop calibrate --run-id <run_id>                 # LLMジャッジと人手ラベルの一致率を確認
uv run evalloop optimize                                    # dspy（GEPA / MIPROv2 / COPRO）でプロンプトを改善（train splitのみ使用）
#   -> task.yaml の optimize.method で手法を選択（未設定=gepa）。最適化後、自動でrun/report/compare(直近のベースrunがあれば)まで実行される
#   ※ いずれの手法も学習は決定的な代理メトリクス（textタスクはトークンF1）で行い、最終評価はllm-rubricのまま（既知の制約参照）
#   ※ どの失敗症状にどの最適化手法を当てるかは docs/APO_GUIDE.md（症状→粒度→手法の診断ガイド）を参照
uv run evalloop blog --runs <run_id>                        # ブログ用の図表・記事ドラフトをblog/に出力
```

## 自分のタスクを追加する

タスクの追加は既存タスクに一切触れない。`uv run evalloop task init <name>` で雛形
（task.yaml・prompts/・PROVENANCE.md のテンプレート）が生成されるので、埋めるのは実質3ファイル:

1. `tasks/<name>/task.yaml` — answer_type・ラベル・judge/optimize設定・使用モデルのalias選択（既存タスクのtask.yamlをテンプレートにコピー）
2. `tasks/<name>/golden.jsonl` — 評価データセット（唯一のソース。フォーマットは [docs/DESIGN.md#5-データ仕様](docs/DESIGN.md#5-データ仕様)）。既定でgitignoreされるため、出典を書いた `PROVENANCE.md` を添えること
3. `tasks/<name>/prompts/task.txt` — `{{input}}` プレースホルダを含むベースプロンプト（textタスクは `prompts/judge_rubric.txt` も）

あとは `--task <name>` で全コマンドが動く。モデル定義（provider ID・単価・
`supports_sampling_params`）はグローバル `config.yaml` の registry に1箇所だけ置き、
タスク側は alias で選択する。`models[].provider` にはpromptfooの表記（例:
`anthropic:messages:claude-...`, `ollama:chat:qwen2.5:7b`）、`optimize.reflection_provider`
にはdspy/litellmの表記（例: `anthropic/claude-...`）を使う。**書式が異なる**ので注意。
単価・provider IDはあくまでサンプル値なので、`doctor` が通らないIDは使わず、単価は
使用時点の公式価格に更新すること。

> **samplingパラメータを受け付けないモデルに注意**: `claude-opus-4-8` や `claude-fable-5` は
> `temperature` 等のsamplingパラメータの指定を **HTTP 400で拒否**する。該当モデルには
> `models[].supports_sampling_params: false` を設定すると、`evalloop build` が生成する
> promptfoo設定からtemperatureが省略される（`max_tokens` は全モデルで送信される）。
> 同梱の `config.yaml` ではopus48 / fable5に設定済み。
> なお `claude-fable-5` はthinkingが常時有効なモデルのため、レイテンシ・出力トークン量が
> 他モデルより大きくなりうる（コスト概算・レイテンシ比較の解釈時に注意）。

## CLIコマンド一覧

| コマンド | 説明 |
|---|---|
| `evalloop doctor` | Node/promptfoo/Ollama/APIキーの疎通確認 |
| `evalloop build [--allow-same-judge] [--yes]` | golden.jsonl→promptfoo設定を生成、実行前コスト概算を表示 |
| `evalloop run [--variant NAME] [--repeat N] [--limit N] [--no-cache]` | promptfoo evalを実行してresults/runs/{run_id}/に記録 |
| `evalloop view` | promptfooのローカルビューア（`promptfoo view`のパススルー） |
| `evalloop report RUN_ID` | モデル×精度×コスト×レイテンシのMarkdownレポート |
| `evalloop calibrate [--run-id ID]` | LLMジャッジとhuman_labels.jsonlの一致率を算出 |
| `evalloop failures RUN_ID` | 失敗ケース抽出、notes.csvにメモ欄を追記（冪等） |
| `evalloop cluster [--notes PATH]` | notes.csvからLLMが失敗タクソノミー案を生成 |
| `evalloop pivot RUN_ID` | 失敗カテゴリ×モデルのクロス集計 |
| `evalloop optimize` | dspy（GEPA / MIPROv2 / COPRO、task.yaml の `optimize.method` で選択）でプロンプト最適化、自動でrun/report/compare（手法選定は [docs/APO_GUIDE.md](docs/APO_GUIDE.md) 参照） |
| `evalloop compare --runs A,B` | 2つのrunのbefore/after比較 |
| `evalloop blog --runs A[,B] [--slug NAME]` | 公開ガード通過後にブログ用一式を生成 |

## テスト / CI

```bash
uv run pytest        # ユニットテスト（promptfoo/GEPAは全てモック。APIキー・Node不要）
uv run ruff check .  # リンタ（CIと同一チェック）
```

label正規化・train/test split分離・output.jsonパーサ・ブログ公開ガードなど、鉄の掟に関わる
ロジックは全てユニットテストでカバーされている（`tests/`）。テストは作業ツリーに何も
書き込まない（`tests/conftest.py` の `isolated_artifact_paths` フィクスチャ参照）。

CI（[.github/workflows/ci.yml](.github/workflows/ci.yml)）は push / PR ごとに
Ubuntu / Windows × Python 3.11 / 3.12 で pytest + ruff を実行する（macOSについては
ローカル実機での検証状態を下記に併記）。さらに master への
push 時、Actions secrets に `OLLAMA_API_KEY` が設定されていれば、Ollama Cloud で
3ケースの実スモーク（`--task sample-inquiry --models gptoss20b`: build → run → report）を
流す（未設定なら自動スキップ）。従量課金のAPIコストは発生しない。

## 実機検証: for mac / for windows

以下の2環境で `doctor`/`build`/`run`/`report`/`blog` を実際に動かして検証した。
モック化したユニットテストだけでは見つからなかった不具合も実機検証で拾い上げている。

### for mac

- macOS（Apple Silicon）+ Node.js + Ollama (qwen2.5:7b)
- `subprocess` 周り・パス解決はUnix系の標準挙動に乗るため、OS固有の不具合は特に見つかっていない
- CJKフォント検出→matplotlibの `font.family` 設定の不具合（後述）はmacOS上でも発見・修正済み
- 同梱の `sample-inquiry` と CUAD-100（データ取得後）の両タスクでフルパイプラインが動くことを確認済み

### for windows

Windows 11 + Node.js + Ollama (qwen2.5:7b) の実機で検証し、以下の不具合を修正した。

- **`subprocess.run(["npx", ...])` がWindowsで `FileNotFoundError`**: `npx` は実体が `npx.cmd`
  で、シェルを介さない`subprocess`はPATHEXT解決をしない。`shutil.which("npx")` で解決するよう修正
- **promptfooの実際のNode.js要件は `^20.20.0 || >=22.22.0`**: 21.x・22.0〜22.21.xは
  promptfoo自身が起動時にハードエラーで拒否する（`node --version` だけでは分からない）
- **`subprocess.run(..., text=True)` がcp932(日本語windows既定コードページ)でクラッシュ**:
  promptfooの出力にcp932で表現できない文字が含まれると `UnicodeDecodeError` で
  読み取りスレッドが落ちる。`encoding="utf-8", errors="replace"` を明示して修正
- **`llm-rubric` の `value: file://...` は `{{input}}`/`{{expected}}` が置換されない**:
  実際のグレーディングプロンプトを確認したところ、file://参照のルーブリックはNunjucks
  テンプレート処理を通らず、プレースホルダが文字通りジャッジに渡っていた
  （インラインの`value`文字列は置換される）。ルーブリックファイルの中身を読み込んで
  インライン文字列として埋め込むよう修正（`build.py`・`calibrate.py`）
- CJKフォントを検出しても実際には`matplotlib`の`font.family`に設定していなかったため、
  日本語グラフラベルの文字化けを引き起こしていた不具合も修正済み（M5実装時に発見。
  mac・windows共通の不具合だがwindows実機検証中にも再確認した）

これらはCUAD-100タスク（下記）を実際に評価してみて初めて表面化した問題であり、
モックだけに頼ったテストの限界を示している。

## for windows + CUAD-100 での実地検証

`config.local-verify.yaml`（Ollama qwen2.5:7bのみ、APIキー不要）を使い、Windows実機上で
`build` → `run` → `report` → `blog` のフルパイプラインを実際に動かして検証した。
5件のサブセットでは判定ロジック（llm-rubricジャッジ）が意味のある合否判定を返すことを確認済み
（例: 「該当条項なし」という出力が、正解が実際に条項ありの場合はfail、正解も
「該当条項なし」の場合はpassと正しく判定される）。全80件（testスプリット）での
本実行はCPU律速のローカル推論のため長時間かかる（1件あたり実測約136秒 = 抽出+採点で
モデル呼び出し2回）。

## 生成物ポリシー（gitに追跡されないファイル）

`evalloop` の各コマンドが生成するファイルはすべてgitignoreされ、タスク別サブツリーに出る。
fresh clone後はクイックスタートの手順どおり `uv run evalloop build` を最初に実行すること。

| コマンド | 生成物（すべてgit非追跡、`<task>`=タスク名） |
|---|---|
| `evalloop build` | `data/build/<task>/`, `promptfoo/<task>/promptfooconfig.yaml` |
| `evalloop run` | `results/<task>/runs/{run_id}/`, `results/<task>/index.jsonl`（マシンローカルの監査台帳） |
| `evalloop report` | `results/<task>/reports/` |
| `evalloop failures` / `cluster` | `tasks/<task>/notes.csv`, `tasks/<task>/taxonomy.draft.yaml` |
| `evalloop optimize` | `promptfoo/<task>/variants/`（`tasks/<task>/optimized/` は実験成果物として任意にコミット可） |
| `evalloop blog` | `blog/<task>/` |

run成果物の生出力（output.json / meta.json）にはローカル絶対パスやプロバイダのエラー
ペイロードが含まれうるため、公開リポジトリにはコミットしない。タスクの**データ**
（`golden.jsonl` / `human_labels.jsonl` / `notes.csv` / `taxonomy*.yaml`）も上記データ
ポリシーにより既定でgitignore。追跡されるのはタスクの「コード」（`task.yaml` / `prompts/` /
`PROVENANCE.md`）とグローバル `config.yaml` のみ。

## データ出自

各タスクのデータ出典と再取得手順は `tasks/<name>/PROVENANCE.md` に記載する。これまで
同梱したデータはすべて公開データセット由来、または本プロジェクトのために創作した
合成データであり、**実在の顧客データ・業務データ・実際の問い合わせとは一切関係ない**。

- `tasks/sample-inquiry/`（追跡・オプトイン） — 問い合わせ4分類の**自作ダミー24件**
  （`meta.source: "self-made"`、一般的なSaaS問い合わせを模した創作文）と、ジャッジ校正
  デモ用の**合成フィクスチャ10件**（`output_raw` は架空のモデル出力）
- `tasks/cuad100/`（データ非追跡） — [CUAD v1](https://www.atticusprojectai.org/cuad)
  （The Atticus Project発行、**CC BY 4.0**）から抽出した100件のサブセット。取得元は
  Hugging Faceの `chenghao/cuad_qa` ミラー。ファイル指紋と復元手順は PROVENANCE.md 参照

## 既知の制約

- `evalloop optimize` は3つのanswer_typeすべてに対応し、3つの最適化手法
  （`optimize.method` で `gepa` / `miprov2` / `copro` を選択、未設定=gepa）を切り替えられる。
  いずれの手法も**学習メトリクスは決定的な代理指標であり、最終評価とは別物**:
  `label` はラベル一致ロジックの移植、`text`（現在アクティブなCUAD-100タスク等）は
  正解スパンとのSQuAD方式トークンF1、`json` はdeep-equalityの移植を使う。
  textタスクの最終評価（promptfoo側）は従来どおりllm-rubricジャッジのままなので、
  学習メトリクスと最終採点は乖離しうる — その乖離の計測自体が最適化ケーススタディの対象である
- 上記「学習メトリクスが代理指標である制約」はGEPA・MIPROv2・COPROすべて、および
  将来追加される他のオプティマイザ（OPRO・APE・EASE等）にも共通する。プロセス内で
  高速に評価を回すには構造化判定（ラベル一致・トークンF1・deep-equal等）が必要で、
  LLMジャッジを毎候補ロールアウトで呼ぶことは鉄の掟（Pythonからモデルproviderを
  直接呼ばない）上できない。よって「代理指標で学習し、最終評価は別指標で検証」は
  本ハーネスのAPO全体に共通する前提となる（手法選定は [docs/APO_GUIDE.md](docs/APO_GUIDE.md) 参照）
- ローカル小型モデル（qwen2.5:7b）をジャッジに使うと、まれに英語・日本語以外の言語で
  採点理由を返すなど、フロンティアモデルほど指示追従が安定しない。ジャッジには
  極力、評価対象より十分強いモデルを使うことを推奨（`config.yaml`本来の設計どおり）
- `tasks/cuad100/human_labels.jsonl` はCUAD-100タスクに対する実際の人手ラベルがまだ無いため
  意図的に空にしてある。同タスクで `evalloop calibrate` を使うには先に人手レビューが必要
  （`sample-inquiry` には校正デモ用の合成ラベル10件が同梱されている）

設計の背景・データ仕様・「鉄の掟」の詳細は [docs/DESIGN.md](docs/DESIGN.md) を参照。

## インストール方針

本プロジェクトはPyPIには公開していない。**git clone + `uv sync` で利用する前提**
（[セットアップ](#セットアップ)参照）。ソースツリー内のパス規約（`data/` `prompts/` `results/`等）に
アンカーした設計のため、site-packagesへのwheelインストールはサポートしない。

## ライセンス

[MIT License](LICENSE)。同梱データのライセンスは別途各ファイルの出典表記に従う
（各タスクの `tasks/<name>/PROVENANCE.md` にライセンスを明記する）。
