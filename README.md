# llm-eval-loop

[![CI](https://github.com/hidetomasuoka/llm-eval-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/hidetomasuoka/llm-eval-loop/actions/workflows/ci.yml)

個人開発用のLLM評価ハーネス。1つのタスクについて、ローカル小型モデルからフロンティアモデルまで
同一条件で評価し、次の問いに答える。

> **どのモデルが必要精度を満たすか。それはいくら（コスト）で達成できるか。**

実行と判定は [promptfoo](https://www.promptfoo.dev/) に任せ、Python（`evalloop`）は
データセット管理・ジャッジ校正・失敗分析・[dspy](https://dspy.ai/) GEPAによるプロンプト最適化・
ブログ公開用エクスポートを担当する薄いグルーレイヤーとして実装されている。

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

## クイックスタート

`data/golden.jsonl`（現在アクティブなタスク）には CUAD v1（Contract Understanding Atticus
Dataset、The Atticus Project発行・CC BY 4.0）から抽出した契約条項抽出タスクが100件
（train 20 / test 80）入っている。answer_type=text で、LLMジャッジ（llm-rubric）を使って
「契約書の抜粋から、指定カテゴリの条項を正しく抜き出せているか」を採点する。
元のサンプルタスク（問い合わせ分類、answer_type=label）は `data/sample/` に保存されており、
`config.yaml` の `task.*` と `data/golden.jsonl`/`prompts/base/*.txt` を差し替えれば
いつでも戻せる（下記「自分のタスクに差し替える」参照）。

```bash
uv run evalloop build --allow-same-judge       # golden.jsonl -> promptfoo設定一式を生成
uv run evalloop run --limit 10                 # 先頭10件だけ試し打ち
uv run evalloop report <表示されたrun_id>       # モデル×精度×コストのMarkdown表を生成
uv run evalloop view                           # promptfooのローカルビューアで結果を見る
```

> `--allow-same-judge` が必要なのは、同梱の `config.yaml` がjudge（sonnet46と同一provider）を
> 評価対象5モデルの中に含めているため（sonnet46の行だけ自己採点になる既知のトレードオフ。
> `config.yaml` のjudgeコメント参照）。judgeを評価対象外のモデルにすれば不要になる。

> API キーなしで試す場合: `config.local-verify.yaml` は Ollama (qwen2.5:7b) だけで
> 実行・採点まで完結する検証専用configで、`ANTHROPIC_API_KEY` が無くても
> `uv run evalloop build --config config.local-verify.yaml --allow-same-judge` →
> `uv run evalloop run --config config.local-verify.yaml --limit 5` で
> パイプライン全体を試せる（judgeが評価対象と同一モデルになるため
> `--allow-same-judge` が必要。自己採点バイアスがある点に注意）。

問題なければ `--limit` を外してフルセットで実行し、失敗分析・改善ループ・ブログ出力に進める。

```bash
uv run evalloop run                                        # フルセットで実行
uv run evalloop failures <run_id>                          # 失敗ケースを抽出、data/notes.csvにメモ欄を追加
#   -> data/notes.csv の note 列に人手で失敗理由を書き込む
uv run evalloop cluster                                    # LLMがカテゴリ案をdata/taxonomy.draft.yamlに提案
#   -> 内容を確認し、data/taxonomy.yaml として保存（draftは自動では上書きしない）
uv run evalloop pivot <run_id>                              # 失敗カテゴリ×モデルのクロス集計
uv run evalloop calibrate --run-id <run_id>                 # LLMジャッジと人手ラベルの一致率を確認
uv run evalloop optimize                                    # dspy GEPAでプロンプトを改善（train splitのみ使用）
#   -> 最適化後、自動でrun/report/compare(直近のベースrunがあれば)まで実行される
#   ※ optimizeはanswer_type=labelのタスク専用。現在のCUAD-100(text)ではエラーになる（既知の制約参照）
uv run evalloop blog --runs <run_id>                        # ブログ用の図表・記事ドラフトをblog/に出力
```

## 自分のタスクに差し替える

触るのは次の3点だけで済む構造になっている。

1. `config.yaml` — タスク名・ラベル一覧・評価対象モデル・単価・ジャッジ設定
2. `data/golden.jsonl` — 評価データセット（唯一のソース。フォーマットは [docs/DESIGN.md#5-データ仕様](docs/DESIGN.md#5-データ仕様)）
3. `prompts/base/task.txt` — `{{input}}` プレースホルダを含むベースプロンプト

`config.yaml` の `models[].provider` にはpromptfooの表記（例: `anthropic:messages:claude-...`,
`ollama:chat:qwen2.5:7b`）、`optimize.reflection_provider` にはdspy/litellmの表記
（例: `anthropic/claude-...`）を使う。**書式が異なる**ので注意。単価・provider IDはあくまで
サンプル値なので、`doctor` が通らないIDは使わず、単価は使用時点の公式価格に更新すること。

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
| `evalloop optimize` | dspy GEPAでプロンプト最適化、自動でrun/report/compare |
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
Ubuntu / Windows × Python 3.11 / 3.12 で pytest + ruff を実行する。さらに master への
push 時、ランナー上の Ollama（qwen2.5:0.5b）で3ケースの実スモーク（build → run → report）を
流す。ローカルモデルなので APIキー・secrets 登録は不要、コストも0。

## Windows実地検証で見つかった問題と修正

Windows 11 + Node.js + Ollama (qwen2.5:7b) の実機で `doctor`/`build`/`run`/`report`/`blog` を
実際に動かして検証し、モック化したユニットテストだけでは見つからなかった以下の不具合を修正した。

- **`subprocess.run(["npx", ...])` がWindowsで `FileNotFoundError`**: `npx` は実体が `npx.cmd`
  で、シェルを介さない`subprocess`はPATHEXT解決をしない。`shutil.which("npx")` で解決するよう修正
- **promptfooの実際のNode.js要件は `^20.20.0 || >=22.22.0`**: 21.x・22.0〜22.21.xは
  promptfoo自身が起動時にハードエラーで拒否する（`node --version` だけでは分からない）
- **`subprocess.run(..., text=True)` がcp932(日本語Windows既定コードページ)でクラッシュ**:
  promptfooの出力にcp932で表現できない文字が含まれると `UnicodeDecodeError` で
  読み取りスレッドが落ちる。`encoding="utf-8", errors="replace"` を明示して修正
- **`llm-rubric` の `value: file://...` は `{{input}}`/`{{expected}}` が置換されない**:
  実際のグレーディングプロンプトを確認したところ、file://参照のルーブリックはNunjucks
  テンプレート処理を通らず、プレースホルダが文字通りジャッジに渡っていた
  （インラインの`value`文字列は置換される）。ルーブリックファイルの中身を読み込んで
  インライン文字列として埋め込むよう修正（`build.py`・`calibrate.py`）
- CJKフォントを検出しても実際には`matplotlib`の`font.family`に設定していなかったため、
  日本語グラフラベルの文字化けを引き起こしていた不具合も修正済み（M5実装時に発見）

これらはCUAD-100タスク（下記）を実際に評価してみて初めて表面化した問題であり、
モックだけに頼ったテストの限界を示している。

## Windows + CUAD-100 での実地検証

`config.local-verify.yaml`（Ollama qwen2.5:7bのみ、APIキー不要）を使い、Windows実機上で
`build` → `run` → `report` → `blog` のフルパイプラインを実際に動かして検証した。
5件のサブセットでは判定ロジック（llm-rubricジャッジ）が意味のある合否判定を返すことを確認済み
（例: 「該当条項なし」という出力が、正解が実際に条項ありの場合はfail、正解も
「該当条項なし」の場合はpassと正しく判定される）。全80件（testスプリット）での
本実行はCPU律速のローカル推論のため長時間かかる（1件あたり実測約136秒 = 抽出+採点で
モデル呼び出し2回）。

## 生成物ポリシー（gitに追跡されないファイル）

`evalloop` の各コマンドが生成するファイルはすべてgitignoreされており、リポジトリには含まれない。
fresh clone後はクイックスタートの手順どおり `uv run evalloop build` を最初に実行して
`promptfoo/promptfooconfig.yaml` と `data/build/` を生成すること。

| コマンド | 生成物（すべてgit非追跡） |
|---|---|
| `evalloop build` | `data/build/`, `promptfoo/promptfooconfig.yaml` |
| `evalloop run` | `results/runs/{run_id}/`, `results/index.jsonl`（マシンローカルの監査台帳） |
| `evalloop report` | `results/reports/` |
| `evalloop failures` / `cluster` | `data/notes.csv`, `data/failures.jsonl`, `data/taxonomy.draft.yaml` |
| `evalloop optimize` | `promptfoo/variants/`（`prompts/optimized/` は実験成果物として任意にコミット可） |
| `evalloop blog` | `blog/` |

run成果物の生出力（output.json / meta.json）にはローカル絶対パスやプロバイダのエラー
ペイロードが含まれうるため、公開リポジトリにはコミットしない。人手でキュレーションする
ファイル（`data/golden.jsonl`, `data/human_labels.jsonl`, `data/taxonomy.yaml`, `prompts/base/`,
`config.yaml`）は通常どおり追跡対象。

## データ出自

同梱データはすべて公開データセット由来、または本プロジェクトのために創作した合成データであり、
**実在の顧客データ・業務データ・実際の問い合わせとは一切関係ない**。

- `data/golden.jsonl` — [CUAD v1](https://www.atticusprojectai.org/cuad)（The Atticus Project発行、
  **CC BY 4.0**）から抽出した100件のサブセット。取得元は
  Hugging Faceの `chenghao/cuad_qa` ミラー（`config.yaml` の `blog.allowed_sources` に出典表記あり）
- `data/human_labels.jsonl` — 現在は意図的に空（CUAD-100への人手レビュー未実施のため。既知の制約参照）
- `data/sample/golden.jsonl` — 旧サンプルタスク（問い合わせ4分類）の**自作ダミー20件**
  （`meta.source: "self-made"`）。一般的なSaaS問い合わせを模した創作文で、実在の問い合わせの
  引用・改変ではない
- `data/sample/human_labels.jsonl` — ジャッジ校正デモ用の**合成フィクスチャ10件**。
  `output_raw` は架空のモデル出力であり、実際のLLM実行結果ではない

## 既知の制約

- `evalloop optimize` は現状 `task.answer_type == "label"` のタスクのみ対応
  （GEPAのmetricがラベル一致ロジックの移植版のみのため）。現在アクティブなCUAD-100タスクは
  `answer_type=text` なので、`optimize` を試すには `data/sample/` のラベル分類タスクに
  一時的に戻すか、text用のmetricを追加実装する必要がある
- ローカル小型モデル（qwen2.5:7b）をジャッジに使うと、まれに英語・日本語以外の言語で
  採点理由を返すなど、フロンティアモデルほど指示追従が安定しない。ジャッジには
  極力、評価対象より十分強いモデルを使うことを推奨（`config.yaml`本来の設計どおり）
- `data/human_labels.jsonl` はCUAD-100タスクに対する実際の人手ラベルがまだ無いため
  意図的に空にしてある。`evalloop calibrate` を使うには先に人手レビューが必要

設計の背景・データ仕様・「鉄の掟」の詳細は [docs/DESIGN.md](docs/DESIGN.md) を参照。

## インストール方針

本プロジェクトはPyPIには公開していない。**git clone + `uv sync` で利用する前提**
（[セットアップ](#セットアップ)参照）。ソースツリー内のパス規約（`data/` `prompts/` `results/`等）に
アンカーした設計のため、site-packagesへのwheelインストールはサポートしない。

## ライセンス

[MIT License](LICENSE)。同梱データのライセンスは別途各ファイルの出典表記に従う
（現在の `data/golden.jsonl` はCUAD v1由来・CC BY 4.0）。
