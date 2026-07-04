# llm-eval-loop

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

## クイックスタート（同梱のサンプルタスクで一通り動かす）

サンプルタスクは「日本語の問い合わせ文を `契約照会 / 障害報告 / 機能要望 / その他` に分類する」
というラベル分類タスクで、`data/golden.jsonl`（20件, train 8 / test 12）と
`data/human_labels.jsonl` に最初から入っている。

```bash
uv run evalloop build                          # golden.jsonl -> promptfoo設定一式を生成
uv run evalloop run --limit 10                 # 先頭10件だけ試し打ち
uv run evalloop report <表示されたrun_id>       # モデル×精度×コストのMarkdown表を生成
uv run evalloop view                           # promptfooのローカルビューアで結果を見る
```

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

## テスト

```bash
uv run pytest
```

label正規化・train/test split分離・output.jsonパーサ・ブログ公開ガードなど、鉄の掟に関わる
ロジックは全てユニットテストでカバーされている（`tests/`）。

## 既知の制約

- 開発時のサンドボックス環境にはNode.js・有効なAPIキー・Ollamaの実行環境がなかったため、
  実際の `npx promptfoo eval` 呼び出しと dspy GEPA の実 API 呼び出し部分は、subprocess/API
  境界をモック化したユニットテストでのみ検証している。実機での `evalloop doctor` 実行が
  最初の実地検証になる
- `evalloop optimize` は現状 `task.answer_type == "label"` のタスクのみ対応
  （GEPAのmetricがラベル一致ロジックの移植版のみのため）

設計の背景・データ仕様・「鉄の掟」の詳細は [docs/DESIGN.md](docs/DESIGN.md) を参照。
