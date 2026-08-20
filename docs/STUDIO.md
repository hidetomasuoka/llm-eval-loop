# evalloop studio — プロセス / データ / モデル管理

Dify（アプリ・ナレッジ・ワークフロー）、LangChain（チェーン／ツール／検索）、
DataRobot（データプロファイリング・AutoML・リーダーボード・デプロイ）に相当する
**ローカル**管理面。評価ハーネス本体（promptfoo / GEPA）は変えず、その上に乗る。

鉄の掟は維持する。studio の LLM ステップは `echo` / `template` / `passthrough` だけ。
ホストされたモデル呼び出しはこれまで通り `evalloop build` / `run`（promptfoo）に任せる。
Python が Anthropic / OpenAI / Gemini SDK を直接叩くことはない。

## 何ができるか

| 層 | 相当 | できること |
|---|---|---|
| データ | DataRobot + Dify knowledge | CSV/JSONL import、カラム型推定、欠損・分布の profile、evalloop タスクの golden を取り込む |
| ナレッジ | Dify | ドキュメント登録と TF-IDF 検索（RAG の retrieve ステップ） |
| モデル | DataRobot + config.yaml registry | LLM provider のカタログ化、AutoML 学習、リーダーボード、winner のデプロイ、predict |
| プロセス | LangChain | YAML でステップ列（prompt / retrieve / classify / branch / switch / map / tool / expr / llm） |
| アプリ | Dify | プロセス＋ナレッジ＋モデルを束ね、CLI とローカル HTTP で実行 |

## クイックスタート

```bash
uv run evalloop studio seed
uv run evalloop studio status
uv run evalloop studio app run inquiry-bot --input "ログインできません。パスワードが通りません。"
uv run evalloop studio train leaderboard job-churn-seed
uv run evalloop studio serve          # http://127.0.0.1:8787
```

`seed` は次を作る（API キー不要）:

- データセット `inquiry`（`tasks/sample-inquiry/golden.jsonl`）と `churn-toy`（合成の解約テーブル）
- ナレッジ `inquiry-faq`
- AutoML ジョブ（majority / naive_bayes / decision_tree / logreg / knn。回帰なら linreg）
- 安定 alias `inquiry-clf` / `churn-clf`
- プロセス `inquiry-triage` と `churn-risk`
- アプリ `inquiry-bot` と `churn-watch`
- `config.yaml` の LLM registry（参照用。studio からは呼ばない）

成果物は `studio/`（gitignore）。消しても `seed` で再生成できる。

## CLI

```
evalloop studio init
evalloop studio seed
evalloop studio status
evalloop studio serve [--host 127.0.0.1] [--port 8787]

evalloop studio data import FILE --id NAME
evalloop studio data from-task TASK [--id NAME]
evalloop studio data list | show ID

evalloop studio knowledge list | search ID QUERY
evalloop studio knowledge import FILE.md --id NAME [--append]

evalloop studio model list [--kind trained]
evalloop studio model from-registry
evalloop studio model predict ID --json '{"input":"..."}'

evalloop studio process list | show ID | graph ID | run ID --input TEXT
evalloop studio process run ID --dataset inquiry --limit 10

evalloop studio app list | run ID --input TEXT
evalloop studio app chat APP_ID "メッセージ" [--session ID]
evalloop studio train start DATASET --target COL
evalloop studio train leaderboard JOB_ID
evalloop studio train compare JOB_A,JOB_B
evalloop studio train score MODEL --dataset NAME
```

どのコマンドも `--root DIR` でワークスペースを差し替えられる（テストは隔離ディレクトリを使う）。

## プロセス YAML

```yaml
id: inquiry-triage
inputs: [{name: input, type: string}]
outputs: [label, reply, answer]
steps:
  - id: label
    kind: classify
    model: inquiry-clf
    feature: input
  - id: passages
    kind: retrieve
    knowledge: inquiry-faq
    query: "{{input}}"
    k: 2
  - id: reply
    kind: branch
    on: "{{label}}"
    cases:
      障害報告: 至急対応が必要です。
    default: 受付しました。
  - id: answer
    kind: llm
    provider: template
    template: "[{{label}}] {{reply}}"
```

ステップ種別: `prompt` `set` `transform` `retrieve` `classify` `predict` `llm` `tool` `branch` `switch` `map` `expr` `parse` `memory` `subprocess` `agent`。

`expr` は AST 許可リスト（算術・比較・添字・`len`/`min`/`max` 等）のみ。`import` や属性アクセスは拒否する。

## AutoML

`train start` は既存の `split` 列があればそれを使い、なければ target で層化して holdout する。
候補を holdout 指標で並べ、勝者を `deployed` モデルとして登録する。依存ライブラリは増やさない
（純 Python。sklearn なし）。

分類指標: accuracy / macro-F1。回帰: MAE / RMSE / R²。
holdout に加えて train 上の層化 k-fold（既定 3）、混同行列、特徴重要度、データ品質フラグ
（欠損・定数列・ターゲットリーク）を `jobs/<id>/diagnostics.json` に残す。
`train compare` はジョブ間のアルゴリズム別スコア、`train score` はデプロイ済みモデルのバッチ推論。

## 会話 / アプリ API

`evalloop studio app chat inquiry-bot "ログインできません"` はセッションを作り、
2回目以降 `--session ID` で履歴を `history` / `history_text` としてプロセスへ渡す。

HTTP:

- `POST /api/apps/<id>/chat`  `{"message":"...","session_id":"..."}`
- `POST /v1/chat/completions`  OpenAI 互換（`model` はアプリ id）
- `GET /api/processes/<id>/graph`  Mermaid DAG
- `GET /api/jobs/<id>`  リーダーボード + 診断

## HTTP

`evalloop studio serve` はループバックのダッシュボードと JSON API を出す。`promptfoo share`
相当のアップロードは無い。

- `GET /` ダッシュボード
- `GET /api/catalog`
- `POST /api/apps/<id>/run`  `{"inputs": {...}}`
- `POST /api/processes/<id>/run`
- `POST /api/models/<id>/predict`
- `POST /api/knowledge/<id>/search`
