# sample-process — データ出自

- `golden.jsonl` — 問い合わせ分類＋定型返信の**自作ダミー8件**（train 4 / test 4、
  `meta.source: "self-made"`）。実在の問い合わせの引用・改変ではない。
- `process.yaml` — 分類 LLM → if-else → テンプレート返信の合成デモグラフ。

すべて合成データのため、データポリシー（issue #47: タスクデータは既定で
gitignore）の**例外として git 追跡にオプトイン**している（`.gitignore` の
否定パターン）。`evalloop process validate --task sample-process` で DAG を確認できる。
