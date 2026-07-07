# sample-inquiry — データ出自

- `golden.jsonl` — 問い合わせ4分類の**自作ダミー24件**（train 12 / test 12、`meta.source: "self-made"`）。
  一般的なSaaS問い合わせを模した創作文で、実在の問い合わせの引用・改変ではない
- `human_labels.jsonl` — ジャッジ校正デモ用の**合成フィクスチャ10件**。
  `output_raw` は架空のモデル出力であり、実際のLLM実行結果ではない

すべて合成データのため、データポリシー（issue #47: タスクデータは既定で
gitignore）の**例外として git 追跡にオプトイン**している（.gitignore の
否定パターン）。fresh clone のクイックスタートと CI スモークのデータ源。
