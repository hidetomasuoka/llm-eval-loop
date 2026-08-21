# sample-agent — データ出自

- `golden.jsonl` — サポートエージェント軌跡の**自作ダミー16件**（train 8 / test 8、
  `meta.source: "self-made"`）。一般的なSaaS問い合わせを模した創作文で、
  実在の問い合わせの引用・改変ではない。期待軌跡は次の4意図:
  - 障害: `kb.search` → `ticket.create`、回答「インシデントを起票しました。」
  - 契約: `kb.search` のみ、回答「契約FAQを案内しました。」
  - 要望: ツールなし、回答「要望を記録しました。」
  - その他: ツールなし、回答「担当へ引き継ぎます。」

すべて合成データのため、データポリシー（issue #47: タスクデータは既定で
gitignore）の**例外として git 追跡にオプトイン**している（.gitignore の
否定パターン）。fresh clone で `answer_type=agent` + PROMST を動かすためのデータ源。
