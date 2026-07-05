# livedoor — データ出自と再取得手順

このタスクのデータ（`golden.jsonl`）は **git 管理外**（issue #47 のデータポリシー:
タスクデータは既定でコミット禁止）。

## 出典

- **Source**: livedoor ニュースコーパス（株式会社ロンウイット配布）
  - Distribution page: https://www.rondhuit.com/download.html#ldcc
  - Archive: `ldcc-20140209.tar.gz`
  - SHA-256 of the archive: `b17606ed8c670013a3809100a9e6104701baab62cc019abc262111bd2acf1063`
- **License**: クリエイティブ・コモンズ 表示 - 改変禁止 2.1 日本（CC BY-ND 2.1 JP）
  (http://creativecommons.org/licenses/by-nd/2.1/jp/) — 各カテゴリディレクトリの
  `LICENSE.txt` に明記。原著作者のクレジット表示と無改変を条件に記事全文の転載・引用が可能。
  記事本文をそのまま `golden.jsonl` に格納しているのはこのライセンス条件（無改変）を
  満たすため。**ブログ記事化する際は全文転載せず短い引用+出典リンクに留めること**
  （`task.yaml` の `blog.allowed_sources` によるガードで機械的に強制）。
- **Retrieved**: 2026-07-04（ユーザーがダウンロードしたアーカイブから抽出）

## サンプリング方法（再現用）

`text/<category>/` の9カテゴリ全件（各511〜901件）から、`random.seed(42)` で
各カテゴリ25件（train 10 / test 15）を無作為抽出:

```python
import random
random.seed(42)
files = sorted(category_dir.glob("*.txt"))  # LICENSE.txt除く
chosen = random.sample(files, 10 + 15)
train_files, test_files = chosen[:10], chosen[10:]
```

- **Categories** (9): dokujo-tsushin, it-life-hack, kaden-channel, livedoor-homme,
  movie-enter, peachy, smax, sports-watch, topic-news
- **Counts**: 225 total（90 train / 135 test; 各カテゴリ25件 = train 10 / test 15）
- **Fields**: `input` = 記事タイトル + 本文（URL・タイムスタンプ行は除去）。
  `meta.source` = `"livedoor-news-corpus-cc-by-nd"`。`meta.orig_filename` に
  元のファイル名を保持（トレーサビリティ用、evalloop本体では未使用）

## ファイル指紋（検証用）

- `golden.jsonl` sha256: `43c5f0d4ba1696227206f71b522a21c1898e398da3ea4ba0e7b4c166225ac31a`
  （2026-07-05 時点、225件）

## 再取得

1. 過去のブランチ `feature/livedoor-news-eval` の git 履歴に旧パス `data/golden.jsonl`
   として存在する（`git show origin/feature/livedoor-news-eval:data/golden.jsonl`）
2. 新規に作り直す場合: 上記アーカイブを取得し、サンプリング方法どおりに抽出
   （上記 sha256 と一致すれば同一データ）
