# GEPA実験: glm52 (target) × glm52 (reflection) — 20260706-075752

## 設定
- task: cuad100 (CUAD-100 契約条項抽出, answer_type=text, llm-rubric judge)
- target_alias: glm52 (ollama:chat:glm-5.2:cloud)
- reflection_provider: ollama_chat/glm-5.2:cloud  (※task.yaml標準はanthropic/claude-opus-4-8。無課金で実行するため一時的にglm52に変更)
- auto: light
- 実行コマンド: `uv run evalloop optimize --task cuad100`
- GEPA反復: 26 iteration / 458 rollouts

## 評価結果（promptfoo llm-rubric judge, glm52自己採点のトレードオフあり）

| run | プロンプト | pass_rate | 失敗 | beyond_95ci vs base |
|---|---|---:|---:|:--|
| 20260705-221551-87f2 (A, base)   | tasks/cuad100/prompts/task.txt            | 81.2% | 15/80 | — |
| 20260706-075752-7e8e (B, optimized) | tasks/cuad100/optimized/glm52/20260706-075752/task.txt | 60.0% | 32/80 | **yes (有意悪化)** |

- delta: **−21.3%** (Wilson 95%CI非重複 = ノイズ枠超えの有意な悪化)
- actual cost: $0.0000 (Ollama cloud経由のため)

## 知見

1. **training metricとfinal judgeのdivergence顕在化**: GEPAはtoken-F1代理指標を最大化したが、llm-rubric judgeの最終評価とは逆相関した。optimize.py docstringが警告していた「training proxy ≠ final grading」の実例。
2. **過小抽出への振れ**: 最適化プロンプトは「最小スパン抽出・過剰抽出禁止」に振り切った結果、一部ケースで過小抽出/語のみ出力を誘発。
   - 例: case-0043 で実日付 `June 21, 1999` を抽出すべきところ `Effective Date` という語のみ出力。
   - 例: case-0060 で `"Liquidated Damages" has the meaning set forth in...` を抽出すべきところ `"Liquidated Damages"` という用語のみ出力。
3. **reflection品質**: reflection_lmもglm52に変更したため、opus-4-8を使う本来のGEPAポテンシャルより低い可能性あり。次回はANTHROPIC_API_KEYを設定してopus-4-8で再実行すべき。
4. **ベースラインプロンプトがGLM-5.2には適合**: 最適化プロンプトはGLM-5.2の挙動に合わず、元のprompts/task.txtを維持すべき。

## ケースレベル遷移分析（base 80件 ↔ optimized 80件）

| 遷移 | 件数 | 意味 |
|---|---:|---|
| 両方 pass | 46 | 最適化でも維持 |
| pass → **fail**（悪化） | **19** | 最適化が壊した |
| fail → pass（改善） | 2 | 最適化が直した |
| 両方 fail | 13 | どちらでも失敗 |

- ネット delta: **−17件**（15失敗 → 32失敗）
- 正しく動いていた19件を壊し、2件しか直せなかった。最適化は全体に有害。

### pass→fail の3パターン（19件の内訳）

**A. 語のみ出力への退化（過小抽出）**
最適化プロンプトの「最も短い句、語、または文のみを出力せよ」が、モデルに「見出し語や質問語だけ出力すればよい」と誤解させた。

| case | base出力（pass） | opt出力（fail） |
|---|---|---|
| case-0043 | `June 21, 1999` | `"Effective Date"` |
| case-0060 | `"Liquidated Damages" has the meaning set forth in Subsections 6.4.4 and 14.4.;...` | `"Liquidated Damages"` |
| case-0024 | `9.1.1 Scope of Grants. Subject to...` | （逆: optが改善側。短縮で通った唯一例） |

**B. 偽陰性への退化（過剰抽出禁止の副作用）**
「過剰抽出を絶対に避けよ」が、モデルに「該当条項なし」と過小申告させた。

| case | base出力（pass） | opt出力（fail） |
|---|---|---|
| case-0030 | `10. LIMITATION ON LIABILITY. EXCEPT IN THE EVENT...` | `該当条項なし` |

**C. 別条項への誤抽出（過剰抽出禁止が条項選択を随意化）**
短くしようとして関連条項の別の文を選んでしまった。

| case | base出力（pass） | opt出力（fail） | reason要旨 |
|---|---|---|---|
| case-0057 | Roche→FMIの許諾文 | i-Escrow→2TheMart方向 | 逆方向の許諾を抜いた |
| case-0038 | HoneywellのCovenant Not To Sue | （同じだが別caseで失敗） | 対象当事者が違う |
| case-0085 | Section 7.1(c) ROFO Purchase | Section 7.2(a) ROFR Sale Notice | 隣接セクションの別文 |

### 根本原因

`tasks/cuad100/optimized/glm52/20260706-075752/task.txt` に GEPA自身が書き込んだ以下の指示が元凶：

```
- カテゴリに直接該当する最も短い句、語、または文のみを出力してください
- 過剰抽出（前後の文の追加や関連条項の結合）は絶対に避けてください
```

GEPAはtoken F1（短いほどprecisionが上がる）を報酬として学習したため、「できるだけ短く抜け」という戦略をプロンプトに書き込んだ。だがCUADの本質は「意味のある条項テキストを抜く」ことで、短さとは矛盾する。token F1という代理指標がGEPAに誤った勾配を与えた。

### 改善案

1. **訓練指標の変更**: token F1ではなく、llm-rubricベースのmetricを使う（iron rule「PythonからAPI直接呼ばない」と要相談。プロセス内LLM judgeは禁止されている）
2. **reflection_lmをopus-4-8に戻す**: 今回はglm52に変更したためプロンプト生成の質が落ちている。ANTHROPIC_API_KEYを設定すれば本来のGEPAポテンシャルを測れる
3. **ベースプロンプト維持**: GLM-5.2には元の `prompts/task.txt` が適合。最適化しない方がマシという結果

## 関連ファイル
- 最適化プロンプト: `tasks/cuad100/optimized/glm52/20260706-075752/task.txt`
- GEPAメタログ: `tasks/cuad100/optimized/glm52/20260706-075752/optimize_log.json`
- variant config: `promptfoo/cuad100/variants/glm52_20260706-075752.yaml` (gitignore)
- run成果物: `results/cuad100/runs/20260706-075752-7e8e/` (gitignore)
- レポート: `results/cuad100/reports/20260706-075752-7e8e.md` (gitignore)
- 比較: `results/cuad100/reports/compare_20260705-221551-87f2_20260706-075752-7e8e.md` (gitignore)

## 失敗分析（base run 20260705-221551-87f2 の15件, tasks/cuad100/taxonomy.yaml）
| カテゴリ | 件数 |
|---|---:|
| 偽陰性（FN: 該当条項あり→「該当条項なし」誤答） | 6 |
| 誤った条項抽出（WE） | 7 |
| 部分抽出（PE） | 1 |
| 過剰抽出（OE） | 1 |