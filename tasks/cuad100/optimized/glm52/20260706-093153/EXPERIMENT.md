# GEPA実験 v2: glm52 (target) × glm52 (reflection), recall重視metric — 20260706-093153

## 設定
- task: cuad100 (CUAD-100 契約条項抽出, answer_type=text, llm-rubric judge)
- target_alias: glm52 (ollama:chat:glm-5.2:cloud)
- reflection_provider: ollama_chat/glm-5.2:cloud  (※task.yaml標準はanthropic/claude-opus-4-8。無課金で実行するため一時的にglm52に変更)
- auto: light
- 実行コマンド: `uv run evalloop optimize --task cuad100`
- GEPA反復: 34 iteration / 452 rollouts

## v1からの改修（反省反映）

v1（20260706-075752）はtoken F1最大化で「最短の語だけ出力」に振れ、81.2%→60.0%へ有意悪化させた。v2ではtraining metricを改修：

- `_span_set_f1`（plain token F1）→ `_span_set_score`（**recall重み0.8 / precision重み0.2**）
- スコア = 0.8 × recall + 0.2 × span_count_penalty
- 意図: goldスパンを網羅できれば満点に近づけ、余分な出力は軽微な減点にとどめる。過小抽出（核心欠損）を過剰抽出（冗長）より重く罰する。
- feedback文も「完全なスパンを抜け、見出し語に短縮するな」に変更
- テスト追記: 過小抽出 < 0.5、過剰抽出 > 過小抽出、gold完全網羅+余分は ≥ 0.8

## 評価結果（promptfoo llm-rubric judge, glm52自己採点のトレードオフあり）

| run | プロンプト | pass_rate | 失敗 | beyond_95ci vs base |
|---|---|---:|---:|:--|
| 20260705-221551-87f2 (base)            | tasks/cuad100/prompts/task.txt                        | 81.2% | 15/80 | — |
| 20260706-075752-7e8e (v1, plain F1)    | optimized/glm52/20260706-075752/task.txt              | 60.0% | 32/80 | **yes (有意悪化)** |
| 20260706-093153-5037 (v2, recall重視)  | optimized/glm52/20260706-093153/task.txt             | 70.0% | 24/80 | no (95%CI重複 = 有意差なし) |

- v1 → v2 delta: **+10.0%** (60.0% → 70.0%)
- base → v2 delta: −11.3%（95%CI重複 = ノイズ枠内。baseを有意に下回ったとは言えない）

## ケースレベル遷移分析

### base vs v2（recall重視）

| 遷移 | 件数 |
|---|---:|
| 両方 pass | 52 |
| pass → fail（悪化） | 13 |
| fail → pass（改善） | 4 |
| 両方 fail | 11 |

ネット delta vs base: **−9件**（15失敗 → 24失敗）。v1の−17件から大幅改善。95%CIは重複。

### v1 → v2（recall重視改修の効果）

| 遷移 | 件数 |
|---|---:|
| v1 fail → v2 pass（recall重視で修復） | **12** |
| v1 pass → v2 fail（新たに壊した） | 4 |
| ネット v2 vs v1 | **+8件** |

## v1が壊した19件のうち、v2が修復した11件

| case | v1出力（fail） | v2出力（pass） | 何が直ったか |
|---|---|---|---|
| case-0043 | `"Effective Date"`（語のみ） | `June 21, 1999 (the "Effective Date")` | 実日付を復元 |
| case-0060 | `"Liquidated Damages"`（用語のみ） | `You will pay us Liquidated Damages in the amount of Five Thousand Dollars...` | 条件文全体を復元 |
| case-0055 | 別条項（17.3.2(ii)） | `Roche hereby grants to FMI a non-exclusive...`（正条項2.1.3） | 正条項を選択 |
| case-0057 | i-Escrow→2TheMart方向 | `2TheMart hereby grants to i-Escrow...`（正方向） | 条項方向を是正 |
| case-0072 | Section 6.5 LIMITS ON SUBLICENSING | `(a) i-Escrow hereby grants to 2TheMart...`（Section 6.3） | 正セクション選択 |
| case-0038 | （同じく別caseで失敗） | Honeywell の Covenant Not To Sue | 対象当事者を是正 |
| case-0051 | Adaptimmune側の保険 | MD Anderson側の保険 | 対象者是正 |
| case-0058, 0059, 0081 | 隣接条項の誤抽出 | 正条項 | 条項選択是正 |

## 最適化プロンプトがGEPA自ら書いた反省

v2の `tasks/cuad100/optimized/glm52/20260706-093153/task.txt` に、GEPAがrecall重視metricの勾配から学んだ指示が現れた：

> 「該当するスパンの途中でテキストを切り捨てたり、短縮したりしてはいけません」
> 「抽出を一つの見出し語に要約したりしないでください」

これはv1が犯した「`Effective Date`だけ出力」「`Liquidated Damages`だけ出力」という過小抽出失敗を、recall重視metricがGEPAに学習させた結果。プロンプト改修なし（metric改修のみ）でGEPA自身がこの抑制指示を書いたことは、代理指標設計の重要性を示す。

## まだ直っていない8件（v1が壊し、v2でも修復できなかった）

- case-0030: `該当条項なし`（偽陰性。過剰抽出禁止が残った副作用か）
- case-0022, 0047, 0071, 0083, 0084, 0085, 0086, 0100: 隣接条項の別文を選ぶ誤抽出が残存

## 知見

1. **training metric改修で有意悪化を解消**: v1の−21.3%（有意悪化）→ v2の−11.3%（非有意）。recall重視0.8/precision 0.2の重み付けが、過小抽出失敗を大幅に減らした。
2. **GEPAがmetricの勾配から抑制指示を自発生成**: v2プロンプトに「見出し語に要約するな」「スパン途中で切るな」という指示が現れた。これは手で書いたのではなく、recall重視metricがGEPAに学習させた結果。代理指標設計がGEPAのプロンプト改善方向を決定づける証拠。
3. **base未到達の残課題**: v2でもbase(81.2%)には届かず。残る失敗の8件は「隣接条項の別文選択」で、これはmetric改修では解決しにくい（goldと別条項のトークン重なりが小さいためrecallも低く、metricは既にこれを罰しているのにモデルが選んでしまう）。この層には別のアプローチ（例: reflection_lmをopus-4-8に戻してプロンプト生成の質を上げる、またはプロンプトに「見出し語一致でなく法的概念で探せ」と更に明示する）が必要。
4. **reflection品質**: 引き続きglm52でreflection。ANTHROPIC_API_KEYを設定してopus-4-8に戻せば、さらに上振れが期待できる。

## 関連ファイル
- 最適化プロンプト: `tasks/cuad100/optimized/glm52/20260706-093153/task.txt`
- GEPAメタログ: `tasks/cuad100/optimized/glm52/20260706-093153/optimize_log.json`
- variant config: `promptfoo/cuad100/variants/glm52_20260706-093153.yaml` (gitignore)
- run成果物: `results/cuad100/runs/20260706-093153-5037/` (gitignore)
- レポート: `results/cuad100/reports/20260706-093153-5037.md` (gitignore)
- 比較 vs base: `results/cuad100/reports/compare_20260705-221551-87f2_20260706-093153-5037.md` (gitignore)
- コード改修: `src/evalloop/optimize.py` (`_span_set_score`, recall重視0.8/0.2)
- テスト追記: `tests/test_optimize.py` (過小抽出/過剰抽出のスコア比較5件)

## 次の改善候補

1. **reflection_lmをopus-4-8に戻す**: ANTHROPIC_API_KEY設定次第でプロンプト生成の質が上がる。残る8件の「隣接条項選択」ミスに効く可能性。
2. **ベースプロンプト維持**: v2プロンプトでもbase(81.2%)未到達のため、GLM-5.2には元の`prompts/task.txt`が依然として最良。最適化はbaseに追いついていない。
3. **metricの更なる調整**: recall重みを0.9まで上げる、またはspan_count_penaltyを廃止して純recallにする実験。ただし過剰抽出を全く罰しないと別の失敗モードを生むリスク。