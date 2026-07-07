# GEPA最適化パイプライン改善計画（cuad100実験の振り返り）

作成: 2026-07-06 / 対象実験: `tasks/cuad100/optimized/glm52/`（v1: 20260706-075752, v2: 20260706-093153）

## 結果サマリ

| run | metric | pass_rate | vs base |
|---|---|---:|---|
| base (20260705-221551-87f2) | — | 81.2% | — |
| v1 (plain token F1) | F1最大化 | 60.0% | −21.3%（有意悪化） |
| v2 (recall 0.8 / precision 0.2) | recall重視 | 70.0% | −11.3%（有意差なし・未検定） |

## 主要な知見

1. **GEPAは代理指標を忠実にプロンプト化する**。metricだけ変えたら（プロンプト手修正なし）、GEPAが自発的に「見出し語に要約するな」「スパン途中で切るな」を書いた。GEPAの出力品質 ≒ 代理指標の設計品質。
2. **v1→v2の+10ptは未検定**。有意判定は vs base のみで、v1↔v2間の paired 検定（McNemar）は未実施。遷移表 b=12, c=4 なら p≈0.077 で、metric改修の効果もまだ有意とは言えない水準。n=80 は15pt級の差しか検出できない。
3. **残存失敗はプロンプト最適化の射程外**。失敗は3層に分離できる:
   - 過小抽出 → v2のrecall重視で解消済み（metricで解ける層）
   - 偽陰性（FN）→ 訓練データにネガティブ例が0件のため学習不能
   - 隣接条項の誤選択（WE）→ モデルの読解力の問題。baseの失敗15件（WE7+FN6）とv2の残存失敗はほぼ同じ壁
4. **実務判断**: 3 run とも base 未達。GLM-5.2 には base プロンプト維持が現時点の正解。
5. **judge の信頼性が未検証**。judge = 被評価モデルと同一の glm-5.2（自己採点）、human_labels.jsonl は空、notes.csv に judge 誤判定の記録あり（case-0024）。81.2% という基準値自体の誤差が不明。

## 構造診断: 仕組みの骨格は良い。欠けているのは3つのゲート

骨格（採点のpromptfoo一元化、train/test分離の二重チェック、calibrate機構、iron rules）は健全。

### ゲート1: GEPA内部の選抜が訓練データで自己採点（optimize.py:467）

```python
return optimizer.compile(student=student, trainset=trainset)  # valsetなし
```

valset を渡していないため、train 20件で訓練し同じ20件で候補選抜 → 過学習が構造的に発生。「trainでは最良だがtestで悪化」は必然。

**改善**: train を train/val に割って `valset=` を渡す。根本的には train 拡充（後述）とセット。

### ゲート2: 出荷ゲートがない（optimize.py:542-550）

GEPA出力を無条件に variant 化し test 80件で run するだけ。「baseに勝ったときだけ採用」の判定が存在せず、実験のたびに test を消費（メタ過学習リスク）。

**改善**: 3-way split（train/dev/test）にし、optimize の自動 run は dev のみ。dev で base を McNemar 有意に上回った場合だけ promoted フラグ。test は最終確認1回に温存。`OptimizeOutcome` に `promoted: bool` を追加。

### ゲート3: calibrate が実装済みなのに未使用

calibrate.py は完成度が高い（echo provider 再生、agreement計測、iron rule #6 警告まで実装済み）が、cuad100 の human_labels.jsonl が空で一度も発火していない。judge の自己採点バイアスも未測定。

**改善**: base run の pass/fail 境界ケース中心に30件へ人手ラベル → `evalloop calibrate` → agreement < 0.85 なら judge を別プロバイダへ。これが済むまで GEPA 再実験は保留。

## metric の改修（text_score_and_feedback, optimize.py:285-321）

- **(a) verbatim 検証の追加**: 抽出タスクなのに「出力が原文の連続部分文字列か」を未チェック。言い換えでもトークン重複で点が入る。rubric は「要約・独自解釈は fail」なので明確なズレ。正規化後 substring 判定＋専用 feedback を追加（数行）。
- **(b) taxonomy 別 feedback**: 現状 WE でも PE でも同一の「COMPLETE clause を抜け」文言。overlap ほぼゼロ（別条項選択=WE）と overlap 高いが欠落（部分抽出=PE）はスコアで判別可能なので分岐し、WE には「見出し語一致でなくカテゴリの法的概念で特定せよ」を返す。taxonomy.yaml の定義を再利用。

## compare の検定強化（optimize.py:583-631）

Wilson CI 非重複は独立2標本向けで、同一80ケースの paired 設計では検出力を捨てている。per-case pass/fail は両 run から取得可能なので、遷移表から McNemar exact test（`math.comb` の二項検定で十分、scipy 不要、~20行）を計算し compare 表に列を追加。

## データ改修（tasks/cuad100）

- **ネガティブケース追加**: `NO_CLAUSE_ANSWER` の訓練分岐（optimize.py:297-310）は実装済みだが、100件全件「該当あり」のため一度も発火しない。プロンプトに「該当条項なし」の出口があるのに正解例0件 → base の FN 6件の根因。CUAD は契約×41カテゴリの直積で「該当なし」が大量にあり抽出容易。10〜15件追加。
- **train 拡充**: 20件・17カテゴリ → 40件程度、test とカテゴリ構成を揃える。

## 優先順位と工数

| # | 改修 | 対象 | 効果 | 工数 |
|---|---|---|---|---|
| 1 | human_labels 30件 + calibrate 実行 | データ作業 | 測定の土台 | 人手1〜2h |
| 2 | McNemar を compare に追加 | optimize.py | 全実験の判定精度 | 小 |
| 3 | verbatim 検証 + taxonomy別 feedback | optimize.py | GEPA の勾配品質 | 小 |
| 4 | dev split + 出荷ゲート | schemas/build/optimize | test 温存・改悪出荷防止 | 中 |
| 5 | ネガティブケース + train 拡充 | golden.jsonl | FN 層の解消 | 中 |
| 6 | GEPA に valset を渡す | optimize.py | 過学習抑制 | 小（5とセット） |

## やらなくていいこと

- **reflection の opus-4-8 化は優先度低**: 残存失敗の主力 WE はプロンプト文面でなくモデル読解力の問題。やるなら few-shot デモ付き最適化の方が筋がいい。
- **recall 重みの追加調整（0.9等）**: 過小抽出層は v2 でほぼ回収済みで効果の上限が見えている。

## 細かい指摘

- 最初の run（20260705-230558）は評価未実施のまま放置。破棄理由を1行残す。
- v2 プロンプトに v1 由来の過小抽出指示（「当事者は短いフレーズのみ抽出」）が残存。手で1行削って再評価するだけで数件拾える可能性（安価な追試として推奨）。
