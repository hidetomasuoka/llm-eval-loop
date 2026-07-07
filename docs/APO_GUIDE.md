# APO 適用ガイド（症状 → 粒度 → 手法 の診断）

> **この文書について**: `evalloop optimize` は **GEPA / MIPROv2 / COPRO の3手法**に対応している（task.yaml の `optimize.method: gepa | miprov2 | copro` で選択、省略時 gepa。手法固有パラメータは `optimize.params`）。APO（Automatic Prompt Optimization）手法は「失敗症状 × 最適化粒度」で選ぶべきものであり、本ガイドは、どの症状にどの粒度のどの手法を当てるかの判断基準を整備する。
>
> **位置づけ**: 本ガイドは [docs/DESIGN.md](DESIGN.md) の「鉄の掟（第11章）」に従う。本文中で鉄の掟と矛盾する記述はないことが前提。コード変更を伴わない docs のみの追加。

---

## 1. 適用判断の3段階

APO に着手する前に、以下の3段階で「本当にAPOが必要か・どの粒度か・評価設計は取れるか」を確認する。

### ① プロンプト層が主因であることの確認

失敗の主因がプロンプト以外（検索・チャンク・パース・ツール選択・ワークフロー）にある場合は **APO を保留** し、先にそちらを直す。プロンプトを最適化しても根本原因が残るため改善しない。

| 主因 | 判断 | 対応 |
|---|---|---|
| 検索未ヒット（RAG が正文を取ってこない） | APO保留 | 検索パラメータ・embedding・チャンク境界を見直す |
| チャンク境界崩れ（正解スパンが複数チャンクに分断） | APO保留 | チャンクサイズ・オーバーラップ・再チャンク化 |
| パース欠損（構造化出力のJSON崩れ・フィールド欠落） | APO保留 | 本ガイド第3章「JSON安定化の優先順位」に従う |
| ツール誤選択（Agent が別ツールを呼ぶ） | APO保留 | ツール説明文・ルーティング設計を見直す |
| ワークフロー破綻（多段推論の途中で道筋が外れる） | APO保留 | ワークフロー設計・状態管理を見直す |
| **指示が曖昧・分類・抽出がぶれる** | **APO適用候補** | 本ガイド第2章で粒度を選ぶ |

### ② 最適化粒度の選定

プロンプト層が主因と判断できたら、第2章「診断マトリクス表」で症状に合う粒度（Instruction / Exemplar / 長文構造 / 多目的 / Agent・Multi-step）を選ぶ。粒度が違うと手法効果も変わるため、症状から逆算する。

### ③ 評価設計の確認

APO には **train / holdout 分割が取れる評価セット** が前提となる。鉄の掟 #1（split 分離はファイル分離で担保、`assert_split_disjoint` で assert）に従い、以下を満たすこと：

- **train split**: 最適化に使うケース群。GEPA等はこのみを読む（`optimize.py` は `split=='train'` のみ抽出し、test と ID が交差すると即異常終了）
- **holdout（test）split**: 最適化に使わず、最終評価のみに使う。ここでの改善確認が「汎化した」と言える唯一の証拠
- **評価指標**: **どの手法でも訓練メトリクスはプロセス内の代理指標（プロキシ）であり、最終評価（llm-rubric）とは別物**（`optimizers/metrics.py` の docstring 参照）。代理指標と最終評価の divergence は測定対象であり、隠さない。

train/holdout が取れない（評価セットが小さすぎる・ラベルがない）場合は、まず評価セット整備が先。APO は評価セットの上に成り立つ。

---

## 2. 診断マトリクス表

症状から最適化粒度・代表手法・evalloop の対応状況へのマッピング。

| 症状 | 粒度 | 代表手法 | evalloop対応 |
|---|---|---|---|
| 指示が曖昧で分類・抽出がぶれる | **7a. Instruction** | GEPA, COPRO, OPRO, APE, ProTeGi, PromptAgent | **対応済: GEPA / COPRO**（`optimize.method: gepa` / `copro`）。MIPROv2 も現状は instruction 提案として利用可 |
| 例の入れ替え・順序で性能がぶれる | **7b. Exemplar** | EASE, MIPROv2, PromptWizard | **一部対応: MIPROv2**（`optimize.method: miprov2` — ※現状 instruction のみ、demos ブートストラップは [APO-17] で解放予定） |
| 長いsystem promptの局所修正で別セクションが壊れる | **7c. 長文構造** | SCULPT | 対象外 |
| コスト・長さ制約が厳しい | **7d. 多目的** | InstOptima, EMO-Prompts | レポート可視化のみ計画 |
| Agent軌跡が破綻 | **7e. Agent/Multi-step** | PROMST | 対象外 |

### 各粒度の補足

- **7a. Instruction 粒度**: 指示文そのものを書き換える。GEPA は reflection LM に「この失敗を直すには指示をどう変えればよいか」を提案させ、train set で候補を評価し、パレートフロントに蓄積する進化的最適化。COPRO は breadth 個の候補生成 × depth 回の反復改善（coordinate ascent）。MIPROv2 はベイズ最適化で instruction 空間を探索する（`optimize.method` で選択）。
- **7b. Exemplar 粒度**: few-shot 例の選択・順序を最適化する。Instruction が完成していても例でぶれる場合はこちら。MIPROv2 が本来この粒度をカバーするが、evalloop では現状 demos ブートストラップを無効化して instruction のみ使っている（[APO-17] で解放予定）。
- **7c. 長文構造粒度**: system prompt が複数セクションから成り、一部を直すと別セクションが壊れる症状。SCULPT はセクション単位の局所編集を保持する。本プロジェクトのプロンプトは短いため対象外。
- **7d. 多目的粒度**: 精度以外にコスト・出力長・レイテンシを同時に最適化。InstOptima/EMO-Prompts はパレートフロントを複数目的で追跡する。evalloop は現状レポート可視化のみ計画（最適化自体は未対応）。
- **7e. Agent/Multi-step粒度**: Agent の多段推論軌跡全体を最適化。PROMST は軌跡の失敗点から改善する。本プロジェクトは単発QA前提のため対象外。

---

## 3. JSON安定化の優先順位

`answer_type=json` タスクで構造化出力が崩れる場合、**プロンプト最適化より先に以下の順で安定化を図る**。プロンプト最適化は③の位置づけで、①②が効かない場合に使う。

1. **API Structured Outputs**: provider がサポートする構造化出力機能（OpenAI の Structured Outputs / response_format、Anthropic の tool_use）を使う。スキーマ強制が最も安定。これが使えるならまず使う。
2. **Schema再設計（PARSE）**: スキーマ自体を簡素化・平坦化し、ネストを減らす。フィールド名をモデルに誤解されにくい形に変更する。PARSE 系の手法はスキーマ再設計を含む。
3. **few-shot / プロンプト最適化**: ①②でも崩れる場合、正例 few-shot や Instruction 粒度のプロンプト最適化（GEPA 等）で出力形式を安定させる。ただし最終評価は `evalloop run` の llm-rubric/deep-equal で行い、代理指標との divergence を測る。
4. **モデル変更・schema-aware評価**: より構造化出力に強いモデルに変更するか、評価側で schema-aware な許容幅（必須フィールドのみ厳格・オプションは緩和）を設ける。④は最後の手段。

---

## 4. 運用ルール7箇条

APO を適用・評価する際の運用上の前提。

1. **評価セットと分割の前提**: train/holdout 分割が取れる評価セットが必須。鉄の掟 #1（`assert_split_disjoint`）に従い、train と test の ID 交差は即異常終了。
2. **適用しやすい症状**: 指示が曖昧・分類がぶれる・抽出が安定しない（Instruction 粒度）。例の順序で性能がぶれる（Exemplar 粒度）。これらは APO の効きやすい症状。
3. **後回しにすべき症状**: 検索未ヒット・チャンク境界崩れ・パース欠損・ツール誤選択・ワークフロー破綻。プロンプト以外が主因の場合は APO より先に根本原因を直す。
4. **holdout側での改善確認**: train でのスコア上昇だけでは「汎化した」と言えない。holdout（test split）で改善が確認できて初めて採用。オプティマイザの train/val スコアと test での最終評価は別物（どの手法でも訓練メトリクスは代理指標であり、最終評価との divergence は測定対象）。
5. **1 prompt × 1 provider原則**: 1回の最適化は1プロンプト・1プロバイダで行う。複数プロバイダを同時に最適化すると、プロバイダ間の挙動差がどの指示変更によるものか分離できなくなる。
6. **本番失敗パターンを評価セットに含める**: 本番で失敗したケースは評価セットに追加し、回帰テスト可能な形にする。APO は評価セットの上に成り立つため、評価セットが本番を代表していないと改善が本番に効かない。
7. **「最強手法」断定の回避**: APO 手法の優劣は**条件依存**（タスク・モデル・データサイズ・評価指標）。あるタスクで GEPA が勝っても別タスクで OPRO が勝りうる。「最強手法」を断定せず、症状と粒度で選ぶ。

---

## 5. Soft Prompt / PEFT系は対象外

Soft Prompt（Prefix-Tuning 等）や PEFT（LoRA 等）は本プロジェクトの対象外とする。

### 理由

- **black-box API前提**: 本プロジェクトは frontier モデルを API 経由で使う。Soft Prompt / PEFT はモデル重みや埋め込み層にアクセスする必要があり、black-box API では適用できない。
- **可読性**: プロンプト最適化（Instruction 粒度）は最適化結果が人間可読な指示文として出力され、レビュー・編集できる。Soft Prompt は連続ベクトルであり可読性がない。本プロジェクトは「最適化結果を人間が確認して採用する」運用を前提とするため、可読性は必須。
- **再現性**: プロンプトはテキストとして commit 可能。Soft Prompt はバイナリで、provider・モデル版が変わると再現しない。鉄の掟（再現性・dataset-version hash による追跡）と相性が悪い。

### 範囲外の具体例

- Prefix-Tuning / Prompt-Tuning（連続ベクトルの学習）
- LoRA / QLoRA / 任意の PEFT（重みの部分更新）
- モデル蒸留（出力分布の模倣）

これらはモデル提供者が自前で学習する領域であり、API利用者の最適化対象ではない。

---

## 参考

- [docs/DESIGN.md](DESIGN.md) — 設計ドキュメント・鉄の掟（第11章）
- `src/evalloop/optimize.py` — オーケストレーション（手法選択・variant生成・run/report/compare）
- `src/evalloop/optimizers/` — 手法実装（gepa.py / miprov2.py / copro.py）と共有の代理メトリクス（metrics.py — 代理指標のdocstring）
- Issue #60 — 本ガイドの作成指示
- APO 計画全22件 — [APO-xx] で参照される依存関係