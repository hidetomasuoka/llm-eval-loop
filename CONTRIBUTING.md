# Contributing to llm-eval-loop

*日本語の要約は[下記](#日本語要約)にあります。*

## Project status & expectations

llm-eval-loop is a **personal project**, maintained on a best-effort basis.

- **Issues are welcome** — bug reports, questions, and ideas are all fine. Japanese or English.
- **Pull requests are welcome**, but review is best-effort and may take time.
- **For larger changes, please open an issue first** so we can discuss the approach before you invest work. The project has strict design rules (see below) and a PR that conflicts with them will not be merged, however well-implemented.
- There is no support SLA. If you need something fixed on a schedule, forking is a legitimate option (MIT license).

## Development setup

Requirements (see [README](README.md#必要環境) for details):

- **Node.js** `^20.20.0` or `>=22.22.0` — promptfoo itself hard-rejects versions outside this range
- **Python 3.11+** and [uv](https://docs.astral.sh/uv/)
- **Ollama** — optional, only for running local models
- `ANTHROPIC_API_KEY` — only needed for real eval runs. **Not needed for development**: the test suite mocks all promptfoo/GEPA calls

```bash
git clone https://github.com/hidetomasuoka/llm-eval-loop.git
cd llm-eval-loop
uv sync --extra dev
```

## Running checks (same as CI)

```bash
uv run pytest        # unit tests -- no API keys, no Node required
uv run ruff check .  # linter
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs exactly these on Ubuntu and Windows × Python 3.11/3.12 for every push and PR. Please make sure both pass locally before opening a PR.

### Test hygiene rules

- Tests must **never write into the checkout**. If your test exercises the real build/run/report/optimize orchestration, take the `isolated_artifact_paths` fixture from [tests/conftest.py](tests/conftest.py) — it redirects every artifact path to a throwaway tree.
- Tests must not depend on generated files (`promptfoo/promptfooconfig.yaml`, `data/build/`) existing — CI runs from a fresh clone.
- `uv run pytest` must pass twice in a row (idempotency).

## Design rules

The project's architecture and its non-negotiable rules ("iron rules") are documented in [docs/DESIGN.md](docs/DESIGN.md). The ones contributors most often hit:

- **Python never calls a model provider directly** — every LLM call goes through promptfoo (even the judge, taxonomy clustering, and calibration).
- **train/test split separation**: `data/build/tests_train.yaml` must never be referenced by any promptfoo eval config. GEPA optimization reads `split=="train"` only.
- **`results/` is an append-only ledger** — nothing deletes or rewrites a prior run.
- **`promptfoo share` is never used** (no cloud upload of results).
- **Label normalization exists in two implementations** (`src/evalloop/asserts/label_match.js` and `optimize.py`). If you touch either, extend the shared fixture `tests/fixtures/label_normalization_cases.json` — a test pins both to it.

## Language

Code comments and documentation are currently a mix of Japanese and English. Either is fine in contributions; pick whichever expresses the constraint more precisely.

---

## 日本語要約

- 本プロジェクトは**個人プロジェクト**であり、対応はベストエフォートです。issue（バグ報告・質問・提案）は日英どちらでも歓迎します
- PR も歓迎しますが、**大きな変更は先に issue で相談**してください。設計上の「鉄の掟」（[docs/DESIGN.md](docs/DESIGN.md)）に反する PR は実装品質に関わらずマージできません
- 開発セットアップ: `uv sync --extra dev`（テストは promptfoo/GEPA を全てモックしているため **API キー・Node 不要**）
- PR 前チェック: `uv run pytest` と `uv run ruff check .`（CI と同一。Ubuntu/Windows × Python 3.11/3.12 で自動実行されます）
- テストは作業ツリーに書き込まないこと（`tests/conftest.py` の `isolated_artifact_paths` フィクスチャを使用）
