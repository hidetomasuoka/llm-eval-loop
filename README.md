# llm-eval-loop

[![CI](https://github.com/hidetomasuoka/llm-eval-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/hidetomasuoka/llm-eval-loop/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**English** | [日本語](README.ja.md)

> This README is kept in sync with the Japanese original ([README.ja.md](README.ja.md)). If the two ever diverge, the Japanese version is authoritative.

An LLM evaluation harness. It evaluates one task across models — from small local models to frontier models — under identical conditions, to answer a single question:

> **Which model meets the accuracy bar, and at what cost?**

Execution and grading are delegated to [promptfoo](https://www.promptfoo.dev/); the Python layer (`evalloop`) is a thin glue layer that owns dataset management, judge calibration, failure analysis, prompt optimization with [dspy](https://dspy.ai/) GEPA, and publish-guarded blog export.

This is an experimental personal project, but it is kept in a state you can `git clone` and use as-is (see the verification sections below for what has actually been exercised on real machines and in CI). Issues and PRs are welcome — support is best-effort. See [CONTRIBUTING.md](CONTRIBUTING.md) for expectations and the development workflow.

For the design rationale, data specifications, and the project's non-negotiable "iron rules", see [docs/DESIGN.md](docs/DESIGN.md) (full design doc, Japanese).

> **Policy**: this project never uses `promptfoo share` (cloud upload). Use `evalloop view` to browse local results.

## Requirements

- **Node.js**: `^20.20.0` or `>=22.22.0` (21.x and 22.0.0–22.21.x are NOT supported — promptfoo itself refuses to start outside this range; check `node --version` first)
- **promptfoo**: the version is pinned in `src/evalloop/run.py` (`PROMPTFOO_VERSION`) and executed as `npx promptfoo@<pinned version>`. `@latest` is never used — that would mean supply-chain exposure and reproducibility drift. To upgrade: bump the pin, pass `evalloop doctor` plus a `run --limit` smoke, then commit
- **Python 3.11+** and [uv](https://docs.astral.sh/uv/)
- **Ollama** (only if you use local models, e.g. `ollama pull qwen2.5:7b`)
- `ANTHROPIC_API_KEY` (required for real runs: evaluation, the LLM judge, and GEPA reflection). `OPENAI_API_KEY` / `GEMINI_API_KEY` are optional depending on which models you configure

## Setup

```bash
node --version                   # must satisfy the range above
uv sync
ollama pull qwen2.5:7b           # if using local models
export ANTHROPIC_API_KEY=...
uv run evalloop doctor           # connectivity check for Node/promptfoo/Ollama/API keys. Always run this first
```

`doctor` runs a single tiny eval against every configured provider (it calls real APIs).

## Quickstart

`data/golden.jsonl` (the currently active task) contains 100 contract-clause-extraction cases (train 20 / test 80) extracted from CUAD v1 (Contract Understanding Atticus Dataset, published by The Atticus Project, CC BY 4.0). It is an `answer_type=text` task graded by an LLM judge (llm-rubric): "did the model correctly extract the clause of the specified category from the contract excerpt?"
The previous sample task (inquiry classification, `answer_type=label`) is preserved under `data/sample/` and can be restored at any time by swapping `config.yaml`'s `task.*` together with `data/golden.jsonl` / `prompts/base/*.txt` (see "Bring your own task" below).

```bash
uv run evalloop build --allow-same-judge       # golden.jsonl -> full set of promptfoo configs
uv run evalloop run --limit 10                 # try just the first 10 cases
uv run evalloop report <run_id printed above>  # Markdown table: model x accuracy x cost
uv run evalloop view                           # browse results in promptfoo's local viewer
```

> `--allow-same-judge` is required because the bundled `config.yaml` includes the judge (same provider as sonnet46) among the 5 evaluated models — a known, documented tradeoff in which only the sonnet46 row is self-graded (see the judge comment in `config.yaml`). Point the judge at a model outside the evaluated set and the flag becomes unnecessary.

> Trying it without API keys: `config.local-verify.yaml` is a verification-only config that runs and grades entirely on Ollama (qwen2.5:7b). Without `ANTHROPIC_API_KEY`, you can exercise the whole pipeline with
> `uv run evalloop build --config config.local-verify.yaml --allow-same-judge` →
> `uv run evalloop run --config config.local-verify.yaml --limit 5`
> (`--allow-same-judge` is required because the judge is the same model being evaluated; mind the self-grading bias).

If everything looks good, drop `--limit` to run the full set, then continue into failure analysis, the improvement loop, and blog export.

```bash
uv run evalloop run                                        # full set
uv run evalloop failures <run_id>                          # extract failing cases, append note rows to data/notes.csv
#   -> fill in the `note` column by hand with failure reasons
uv run evalloop cluster                                    # an LLM drafts category proposals into data/taxonomy.draft.yaml
#   -> review, then save as data/taxonomy.yaml (the draft never overwrites it automatically)
uv run evalloop pivot <run_id>                              # failure-category x model cross-tab
uv run evalloop calibrate --run-id <run_id>                 # agreement rate between the LLM judge and human labels
uv run evalloop optimize                                    # improve the prompt with dspy GEPA (uses the train split only)
#   -> afterwards run/report/compare (against the latest base run, if any) execute automatically
#   NOTE: GEPA trains against a deterministic proxy metric (token F1 for text tasks); the final eval stays llm-rubric (see Known constraints)
uv run evalloop blog --runs <run_id>                        # figures/tables/article draft into blog/
```

## Bring your own task

Only three things need touching:

1. `config.yaml` — task name, label list, evaluated models, prices, judge settings
2. `data/golden.jsonl` — the eval dataset (single source of truth; format in [docs/DESIGN.md#5-データ仕様](docs/DESIGN.md#5-データ仕様), Japanese)
3. `prompts/base/task.txt` — the base prompt containing the `{{input}}` placeholder

`config.yaml`'s `models[].provider` uses promptfoo notation (e.g. `anthropic:messages:claude-...`, `ollama:chat:qwen2.5:7b`), while `optimize.reflection_provider` uses dspy/litellm notation (e.g. `anthropic/claude-...`). **The two formats differ.** Prices and provider IDs in the bundled config are samples only: never use an ID that doesn't pass `doctor`, and update prices to the official pricing at the time of use.

> **Models that reject sampling parameters**: `claude-opus-4-8` and `claude-fable-5` reject `temperature` and other sampling parameters with **HTTP 400**. Set `models[].supports_sampling_params: false` for such models and `evalloop build` will omit temperature from the generated promptfoo config (`max_tokens` is always sent). The bundled `config.yaml` already sets this for opus48 / fable5.
> Also note that `claude-fable-5` has always-on thinking, so its latency and output token counts can be larger than other models' (keep this in mind when interpreting cost estimates and latency comparisons).

## CLI commands

| Command | Description |
|---|---|
| `evalloop doctor` | Connectivity check for Node/promptfoo/Ollama/API keys |
| `evalloop build [--allow-same-judge] [--yes]` | golden.jsonl → promptfoo configs, with a pre-run cost estimate |
| `evalloop run [--variant NAME] [--repeat N] [--limit N] [--no-cache]` | Run promptfoo eval and record results/runs/{run_id}/ |
| `evalloop view` | promptfoo's local viewer (pass-through to `promptfoo view`) |
| `evalloop report RUN_ID` | Markdown report: model × accuracy × cost × latency |
| `evalloop calibrate [--run-id ID]` | Agreement rate between the LLM judge and human_labels.jsonl |
| `evalloop failures RUN_ID` | Extract failing cases, append note rows to notes.csv (idempotent) |
| `evalloop cluster [--notes PATH]` | An LLM drafts a failure taxonomy from notes.csv |
| `evalloop pivot RUN_ID` | Failure-category × model cross-tab |
| `evalloop optimize` | Prompt optimization with dspy GEPA, then automatic run/report/compare |
| `evalloop compare --runs A,B` | Before/after comparison of two runs |
| `evalloop blog --runs A[,B] [--slug NAME]` | Publish-guarded export of the blog bundle |

## Tests / CI

```bash
uv run pytest        # unit tests (promptfoo/GEPA fully mocked; no API keys or Node needed)
uv run ruff check .  # linter (the same check CI runs)
```

Everything related to the iron rules — label normalization, train/test split separation, the output.json parser, the blog publish guards — is covered by unit tests (`tests/`). Tests never write into the checkout (see the `isolated_artifact_paths` fixture in [tests/conftest.py](tests/conftest.py)).

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs pytest + ruff on Ubuntu / Windows × Python 3.11 / 3.12 for every push and PR. Additionally, on pushes to master, if the `OLLAMA_API_KEY` secret is configured, a 3-case live smoke (build → run → report) runs against Ollama Cloud (gpt-oss:20b); it is skipped automatically otherwise. No metered API cost is incurred.

## Issues found (and fixed) during live Windows verification

The commands `doctor`/`build`/`run`/`report`/`blog` were exercised on a real Windows 11 machine with Node.js and Ollama (qwen2.5:7b), which surfaced the following bugs that fully mocked unit tests had not caught:

- **`subprocess.run(["npx", ...])` raises `FileNotFoundError` on Windows**: `npx` is actually `npx.cmd`, and `subprocess` without a shell does not apply PATHEXT resolution. Fixed by resolving through `shutil.which("npx")`
- **promptfoo's actual Node.js requirement is `^20.20.0 || >=22.22.0`**: 21.x and 22.0–22.21.x are hard-rejected by promptfoo itself at startup (you can't tell from `node --version` alone)
- **`subprocess.run(..., text=True)` crashes under cp932 (the default code page on Japanese Windows)**: when promptfoo's output contains characters not representable in cp932, the reader thread dies with `UnicodeDecodeError`. Fixed by specifying `encoding="utf-8", errors="replace"` explicitly
- **`llm-rubric` with `value: file://...` does not substitute `{{input}}`/`{{expected}}`**: inspecting the actual grading prompt showed that file://-referenced rubrics bypass Nunjucks templating and the placeholders reach the judge verbatim (inline `value` strings ARE templated). Fixed by reading the rubric file's content and embedding it as an inline string (`build.py`, `calibrate.py`)
- A bug where a detected CJK font was never actually set as matplotlib's `font.family`, causing garbled Japanese chart labels, was also fixed (found during the M5 implementation)

These issues only surfaced when actually evaluating the CUAD-100 task (below) — a demonstration of the limits of purely mock-based testing.

## Live verification on Windows + CUAD-100

Using `config.local-verify.yaml` (Ollama qwen2.5:7b only, no API keys), the full `build` → `run` → `report` → `blog` pipeline was executed on a real Windows machine. On a 5-case subset, the grading logic (llm-rubric judge) was confirmed to produce meaningful pass/fail verdicts (e.g. an output of "no applicable clause" is correctly failed when the gold answer contains a clause, and passed when the gold answer is also "no applicable clause"). A full run over all 80 test-split cases takes a long time due to CPU-bound local inference (measured ~136 seconds per case = two model calls: extraction + grading).

## Generated artifacts policy (files not tracked by git)

Everything the `evalloop` commands generate is gitignored and not part of the repository. After a fresh clone, run `uv run evalloop build` first, as in the Quickstart, to generate `promptfoo/promptfooconfig.yaml` and `data/build/`.

| Command | Artifacts (all untracked) |
|---|---|
| `evalloop build` | `data/build/`, `promptfoo/promptfooconfig.yaml` |
| `evalloop run` | `results/runs/{run_id}/`, `results/index.jsonl` (machine-local audit ledger) |
| `evalloop report` | `results/reports/` |
| `evalloop failures` / `cluster` | `data/notes.csv`, `data/failures.jsonl`, `data/taxonomy.draft.yaml` |
| `evalloop optimize` | `promptfoo/variants/` (`prompts/optimized/` may optionally be committed as experiment artifacts) |
| `evalloop blog` | `blog/` |

Raw run outputs (output.json / meta.json) can contain local absolute paths and provider error payloads, so they are never committed to the public repository. Hand-curated files (`data/golden.jsonl`, `data/human_labels.jsonl`, `data/taxonomy.yaml`, `prompts/base/`, `config.yaml`) are tracked as usual.

## Data provenance

All bundled data comes from public datasets or was created synthetically for this project; **none of it is related to real customer data, business data, or actual inquiries**.

- `data/golden.jsonl` — a 100-case subset extracted from [CUAD v1](https://www.atticusprojectai.org/cuad) (published by The Atticus Project, **CC BY 4.0**), obtained via the `chenghao/cuad_qa` mirror on Hugging Face (source attribution in `config.yaml`'s `blog.allowed_sources`)
- `data/human_labels.jsonl` — intentionally empty at the moment (no human review pass over CUAD-100 yet; see Known constraints)
- `data/sample/golden.jsonl` — the previous sample task (4-way inquiry classification): **20 self-made dummy cases** (`meta.source: "self-made"`). Invented texts imitating generic SaaS inquiries; not quotes or adaptations of real inquiries
- `data/sample/human_labels.jsonl` — **10 synthetic fixtures** for the judge-calibration demo. `output_raw` values are fictional model outputs, not real LLM results

## Known constraints

- `evalloop optimize` supports all three answer types, but the GEPA **training metric is a deterministic proxy, not the final evaluation**: `label` uses the label-match port, `text` (e.g. the active CUAD-100 task) uses SQuAD-style token F1 against the gold span(s), and `json` uses a deep-equality port. For text tasks the final promptfoo evaluation still uses the llm-rubric judge, so training metric and final grading can diverge — measuring that divergence is part of the GEPA case study
- With a small local model (qwen2.5:7b) as judge, instruction following is less stable than with frontier models (e.g. it occasionally returns grading rationales in languages other than English/Japanese). Prefer a judge substantially stronger than the models being evaluated (as `config.yaml` is designed to do)
- `data/human_labels.jsonl` is intentionally empty because there are no human labels for the CUAD-100 task yet. Using `evalloop calibrate` requires a human review pass first

For design background, data specs, and the details of the "iron rules", see [docs/DESIGN.md](docs/DESIGN.md) (Japanese).

## Installation policy

This project is not published to PyPI. **It is meant to be used via git clone + `uv sync`** (see [Setup](#setup)). The design is anchored to in-tree path conventions (`data/`, `prompts/`, `results/`, etc.), so wheel installation into site-packages is unsupported.

## License

[MIT License](LICENSE). Bundled data follows the license stated in its provenance notes (the current `data/golden.jsonl` derives from CUAD v1, CC BY 4.0).
