# llm-eval-loop

[![CI](https://github.com/hidetomasuoka/llm-eval-loop/actions/workflows/ci.yml/badge.svg)](https://github.com/hidetomasuoka/llm-eval-loop/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**English** | [日本語](README.ja.md)

> This README is kept in sync with the Japanese original ([README.ja.md](README.ja.md)). If the two ever diverge, the Japanese version is authoritative.

An LLM evaluation harness. It evaluates one task across models — from small local models to frontier models — under identical conditions, to answer a single question:

> **Which model meets the accuracy bar, and at what cost?**

Execution and grading are delegated to [promptfoo](https://www.promptfoo.dev/); the Python layer (`evalloop`) is a thin glue layer that owns dataset management, judge calibration, failure analysis, prompt optimization with [dspy](https://dspy.ai/) GEPA, and publish-guarded blog export.

This is an experimental personal project, but it is kept in a state you can `git clone` and use as-is (see the for mac / for windows verification sections below for what has actually been exercised on real machines and in CI). Issues and PRs are welcome — support is best-effort. See [CONTRIBUTING.md](CONTRIBUTING.md) for expectations and the development workflow.

For the design rationale, data specifications, and the project's non-negotiable "iron rules", see [docs/DESIGN.md](docs/DESIGN.md) (full design doc, Japanese).

> **Policy**: this project never uses `promptfoo share` (cloud upload). Use `evalloop view` to browse local results.

## Requirements

- **Node.js**: `^20.20.0` or `>=22.22.0` (21.x and 22.0.0–22.21.x are NOT supported — promptfoo itself refuses to start outside this range; check `node --version` first)
- **promptfoo**: the version is pinned in `src/evalloop/run.py` (`PROMPTFOO_VERSION`) and executed as `npx promptfoo@<pinned version>`. `@latest` is never used — that would mean supply-chain exposure and reproducibility drift. To upgrade: bump the pin, pass `evalloop doctor` plus a `run --limit` smoke, then commit
- **Python 3.11+** and [uv](https://docs.astral.sh/uv/)
- **Ollama** (only if you use local models, e.g. `ollama pull qwen2.5:7b`)
- `ANTHROPIC_API_KEY` (required for real runs: evaluation, the LLM judge, and GEPA reflection). `OPENAI_API_KEY` / `GEMINI_API_KEY` are optional depending on which models you configure

Pre-run cost estimates use Anthropic's free official token-counting API when
`ANTHROPIC_API_KEY` is available and local tiktoken for recognized OpenAI models.
Missing credentials, unavailable networks, and unsupported providers fall back
without failing to an explicit mixed Japanese/English heuristic. Set
`EVALLOOP_TOKEN_COUNT_API=off` for fully offline operation. The method used is
shown in the `build` and `optimize` estimate output.

## Setup

```bash
node --version                   # must satisfy the range above
uv sync
ollama pull qwen2.5:7b           # if using local models
export ANTHROPIC_API_KEY=...
uv run evalloop doctor           # connectivity check for Node/promptfoo/Ollama/API keys. Always run this first
```

`doctor` runs a single tiny eval against every configured provider (it calls real APIs).

## Tasks are self-contained workspaces

Every evaluation task lives in its own directory under `tasks/<name>/` (issue #47): `task.yaml` (what to measure and how), `golden.jsonl` (the dataset), `prompts/task.txt` (+ `judge_rubric.txt` for text tasks), plus the hand-curated analysis files (`human_labels.jsonl`, `taxonomy.yaml`, `notes.csv`). Generated artifacts go to per-task subtrees (`data/build/<task>/`, `promptfoo/<task>/`, `results/<task>/`, `blog/<task>/`). Select a task with `--task NAME` on any command, the `EVALLOOP_TASK` env var, or `default_task` in the global `config.yaml` (which also holds the shared model registry with prices). `evalloop task list` shows what exists.

**Data policy: task data is gitignored by default.** Only `task.yaml`, `prompts/` and `PROVENANCE.md` (source + how to re-obtain the data) are tracked, so private datasets can safely live here. The synthetic `sample-inquiry` demo task is the one opt-in exception — its data is tracked, which is why it is the default task and the CI smoke target.

## Quickstart

Works on a fresh clone with the bundled `sample-inquiry` task (24 synthetic inquiry-classification cases, `answer_type=label`, graded by a deterministic assert — no LLM judge, and with a local Ollama model no API key at all):

```bash
uv run evalloop build --models qwen7b          # sample-inquiry is the default task
uv run evalloop run --limit 10                 # try just the first 10 cases
uv run evalloop report <run_id printed above>  # Markdown table: model x accuracy x cost
uv run evalloop view                           # browse results in promptfoo's local viewer
```

Drop `--models qwen7b` to evaluate the full model registry (needs `ANTHROPIC_API_KEY`).

The CUAD-100 contract-clause-extraction task (`answer_type=text`, llm-rubric judge) is defined in `tasks/cuad100/` but ships without data — see [tasks/cuad100/PROVENANCE.md](tasks/cuad100/PROVENANCE.md) to obtain it (CUAD v1, CC BY 4.0), then:

```bash
uv run evalloop build --task cuad100 --allow-same-judge
uv run evalloop run --task cuad100 --limit 10
```

> The checked-in CUAD configuration currently evaluates only `glm52` and uses the same `ollama:chat:glm-5.2:cloud` provider as its judge. Therefore the entire glm52 result is self-graded, and `--allow-same-judge` is required. Point the judge at a provider outside the evaluated model set to remove both the flag and this self-grading bias.

If everything looks good, drop `--limit` to run the full set, then continue into failure analysis, the improvement loop, and blog export.

```bash
uv run evalloop run                                        # full set
uv run evalloop failures <run_id>                          # extract failing cases, append note rows to data/notes.csv
#   -> fill in the `note` column by hand with failure reasons
uv run evalloop cluster                                    # an LLM drafts category proposals into data/taxonomy.draft.yaml
#   -> review, then save as data/taxonomy.yaml (the draft never overwrites it automatically)
uv run evalloop pivot <run_id>                              # failure-category x model cross-tab
uv run evalloop calibrate --run-id <run_id>                 # agreement rate between the LLM judge and human labels
uv run evalloop optimize                                    # improve the prompt with dspy (GEPA / MIPROv2 / COPRO, uses the train split only)
#   -> select the method via optimize.method in task.yaml (unset = gepa). afterwards run/report/compare (against the latest base run, if any) execute automatically
#   -> after the automatic holdout run, optimize prints a train-vs-holdout generalization gate (pass/fail vs baseline holdout; display-only, exit code unchanged) and records it in optimize_log.json
#   -> a rough cost estimate (train size x per-method iteration factor x registry prices) is shown first; exceeding run.cost_warn_usd prompts for confirmation (--yes suppresses, for CI)
#   NOTE: every method trains on a deterministic proxy metric by answer_type (label match / token F1 / JSON deep-equal); the final promptfoo eval still uses the task's configured grader (label_match for sample-inquiry, llm-rubric for text tasks — see Known constraints)
#   NOTE: for which failure symptoms warrant which optimization technique, see docs/APO_GUIDE.md (a symptom → granularity → method diagnostic guide)
uv run evalloop blog --runs <run_id>                        # figures/tables/article draft into blog/
```

## Bring your own task

Adding a task never touches existing tasks. `uv run evalloop task init <name>` scaffolds the workspace (task.yaml, prompts/, a PROVENANCE.md template), leaving essentially three files to fill in:

1. `tasks/<name>/task.yaml` — answer_type, labels, judge/optimize settings, and (optionally) which registry models to evaluate (copy an existing task.yaml as a template)
2. `tasks/<name>/golden.jsonl` — the eval dataset (single source of truth; format in [docs/DESIGN.md#5-データ仕様](docs/DESIGN.md#5-データ仕様), Japanese). Gitignored by default — add a `PROVENANCE.md` describing where it came from
3. `tasks/<name>/prompts/task.txt` — the base prompt containing the `{{input}}` placeholder (text tasks also need `prompts/judge_rubric.txt`)

Then run everything with `--task <name>`. Model definitions (provider IDs, prices, `supports_sampling_params`) live once in the global `config.yaml` registry; a task picks aliases from it. `models[].provider` uses promptfoo notation (e.g. `anthropic:messages:claude-...`, `ollama:chat:qwen2.5:7b`), while `optimize.reflection_provider` uses dspy/litellm notation (e.g. `anthropic/claude-...`). **The two formats differ.** Prices and provider IDs in the bundled config are samples only: never use an ID that doesn't pass `doctor`, and update prices to the official pricing at the time of use.

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
| `evalloop diagnose [--answers 1,2,3]` | Interactive symptom → granularity → method checklist (APO readiness and recommended `optimize.method`; no LLM) |
| `evalloop optimize` | Prompt optimization with dspy (GEPA / MIPROv2 / COPRO, chosen via `optimize.method` in task.yaml), then automatic run/report/compare (method selection guide: [docs/APO_GUIDE.md](docs/APO_GUIDE.md)) |
| `evalloop compare --runs A,B[,C...]` | Compare 2 runs (before/after deltas + cost%/tokens/prompt-length tradeoff note) or 3+ runs (model×run matrix) |
| `evalloop blog --runs A[,B] [--slug NAME]` | Publish-guarded export of the blog bundle |

## Tests / CI

```bash
uv run pytest        # unit tests (promptfoo/GEPA fully mocked; no API keys or Node needed)
uv run ruff check .  # linter (the same check CI runs)
```

Everything related to the iron rules — label normalization, train/test split separation, the output.json parser, the blog publish guards — is covered by unit tests (`tests/`). Tests never write into the checkout (see the `isolated_artifact_paths` fixture in [tests/conftest.py](tests/conftest.py)).

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs pytest + ruff on Ubuntu / Windows × Python 3.11 / 3.12 for every push and PR (macOS verification status is noted in the for mac section below). Additionally, on pushes to master, if the `OLLAMA_API_KEY` secret is configured, a 3-case live smoke (`--task sample-inquiry --models gptoss20b`: build → run → report) runs against Ollama Cloud; it is skipped automatically otherwise. No metered API cost is incurred.

## Live verification: for mac / for windows

The commands `doctor`/`build`/`run`/`report`/`blog` were exercised on two real machines. Bugs that fully mocked unit tests had not caught were surfaced during this live verification.

### for mac

- macOS (Apple Silicon) + Node.js + Ollama (qwen2.5:7b)
- `subprocess` and path resolution follow standard Unix behavior, so no OS-specific bugs were found
- The CJK-font-detection → matplotlib `font.family` bug (below) was also discovered and fixed on macOS
- Both the bundled `sample-inquiry` and CUAD-100 (after data acquisition) tasks have been confirmed to run the full pipeline

### for windows

The commands were exercised on a real Windows 11 machine with Node.js and Ollama (qwen2.5:7b), which surfaced the following bugs that fully mocked unit tests had not caught:

- **`subprocess.run(["npx", ...])` raises `FileNotFoundError` on Windows**: `npx` is actually `npx.cmd`, and `subprocess` without a shell does not apply PATHEXT resolution. Fixed by resolving through `shutil.which("npx")`
- **promptfoo's actual Node.js requirement is `^20.20.0 || >=22.22.0`**: 21.x and 22.0–22.21.x are hard-rejected by promptfoo itself at startup (you can't tell from `node --version` alone)
- **`subprocess.run(..., text=True)` crashes under cp932 (the default code page on Japanese Windows)**: when promptfoo's output contains characters not representable in cp932, the reader thread dies with `UnicodeDecodeError`. Fixed by specifying `encoding="utf-8", errors="replace"` explicitly
- **`llm-rubric` with `value: file://...` does not substitute `{{input}}`/`{{expected}}`**: inspecting the actual grading prompt showed that file://-referenced rubrics bypass Nunjucks templating and the placeholders reach the judge verbatim (inline `value` strings ARE templated). Fixed by reading the rubric file's content and embedding it as an inline string (`build.py`, `calibrate.py`)
- A bug where a detected CJK font was never actually set as matplotlib's `font.family`, causing garbled Japanese chart labels, was also fixed (found during the M5 implementation; a shared mac/windows bug re-confirmed during windows live verification)

These issues only surfaced when actually evaluating the CUAD-100 task (below) — a demonstration of the limits of purely mock-based testing.

## for windows + CUAD-100 live verification

Using `config.local-verify.yaml` (Ollama qwen2.5:7b only, no API keys), the full `build` → `run` → `report` → `blog` pipeline was executed on a real Windows machine. On a 5-case subset, the grading logic (llm-rubric judge) was confirmed to produce meaningful pass/fail verdicts (e.g. an output of "no applicable clause" is correctly failed when the gold answer contains a clause, and passed when the gold answer is also "no applicable clause"). A full run over all 80 test-split cases takes a long time due to CPU-bound local inference (measured ~136 seconds per case = two model calls: extraction + grading).

## Generated artifacts policy (files not tracked by git)

Everything the `evalloop` commands generate is gitignored and lives in per-task subtrees. After a fresh clone, run `uv run evalloop build` first, as in the Quickstart.

| Command | Artifacts (all untracked, `<task>` = task name) |
|---|---|
| `evalloop build` | `data/build/<task>/`, `promptfoo/<task>/promptfooconfig.yaml` |
| `evalloop run` | `results/<task>/runs/{run_id}/`, `results/<task>/index.jsonl` (machine-local audit ledger) |
| `evalloop report` | `results/<task>/reports/` |
| `evalloop failures` / `cluster` | `tasks/<task>/notes.csv`, `tasks/<task>/taxonomy.draft.yaml` |
| `evalloop optimize` | `promptfoo/<task>/variants/` and `tasks/<task>/optimized/<alias>/{method}-{ts}-{slug}/` plus `tasks/<task>/optimized/index.jsonl` (may optionally be committed as experiment artifacts) |
| `evalloop blog` | `blog/<task>/` |

Raw run outputs (output.json / meta.json) can contain local absolute paths and provider error payloads, so they are never committed to the public repository. Task **data** (`golden.jsonl`, `human_labels.jsonl`, `notes.csv`, `taxonomy*.yaml`) is also gitignored by default per the data policy above; only the task's "code" (`task.yaml`, `prompts/`, `PROVENANCE.md`) and the global `config.yaml` are tracked.

## Data provenance

Each task documents its data source and how to re-obtain it in `tasks/<name>/PROVENANCE.md`. All data ever bundled here comes from public datasets or was created synthetically for this project; **none of it is related to real customer data, business data, or actual inquiries**.

- `tasks/sample-inquiry/` (tracked, opt-in) — **24 self-made dummy cases** for 4-way inquiry classification (`meta.source: "self-made"`; invented texts imitating generic SaaS inquiries) plus **10 synthetic fixtures** for the judge-calibration demo (`output_raw` values are fictional model outputs)
- `tasks/cuad100/` (data untracked) — a 100-case subset extracted from [CUAD v1](https://www.atticusprojectai.org/cuad) (published by The Atticus Project, **CC BY 4.0**), obtained via the `chenghao/cuad_qa` mirror on Hugging Face; see its PROVENANCE.md for the file fingerprint and recovery steps

## Known constraints

- `evalloop optimize` supports all three answer types and three optimization methods (select via `optimize.method`: `gepa` / `miprov2` / `copro`, unset = gepa). Every method's **training metric is a deterministic proxy, not the final evaluation**: `label` uses the label-match port, `text` (e.g. the active CUAD-100 task) uses SQuAD-style token F1 against the gold span(s), and `json` uses a deep-equality port. For text tasks the final promptfoo evaluation still uses the llm-rubric judge, so training metric and final grading can diverge — measuring that divergence is part of the optimization case study
- The "training metric is a proxy" constraint above is common to **GEPA, MIPROv2, and COPRO**, and to any future optimizer (OPRO, APE, EASE, etc.) this harness may add. Fast in-process candidate evaluation requires a structured verdict (label match, token F1, deep-equal, etc.); invoking an LLM judge per candidate rollout is forbidden by the iron rule (Python never calls a model provider directly). So "train on a proxy metric, verify on a separate final metric" is a harness-wide APO premise (see [docs/APO_GUIDE.md](docs/APO_GUIDE.md) for method selection)
- With a small local model (qwen2.5:7b) as judge, instruction following is less stable than with frontier models (e.g. it occasionally returns grading rationales in languages other than English/Japanese). Prefer a judge substantially stronger than the models being evaluated (as `config.yaml` is designed to do)
- `tasks/cuad100/human_labels.jsonl` is intentionally empty because there are no human labels for the CUAD-100 task yet. Using `evalloop calibrate` there requires a human review pass first (the `sample-inquiry` task ships 10 synthetic labels for the calibration demo)

For design background, data specs, and the details of the "iron rules", see [docs/DESIGN.md](docs/DESIGN.md) (Japanese).

## Installation policy

This project is not published to PyPI. **It is meant to be used via git clone + `uv sync`** (see [Setup](#setup)). The design is anchored to in-tree path conventions (`data/`, `prompts/`, `results/`, etc.), so wheel installation into site-packages is unsupported.

## License

[MIT License](LICENSE). Bundled data follows the license stated in each task's `tasks/<name>/PROVENANCE.md`.
