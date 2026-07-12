# Reproducibility conditions

- experiment date: 2026-07-11
- models:
  - `ollama:chat:glm-5.2:cloud` (alias: glm52, tier: frontier)
  - `anthropic:messages:claude-sonnet-5` (alias: sonnet5, tier: mid)
- test cases (approx, n per model): 60
- repeat: 1
- temperature: 0.0
- prompt sha256 (first 8): `def601c4`
- promptfoo config sha256 (first 8): `09d5f850`
- grader: `llm-rubric` (provider: `ollama:chat:glm-5.2:cloud`, calibration: uncalibrated, agreement: n/a)
- total cost: $5.1997 (~780 JPY)
- promptfoo version: `0.121.18`
- dspy version: `3.2.1`
- fig03 (failure heatmap): included

## reproduce
```bash
evalloop build --task cuad100 --allow-same-judge
evalloop run --task cuad100 --repeat 1
evalloop report --task cuad100 20260711-152509-e040
```
