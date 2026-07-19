from pathlib import Path

import yaml

from evalloop import cli
from evalloop import run as run_mod
from evalloop.schemas import (
    BlogConfig,
    Config,
    JudgeConfig,
    ModelConfig,
    OptimizeConfig,
    RunConfig,
    TaskConfig,
)


def test_env_key_for_provider_anthropic():
    assert cli._env_key_for_provider("anthropic:messages:claude-sonnet-4-6") == "ANTHROPIC_API_KEY"


def test_env_key_for_provider_openai():
    assert cli._env_key_for_provider("openai:gpt-5") == "OPENAI_API_KEY"


def test_env_key_for_provider_ollama_is_none():
    assert cli._env_key_for_provider("ollama:chat:qwen2.5:7b") is None


def test_node_version_ok_true_for_v20_20_plus():
    assert cli._node_version_ok("v20.20.0") is True
    assert cli._node_version_ok("v20.25.0") is True


def test_node_version_ok_true_for_v22_22_plus():
    assert cli._node_version_ok("v22.22.0") is True
    assert cli._node_version_ok("v23.0.0") is True


def test_node_version_ok_false_for_gap_versions():
    # confirmed against a real promptfoo runtime check: v21.x and v22.0-22.21
    # are rejected even though naive ">=20.20" checks would accept them
    assert cli._node_version_ok("v21.0.0") is False
    assert cli._node_version_ok("v22.17.0") is False
    assert cli._node_version_ok("v20.19.0") is False


def test_node_version_ok_false_for_old():
    assert cli._node_version_ok("v18.19.0") is False


def test_node_version_ok_none_for_unparsable():
    assert cli._node_version_ok("unknown") is None


def _smoke_cfg():
    return Config(
        task=TaskConfig(name="t", answer_type="text", prompt_file="prompts/base/task.txt"),
        models=[
            ModelConfig(provider="p:samples", alias="samples", tier="small"),
            ModelConfig(provider="p:nosample", alias="nosample", tier="frontier", supports_sampling_params=False),
        ],
        run=RunConfig(),
        judge=JudgeConfig(provider="p:judge"),
        optimize=OptimizeConfig(target_alias="samples", reflection_provider="r"),
        blog=BlogConfig(),
        path=Path("config.yaml"),
    )


def test_smoke_test_omits_temperature_for_no_sampling_models(monkeypatch):
    # opus48/fable5-style models reject temperature with HTTP 400; the doctor
    # smoke test must mirror build.py's supports_sampling_params handling or
    # it reports a false connectivity failure for exactly those models
    captured = {}

    def fake_eval(config_path, output_path, **kwargs):
        captured["config"] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

        class _P:
            returncode = 1
            stdout = ""
            stderr = ""

        return _P()

    monkeypatch.setattr(run_mod, "run_promptfoo_eval", fake_eval)

    cli._smoke_test_providers(_smoke_cfg())

    by_label = {p["label"]: p["config"] for p in captured["config"]["providers"]}
    assert by_label["samples"] == {"temperature": 0.0, "max_tokens": 16}
    assert by_label["nosample"] == {"max_tokens": 16}
    # case_id keeps parse_promptfoo_output from warning on every smoke row
    assert captured["config"]["tests"][0]["vars"]["case_id"] == "doctor-smoke-0001"
