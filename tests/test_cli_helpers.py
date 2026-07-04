from evalloop import cli


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
