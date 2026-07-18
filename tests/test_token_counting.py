import io
import json
import urllib.error

import tiktoken

from evalloop import token_counting

# Old build.py constant kept here so the accuracy regression test stays pinned
# to the pre-#109 heuristic rather than whatever the production code exports.
_LEGACY_CHARS_PER_TOKEN = 2.0


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return io.BytesIO(self._body).read()


def test_mixed_text_heuristic_distinguishes_japanese_and_ascii_density():
    japanese = token_counting.heuristic_token_count("契約条項を確認します")
    ascii_text = token_counting.heuristic_token_count("abcdefghijkl")

    assert japanese == 10
    assert ascii_text == 3
    assert japanese > ascii_text


def test_anthropic_official_counting_api_is_used_when_available(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        seen["headers"] = {key.lower(): value for key, value in request.header_items()}
        seen["timeout"] = timeout
        return _FakeResponse({"input_tokens": 21})

    monkeypatch.setenv("EVALLOOP_TOKEN_COUNT_API", "auto")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(token_counting.urllib.request, "urlopen", fake_urlopen)

    result = token_counting.average_input_tokens(
        "anthropic:messages:claude-sonnet-5", ["日本語の入力", "English input"]
    )

    assert result.average_input_tokens == 11
    assert result.method == "anthropic-count-tokens-api"
    assert result.sampled_case_count == 2
    assert seen["url"] == token_counting.ANTHROPIC_COUNT_TOKENS_URL
    assert seen["body"]["model"] == "claude-sonnet-5"
    assert seen["headers"]["x-api-key"] == "test-key"
    assert seen["headers"]["anthropic-version"] == token_counting.ANTHROPIC_API_VERSION
    assert seen["timeout"] == 3


def test_anthropic_without_key_falls_back_without_network(monkeypatch):
    monkeypatch.setenv("EVALLOOP_TOKEN_COUNT_API", "auto")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        token_counting.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network must not be called")),
    )

    result = token_counting.average_input_tokens("anthropic:messages:claude-sonnet-5", ["契約を確認"])

    assert result.average_input_tokens > 0
    assert result.method == "heuristic:mixed-text-v1"


def test_anthropic_network_failure_falls_back_instead_of_failing(monkeypatch):
    monkeypatch.setenv("EVALLOOP_TOKEN_COUNT_API", "auto")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        token_counting.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )

    result = token_counting.average_input_tokens("anthropic:messages:claude-sonnet-5", ["offline input"])

    assert result.average_input_tokens > 0
    assert result.method == "heuristic:mixed-text-v1"


def test_token_count_api_off_forces_heuristic_even_with_key(monkeypatch):
    monkeypatch.setenv("EVALLOOP_TOKEN_COUNT_API", "off")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        token_counting.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network must not be called")),
    )

    result = token_counting.average_input_tokens("anthropic:messages:claude-sonnet-5", ["契約を確認"])

    assert result.method == "heuristic:mixed-text-v1"


def test_recognized_openai_model_uses_local_tiktoken():
    result = token_counting.average_input_tokens("openai:gpt-4o", ["日本語", "English"])

    assert result.average_input_tokens > 0
    assert result.method.startswith("tiktoken:")
    assert result.sampled_case_count == 2


def test_provider_aware_count_beats_legacy_chars_per_token_vs_tiktoken():
    """Acceptance #109: against a real tokenizer, the shared counter is closer
    than the old fixed chars/token heuristic on mixed Japanese/English text.

    Live meta.json costs vary by provider pricing and output length; tiktoken
    is the offline stand-in for measured input tokens on OpenAI chat models.
    """
    texts = [
        "契約書の解除条項を抽出してください。Party A may terminate upon 30 days notice.",
        "Confidentiality obligations survive for three (3) years after termination. 秘密保持義務は終了後も存続する。",
        "Governing Law: This Agreement shall be governed by the laws of California. 準拠法はカリフォルニア州法とする。",
    ]
    encoding = tiktoken.encoding_for_model("gpt-4o")
    actual_avg = sum(len(encoding.encode(t, disallowed_special=())) for t in texts) / len(texts)

    legacy_avg = sum(max(1, int(len(t) / _LEGACY_CHARS_PER_TOKEN)) for t in texts) / len(texts)
    counted = token_counting.average_input_tokens("openai:gpt-4o", texts)

    assert counted.method.startswith("tiktoken:")
    new_error = abs(counted.average_input_tokens - actual_avg)
    legacy_error = abs(legacy_avg - actual_avg)
    assert new_error < legacy_error
    # Exact tokenizer path should be within one token of the mean (ceil only).
    assert new_error <= 1.0
    assert legacy_error > 1.0


def test_offline_heuristic_also_beats_legacy_chars_per_token_on_mixed_text():
    texts = [
        "契約条項の確認を行います",
        "Review the termination clause carefully before signing.",
        "秘密情報の取扱いについて定める。Confidential Information means...",
    ]
    encoding = tiktoken.get_encoding("cl100k_base")
    actual_avg = sum(len(encoding.encode(t, disallowed_special=())) for t in texts) / len(texts)

    legacy_avg = sum(max(1, int(len(t) / _LEGACY_CHARS_PER_TOKEN)) for t in texts) / len(texts)
    heuristic_avg = sum(token_counting.heuristic_token_count(t) for t in texts) / len(texts)

    assert abs(heuristic_avg - actual_avg) < abs(legacy_avg - actual_avg)


def test_render_case_prompts_replaces_input_per_case():
    assert token_counting.render_case_prompts("Input: {{input}}", ["日本語", "English"]) == [
        "Input: 日本語",
        "Input: English",
    ]


def test_empty_texts_return_explicit_heuristic_without_raising():
    result = token_counting.average_input_tokens("openai:gpt-4o", [])
    assert result.average_input_tokens == 1
    assert result.method == "heuristic:mixed-text-v1"
    assert result.sampled_case_count == 0
