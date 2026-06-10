"""Unit tests for server.py — all HTTP mocked, no runtime required."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

import server


# --- helpers -----------------------------------------------------------------
class FakeResp:
    def __init__(self, json_data, status=200):
        self._json = json_data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


OLLAMA_TAGS = {
    "models": [
        {
            "name": "big:14b",
            "size": 9 * 1024**3,
            "details": {"parameter_size": "14B", "quantization_level": "Q4_K_M", "family": "qwen2"},
        },
        {
            "name": "small:1b",
            "size": 1 * 1024**3,
            "details": {"parameter_size": "1B", "quantization_level": "Q8_0", "family": "llama"},
        },
        {
            "name": "mid:7b",
            "size": 4 * 1024**3,
            "details": {"parameter_size": "7B", "quantization_level": "Q4_K_M", "family": "qwen2"},
        },
        {
            "name": "tiny:0.5b",
            "size": 300 * 1024**2,
            "details": {"parameter_size": "494M", "quantization_level": "Q4_0", "family": "qwen2"},
        },
    ]
}

OPENAI_MODELS = {"data": [{"id": "alpha"}, {"id": "beta"}, {"id": "gamma"}, {"id": "delta"}]}


def fake_get_ollama(url, timeout=None):
    if url.endswith("/api/tags"):
        return FakeResp(OLLAMA_TAGS)
    if url.endswith("/api/ps"):
        return FakeResp({"models": [{"name": "small:1b"}]})
    return FakeResp({}, status=404)


def fake_get_openai_only(url, timeout=None):
    if url.endswith("/v1/models"):
        return FakeResp(OPENAI_MODELS)
    return FakeResp({}, status=404)


def fake_get_down(url, timeout=None):
    raise requests.exceptions.ConnectionError("connection refused")


def make_fake_client(reply="hello", empty=False, raise_exc=None):
    client = MagicMock()
    if raise_exc:
        client.chat.completions.create.side_effect = raise_exc
    else:
        choices = [] if empty else [SimpleNamespace(message=SimpleNamespace(content=reply))]
        client.chat.completions.create.return_value = SimpleNamespace(choices=choices)
    return client


# --- _normalize_host ----------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
        ("http://localhost:1234/v1", "http://localhost:1234"),
        ("http://localhost:1234/v1/", "http://localhost:1234"),
        ("http://localhost:1234/V1", "http://localhost:1234"),
        ("  http://h:8000/v1  ", "http://h:8000"),
    ],
)
def test_normalize_host(raw, expected):
    assert server._normalize_host(raw) == expected


# --- _parse_timeout ------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 300.0),
        ("", 300.0),
        ("   ", 300.0),
        ("abc", 300.0),
        ("0", 300.0),
        ("-5", 300.0),
        ("inf", 300.0),
        ("nan", 300.0),
        ("120", 120.0),
        ("600.5", 600.5),
    ],
)
def test_parse_timeout(raw, expected):
    assert server._parse_timeout(raw) == expected


# --- _list_local_models ---------------------------------------------------------
def test_list_local_models_ollama(monkeypatch):
    monkeypatch.setattr(server.requests, "get", fake_get_ollama)
    models, style = server._list_local_models("http://x")
    assert style == "ollama"
    assert len(models) == 4


def test_list_local_models_openai_fallback(monkeypatch):
    monkeypatch.setattr(server.requests, "get", fake_get_openai_only)
    models, style = server._list_local_models("http://x")
    assert style == "openai"
    assert [m["name"] for m in models] == ["alpha", "beta", "gamma", "delta"]


def test_list_local_models_both_fail(monkeypatch):
    monkeypatch.setattr(server.requests, "get", fake_get_down)
    with pytest.raises(requests.exceptions.ConnectionError):
        server._list_local_models("http://x")


# --- list_models tool ------------------------------------------------------------
def test_list_models_ollama_format(monkeypatch):
    monkeypatch.setattr(server.requests, "get", fake_get_ollama)
    out = server.list_models()
    assert "big:14b" in out
    assert "Q4_K_M" in out
    assert "GB" in out


def test_list_models_nonstring_metadata_no_crash(monkeypatch):
    tags = {"models": [{"name": "weird:1b", "size": None, "details": {"parameter_size": 7, "quantization_level": None}}]}
    monkeypatch.setattr(server.requests, "get", lambda url, timeout=None: FakeResp(tags) if url.endswith("/api/tags") else FakeResp({}, 404))
    out = server.list_models()
    assert "weird:1b" in out
    assert not out.startswith("ERROR")


def test_list_models_openai_names_and_note(monkeypatch):
    monkeypatch.setattr(server.requests, "get", fake_get_openai_only)
    out = server.list_models()
    assert "alpha" in out
    assert "names only" in out


def test_list_models_runtime_down(monkeypatch):
    monkeypatch.setattr(server.requests, "get", fake_get_down)
    out = server.list_models()
    assert out.startswith("ERROR (connection)")
    assert "HINT" in out


def test_list_models_host_override_normalized(monkeypatch):
    seen = []

    def spy(url, timeout=None):
        seen.append(url)
        return fake_get_ollama(url, timeout)

    monkeypatch.setattr(server.requests, "get", spy)
    server.list_models(host="http://other:1234/v1/")
    assert seen[0] == "http://other:1234/api/tags"


# --- health tool -------------------------------------------------------------------
def test_health_ok_ollama(monkeypatch):
    monkeypatch.setattr(server.requests, "get", fake_get_ollama)
    out = server.health()
    assert out.startswith("OK")
    assert "ollama" in out
    assert "models installed: 4" in out
    assert "small:1b" in out  # loaded-in-VRAM line


def test_health_down(monkeypatch):
    monkeypatch.setattr(server.requests, "get", fake_get_down)
    out = server.health()
    assert out.startswith("ERROR (connection)")


# --- local_chat ----------------------------------------------------------------------
def test_local_chat_success(monkeypatch):
    monkeypatch.setattr(server, "_client_for", lambda h: make_fake_client("the reply"))
    assert server.local_chat("m", "hi") == "the reply"


def test_local_chat_system_prompt_passed(monkeypatch):
    client = make_fake_client("ok")
    monkeypatch.setattr(server, "_client_for", lambda h: client)
    server.local_chat("m", "hi", system="be terse")
    msgs = client.chat.completions.create.call_args.kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "be terse"}


def test_local_chat_empty_choices(monkeypatch):
    monkeypatch.setattr(server, "_client_for", lambda h: make_fake_client(empty=True))
    assert server.local_chat("m", "hi").startswith("ERROR (empty-response)")


def test_local_chat_model_not_found(monkeypatch):
    exc = Exception("model 'nope:1b' not found, try pulling it first")
    monkeypatch.setattr(server, "_client_for", lambda h: make_fake_client(raise_exc=exc))
    out = server.local_chat("nope:1b", "hi")
    assert out.startswith("ERROR (model-not-found)")
    assert "list_models" in out


def test_local_chat_timeout_classified(monkeypatch):
    exc = Exception("Request timed out.")
    monkeypatch.setattr(server, "_client_for", lambda h: make_fake_client(raise_exc=exc))
    out = server.local_chat("m", "hi")
    assert out.startswith("ERROR (timeout)")
    assert "LOCAL_LLM_TIMEOUT" in out


# --- local_compare --------------------------------------------------------------------
def test_local_compare_explicit_order_preserved(monkeypatch):
    monkeypatch.setattr(server, "_client_for", lambda h: make_fake_client("ans"))
    out = server.local_compare("q", models=["m1", "m2", "m3"])
    assert out.index("=== m1 ===") < out.index("=== m2 ===") < out.index("=== m3 ===")


def test_local_compare_auto_select_smallest_three(monkeypatch):
    monkeypatch.setattr(server.requests, "get", fake_get_ollama)
    client = make_fake_client("ans")
    monkeypatch.setattr(server, "_client_for", lambda h: client)
    out = server.local_compare("q")
    assert "tiny:0.5b" in out and "small:1b" in out and "mid:7b" in out
    assert "big:14b" not in out


def test_local_compare_auto_select_openai_first_three(monkeypatch):
    monkeypatch.setattr(server.requests, "get", fake_get_openai_only)
    monkeypatch.setattr(server, "_client_for", lambda h: make_fake_client("ans"))
    out = server.local_compare("q")
    assert "auto-selected first 3" in out
    assert "=== alpha ===" in out and "=== gamma ===" in out
    assert "=== delta ===" not in out


def test_local_compare_no_models(monkeypatch):
    monkeypatch.setattr(server.requests, "get", lambda url, timeout=None: FakeResp({"models": []}) if url.endswith("/api/tags") else FakeResp({}, 404))
    assert server.local_compare("q") == "(no models installed on local runtime)"


def test_local_compare_auto_select_skips_embedding_models(monkeypatch):
    tags = {
        "models": [
            {"name": "embeddinggemma:latest", "size": 100, "details": {}},
            {"name": "chat:1b", "size": 200, "details": {}},
        ]
    }
    monkeypatch.setattr(server.requests, "get", lambda url, timeout=None: FakeResp(tags) if url.endswith("/api/tags") else FakeResp({}, 404))
    monkeypatch.setattr(server, "_client_for", lambda h: make_fake_client("ans"))
    out = server.local_compare("q")
    assert "embeddinggemma" not in out
    assert "=== chat:1b ===" in out


def test_local_compare_only_embedding_models(monkeypatch):
    tags = {"models": [{"name": "nomic-embed-text", "size": 100, "details": {}}]}
    monkeypatch.setattr(server.requests, "get", lambda url, timeout=None: FakeResp(tags) if url.endswith("/api/tags") else FakeResp({}, 404))
    assert "embedding" in server.local_compare("q")


def test_local_chat_embedding_model_classified(monkeypatch):
    exc = Exception('Error code: 400 - "embeddinggemma:latest" does not support chat')
    monkeypatch.setattr(server, "_client_for", lambda h: make_fake_client(raise_exc=exc))
    out = server.local_chat("embeddinggemma:latest", "hi")
    assert out.startswith("ERROR (not-a-chat-model)")


def test_local_compare_one_model_fails_others_survive(monkeypatch):
    def per_model_client(h):
        client = MagicMock()

        def create(**kwargs):
            if kwargs["model"] == "bad":
                raise Exception("boom")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fine"))])

        client.chat.completions.create.side_effect = create
        return client

    monkeypatch.setattr(server, "_client_for", per_model_client)
    out = server.local_compare("q", models=["good", "bad"])
    assert "=== good ===\nfine" in out
    assert "=== bad ===\nERROR" in out
