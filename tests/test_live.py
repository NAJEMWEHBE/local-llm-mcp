"""Live smoke tests against a real local runtime (default: Ollama).

Excluded from the default run (and CI) via `-m "not live"` addopts.
Run locally before a release with:  pytest -m live
"""

import pytest

import server

pytestmark = pytest.mark.live


def test_live_health():
    out = server.health()
    assert out.startswith("OK"), out


def test_live_list_models():
    out = server.list_models()
    assert not out.startswith("ERROR"), out
    assert out.strip(), "expected at least one installed model"


def test_live_chat_smallest_model():
    models, style = server._list_local_models(server.HOST)
    models = [m for m in models if not server._is_embedding_model(str(m.get("name", "")))]
    assert models, "no chat-capable models installed"
    if style == "ollama":
        models.sort(key=lambda m: m.get("size") or 0)
    name = str(models[0]["name"])
    out = server.local_chat(name, "Reply with the single word: OK", max_tokens=10, temperature=0.0)
    assert not out.startswith("ERROR"), out
    assert out.strip(), "expected non-empty reply"
