#!/usr/bin/env python
"""Local OSS LLM MCP server.

Exposes locally-installed OSS models (served by any local OpenAI-compatible
runtime: Ollama, LM Studio, vLLM, llama.cpp server, text-generation-webui,
or anything else exposing /v1/chat/completions) as MCP tools that
Claude Code / Claude Desktop / Codex / Copilot can call.

Host resolution order:
    1. LOCAL_LLM_HOST     — preferred, runtime-agnostic name
    2. OPENAI_BASE_URL    — common in OpenAI-SDK ecosystems
    3. OLLAMA_HOST        — legacy / Ollama-default name (kept for compat)
    4. http://127.0.0.1:11434 (Ollama default)

A trailing /v1 on any of these is stripped (OPENAI_BASE_URL conventionally
includes it; appending another would 404 every call).

Model listing tries Ollama's /api/tags first (rich metadata: size, quant,
parameter count) and falls back to the OpenAI-standard /v1/models
(names only), so listing works on every runtime, not just Ollama.

Timeout (v0.1.1): per-call request timeout 300s default, overridable via
LOCAL_LLM_TIMEOUT env. Bad override values (empty, non-numeric, zero,
negative, inf/nan) fall back to the default rather than crashing module
import. SDK auto-retries disabled (max_retries=0) — cold-load is
deterministic, not transient.
"""

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP("local-llm")


def _normalize_host(raw: str) -> str:
    """Strip whitespace, trailing slashes, and a trailing /v1.

    OPENAI_BASE_URL conventionally already ends in /v1 (LM Studio docs say
    http://localhost:1234/v1); we always re-append /v1 when building the
    OpenAI client, so a kept suffix would produce /v1/v1 and 404.
    """
    host = raw.strip().rstrip("/")
    if host.lower().endswith("/v1"):
        host = host[: -len("/v1")].rstrip("/")
    return host


HOST = _normalize_host(
    os.environ.get("LOCAL_LLM_HOST")
    or os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("OLLAMA_HOST")
    or "http://127.0.0.1:11434"
)

# --- request timeout -------------------------------------------------------
_DEFAULT_TIMEOUT = 300.0
_MIN_TIMEOUT = 0.1


def _parse_timeout(raw: str | None) -> float:
    """Parse LOCAL_LLM_TIMEOUT defensively.

    A bad override (empty, non-numeric, zero, negative, inf/nan) must never
    crash module import — fall back to the safe default instead.
    """
    if raw is None or not raw.strip():
        return _DEFAULT_TIMEOUT
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT
    if not math.isfinite(val) or val < _MIN_TIMEOUT:
        return _DEFAULT_TIMEOUT
    return val


_REQUEST_TIMEOUT = _parse_timeout(os.environ.get("LOCAL_LLM_TIMEOUT"))

# --- per-host OpenAI clients ------------------------------------------------
# api_key is a placeholder; the local runtime ignores it but the SDK
# refuses to construct a client without one.
# max_retries=0: cold-load failures are deterministic (VRAM page-in), not
# transient — silent SDK retries would multiply wall-clock 3x with no benefit.
_clients: dict[str, OpenAI] = {}


def _client_for(host: str) -> OpenAI:
    if host not in _clients:
        _clients[host] = OpenAI(
            base_url=f"{host}/v1", api_key="local", max_retries=0
        )
    return _clients[host]


def _resolve_host(host: str) -> str:
    """Per-call host override; empty string means the configured default."""
    return _normalize_host(host) if host.strip() else HOST


# --- error classification ---------------------------------------------------
def _describe_error(e: Exception, host: str, model: str = "") -> str:
    """One ERROR line with a cause class and a one-line fix hint, so a
    calling agent can self-recover instead of guessing."""
    low = str(e).lower()
    if isinstance(e, requests.exceptions.ConnectTimeout) or "timed out" in low or "timeout" in low:
        return (
            f"ERROR (timeout): {e}\n"
            f"HINT: call exceeded {_REQUEST_TIMEOUT:.0f}s — cold VRAM load? "
            f"Raise LOCAL_LLM_TIMEOUT or retry once the model is warm."
        )
    if isinstance(e, requests.exceptions.ConnectionError) or "connection" in low or "connect" in low:
        return (
            f"ERROR (connection): {e}\n"
            f"HINT: runtime not reachable at {host} — is it running? "
            f"Check LOCAL_LLM_HOST / OPENAI_BASE_URL / OLLAMA_HOST."
        )
    if "not found" in low or "404" in low or "does not exist" in low:
        return (
            f"ERROR (model-not-found): {e}\n"
            f"HINT: '{model or '?'}' is not installed — call list_models "
            f"for the exact installed tags."
        )
    if "does not support chat" in low or "embedding" in low:
        return (
            f"ERROR (not-a-chat-model): {e}\n"
            f"HINT: '{model or '?'}' looks like an embedding model — pick a "
            f"chat model from list_models instead."
        )
    return f"ERROR: {e}"


def _is_embedding_model(name: str) -> bool:
    """Heuristic filter for auto-select: embedding models 400 on /chat/completions."""
    return "embed" in name.lower()


# --- model listing -----------------------------------------------------------
def _list_local_models(host: str) -> tuple[list[dict], str]:
    """Ask the runtime which models are installed right now.

    Returns (models, style) where style is 'ollama' (rich metadata from
    /api/tags) or 'openai' (names only from /v1/models). Tries Ollama first
    because its metadata is strictly richer; anything else falls through to
    the OpenAI standard endpoint. Raises if both fail.
    """
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        r.raise_for_status()
        return r.json().get("models", []), "ollama"
    except Exception:
        pass
    r = requests.get(f"{host}/v1/models", timeout=5)
    r.raise_for_status()
    data = r.json().get("data", [])
    return [{"name": m.get("id", "?")} for m in data], "openai"


@mcp.tool()
def list_models(host: str = "") -> str:
    """List locally-installed OSS models served by the local runtime.

    Works with Ollama (rich metadata: parameter count, quantization, on-disk
    size) and any OpenAI-compatible runtime (LM Studio, vLLM, llama.cpp —
    names only). The list is fetched live, so freshly-pulled or
    freshly-deleted models are reflected immediately.

    Args:
        host: Optional runtime base URL override (e.g. http://localhost:1234).
              Empty = the configured default.
    """
    h = _resolve_host(host)
    try:
        models, style = _list_local_models(h)
    except Exception as e:
        return _describe_error(e, h)
    if not models:
        return "(no models installed on local runtime)"
    lines = []
    for m in models:
        name = str(m.get("name", "?"))
        if style == "ollama":
            size_gb = (m.get("size") or 0) / (1024**3)
            details = m.get("details", {}) or {}
            params = str(details.get("parameter_size") or "?")
            quant = str(details.get("quantization_level") or "?")
            family = str(details.get("family") or "?")
            lines.append(
                f"{name:30s}  {params:>8s}  {quant:>10s}  {size_gb:5.1f} GB  ({family})"
            )
        else:
            lines.append(name)
    if style == "openai":
        lines.append("(runtime reports names only — no size/quant metadata)")
    return "\n".join(lines)


@mcp.tool()
def health(host: str = "") -> str:
    """Check the local runtime: reachability, endpoint style, model count, latency.

    Use this first when any other tool returns ERROR — it tells you whether
    the runtime is down, which API flavor it speaks, and how fast it answers.

    Args:
        host: Optional runtime base URL override. Empty = configured default.
    """
    h = _resolve_host(host)
    t0 = time.perf_counter()
    try:
        models, style = _list_local_models(h)
    except Exception as e:
        return _describe_error(e, h)
    ms = (time.perf_counter() - t0) * 1000
    lines = [
        f"OK {h}",
        f"endpoint style : {style}",
        f"models installed: {len(models)}",
        f"list latency   : {ms:.0f} ms",
        f"call timeout   : {_REQUEST_TIMEOUT:.0f}s (LOCAL_LLM_TIMEOUT)",
    ]
    if style == "ollama":
        # Loaded-in-VRAM models — best-effort, Ollama-only endpoint.
        try:
            r = requests.get(f"{h}/api/ps", timeout=5)
            r.raise_for_status()
            loaded = [str(m.get("name", "?")) for m in r.json().get("models", [])]
            lines.append(f"loaded in VRAM : {', '.join(loaded) if loaded else '(none — first call will cold-load)'}")
        except Exception:
            pass
    return "\n".join(lines)


def _chat_once(
    host: str,
    model: str,
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = _client_for(host).chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=_REQUEST_TIMEOUT,
    )
    if not resp.choices:
        return "ERROR (empty-response): model returned an empty choices list"
    return resp.choices[0].message.content or ""


@mcp.tool()
def local_chat(
    model: str,
    prompt: str,
    system: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.5,
    host: str = "",
) -> str:
    """Send a prompt to a locally-hosted OSS LLM and return the response.

    Args:
        model: Model tag as returned by list_models (e.g. 'gemma4:e4b').
        prompt: The user prompt.
        system: Optional system prompt.
        max_tokens: Max completion tokens. Default 2048.
        temperature: Sampling temperature 0.0-1.0. Default 0.5.
        host: Optional runtime base URL override. Empty = configured default.

    Returns:
        The model's reply as a string, or an ERROR line if the call failed.
    """
    h = _resolve_host(host)
    try:
        return _chat_once(h, model, prompt, system, max_tokens, temperature)
    except Exception as e:
        return _describe_error(e, h, model)


@mcp.tool()
def local_compare(
    prompt: str,
    models: list[str] | None = None,
    system: str = "",
    max_tokens: int = 1024,
    host: str = "",
) -> str:
    """Run the same prompt across multiple local OSS models for ensemble review.

    Calls run in parallel — on runtimes with true concurrency (vLLM,
    LM Studio) wall-clock is the slowest single model, not the sum.

    Args:
        prompt: The user prompt.
        models: List of model tags. If omitted, defaults to the smallest
                three currently-installed models (by on-disk size) on Ollama;
                on runtimes without size metadata, the first three listed.
                Embedding models are excluded from auto-select.
        system: Optional system prompt applied to every call.
        max_tokens: Max completion tokens per model. Default 1024.
        host: Optional runtime base URL override. Empty = configured default.

    Returns:
        Concatenated, labeled outputs from each model.
    """
    h = _resolve_host(host)
    chosen = models
    note = ""
    if not chosen:
        try:
            installed, style = _list_local_models(h)
        except Exception as e:
            return _describe_error(e, h)
        if not installed:
            return "(no models installed on local runtime)"
        installed = [m for m in installed if not _is_embedding_model(str(m.get("name", "")))]
        if not installed:
            return "(only embedding models installed — none support chat)"
        if style == "ollama":
            installed.sort(key=lambda m: m.get("size") or 0)
        else:
            note = "(auto-selected first 3 — runtime reports no size metadata)\n\n"
        chosen = [str(m["name"]) for m in installed[:3]]

    def _one(name: str) -> str:
        try:
            out = _chat_once(h, name, prompt, system, max_tokens, 0.5)
        except Exception as e:
            out = _describe_error(e, h, name)
        return f"=== {name} ===\n{out}"

    with ThreadPoolExecutor(max_workers=min(len(chosen), 8)) as pool:
        parts = list(pool.map(_one, chosen))
    return note + "\n\n".join(parts)


if __name__ == "__main__":
    mcp.run()
