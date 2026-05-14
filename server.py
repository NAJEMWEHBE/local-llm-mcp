#!/usr/bin/env python
"""Local OSS LLM MCP server.

Exposes locally-installed OSS models (served by any local OpenAI-compatible
runtime: Ollama, LM Studio, vLLM, llama.cpp server, text-generation-webui,
or anything else exposing /v1/chat/completions and /api/tags) as MCP tools
that Claude Code / Claude Desktop / Codex / Copilot can call.

Host resolution order:
    1. LOCAL_LLM_HOST     — preferred, runtime-agnostic name
    2. OPENAI_BASE_URL    — common in OpenAI-SDK ecosystems
    3. OLLAMA_HOST        — legacy / Ollama-default name (kept for compat)
    4. http://127.0.0.1:11434 (Ollama default)
"""

import os
import sys

import requests
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

mcp = FastMCP("local-llm")

HOST = (
    os.environ.get("LOCAL_LLM_HOST")
    or os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("OLLAMA_HOST")
    or "http://127.0.0.1:11434"
).rstrip("/")

# api_key is a placeholder; the local runtime ignores it but the SDK
# refuses to construct a client without one.
client = OpenAI(base_url=f"{HOST}/v1", api_key="local")


def _list_local_models() -> list[dict]:
    """Ask the runtime which models are installed right now."""
    r = requests.get(f"{HOST}/api/tags", timeout=5)
    r.raise_for_status()
    return r.json().get("models", [])


@mcp.tool()
def list_models() -> str:
    """List locally-installed OSS models served by the local runtime.

    Returns one line per model: name, parameter count, quantization, on-disk
    size in GB. The list is fetched live from the runtime, so freshly-pulled
    or freshly-deleted models are reflected immediately.
    """
    try:
        models = _list_local_models()
    except Exception as e:
        return f"ERROR contacting {HOST}/api/tags: {e}"
    if not models:
        return "(no models installed on local runtime)"
    lines = []
    for m in models:
        name = m.get("name", "?")
        size_gb = m.get("size", 0) / (1024 ** 3)
        details = m.get("details", {}) or {}
        params = details.get("parameter_size", "?")
        quant = details.get("quantization_level", "?")
        family = details.get("family", "?")
        lines.append(
            f"{name:30s}  {params:>8s}  {quant:>10s}  {size_gb:5.1f} GB  ({family})"
        )
    return "\n".join(lines)


@mcp.tool()
def local_chat(
    model: str,
    prompt: str,
    system: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.5,
) -> str:
    """Send a prompt to a locally-hosted OSS LLM and return the response.

    Args:
        model: Model tag as returned by list_models (e.g. 'gemma4:e4b').
        prompt: The user prompt.
        system: Optional system prompt.
        max_tokens: Max completion tokens. Default 2048.
        temperature: Sampling temperature 0.0-1.0. Default 0.5.

    Returns:
        The model's reply as a string, or an ERROR line if the call failed.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool()
def local_compare(
    prompt: str,
    models: list[str] | None = None,
    system: str = "",
    max_tokens: int = 1024,
) -> str:
    """Run the same prompt across multiple local OSS models for ensemble review.

    Args:
        prompt: The user prompt.
        models: List of model tags. If omitted, defaults to the smallest
                three currently-installed models (by on-disk size), so the
                fan-out completes quickly on consumer GPUs.
        system: Optional system prompt applied to every call.
        max_tokens: Max completion tokens per model. Default 1024.

    Returns:
        Concatenated, labeled outputs from each model.
    """
    chosen = models
    if not chosen:
        try:
            installed = _list_local_models()
        except Exception as e:
            return f"ERROR auto-selecting models from {HOST}/api/tags: {e}"
        installed.sort(key=lambda m: m.get("size", 0))
        chosen = [m["name"] for m in installed[:3]]
        if not chosen:
            return "(no models installed on local runtime)"
    parts: list[str] = []
    for name in chosen:
        try:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            r = client.chat.completions.create(
                model=name,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=0.5,
            )
            content = r.choices[0].message.content or ""
            parts.append(f"=== {name} ===\n{content}")
        except Exception as e:
            parts.append(f"=== {name} ===\nERROR: {e}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    mcp.run()
