# local-llm-mcp

MCP server that exposes locally-hosted OSS LLMs (Ollama, LM Studio, vLLM, any OpenAI-compatible runtime) as tools for Claude Code, Claude Desktop, and Codex CLI.

Pairs with cloud-model MCPs like [nvidia-models-mcp](https://github.com/NAJEMWEHBE/nvidia-models-mcp) for tiered workflows: small/fast local models for high-frequency mechanical tasks, big cloud models for heavy reasoning.

## Tools

| Tool | Purpose |
|------|---------|
| `list_models` | Fetches the live model inventory from the runtime (`/api/tags`). |
| `local_chat` | Single-model completion. Args: `model`, `prompt`, optional `system`, `max_tokens`, `temperature`. |
| `local_compare` | Ensemble fan-out: runs the same prompt across multiple local models and returns labeled outputs. Defaults to the smallest three installed models. |

## Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (fast Python package manager)
- A local OpenAI-compatible runtime running. Tested with:
  - [Ollama](https://ollama.com) — default `http://127.0.0.1:11434`
  - [LM Studio](https://lmstudio.ai) — default `http://127.0.0.1:1234` (set `OLLAMA_HOST` accordingly)
  - vLLM, llama.cpp server, text-generation-webui — any runtime exposing `/v1/chat/completions` and `/api/tags`

## Install

### Claude Desktop (DXT one-click)

1. Download the latest `local-llm.dxt` from [Releases](https://github.com/NAJEMWEHBE/local-llm-mcp/releases).
2. Drag-drop into Claude Desktop → Settings → Extensions.
3. Restart Claude Desktop.

### Claude Code (manual)

Add to `~/.claude.json` (top level):

```json
{
  "mcpServers": {
    "local-llm": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/local-llm-mcp",
        "run",
        "server.py"
      ],
      "env": {
        "LOCAL_LLM_HOST": "http://127.0.0.1:11434"
      }
    }
  }
}
```

Restart Claude Code. Tools appear as `mcp__local-llm__list_models`, `mcp__local-llm__local_chat`, `mcp__local-llm__local_compare`.

### Codex CLI

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.local-llm]
command = "uv"
args = ["--directory", "/absolute/path/to/local-llm-mcp", "run", "server.py"]
env = { LOCAL_LLM_HOST = "http://127.0.0.1:11434" }
```

## Configuration

The host URL is resolved in this order (first set wins):

| Priority | Env var | Notes |
|---------|---------|-------|
| 1 | `LOCAL_LLM_HOST` | Preferred. Runtime-agnostic name. |
| 2 | `OPENAI_BASE_URL` | Common in OpenAI-SDK ecosystems. |
| 3 | `OLLAMA_HOST` | Legacy / Ollama-default. Kept for backward compatibility. |
| 4 | `http://127.0.0.1:11434` | Built-in fallback (Ollama default port). |

Examples:
- Ollama: `LOCAL_LLM_HOST=http://127.0.0.1:11434` (or just leave unset)
- LM Studio: `LOCAL_LLM_HOST=http://127.0.0.1:1234`
- vLLM: `LOCAL_LLM_HOST=http://127.0.0.1:8000`
- Remote box on LAN: `LOCAL_LLM_HOST=http://192.168.1.50:11434`

The server hits two endpoints on the host: `GET /api/tags` (model inventory) and `POST /v1/chat/completions` (inference). Any runtime exposing both works.

## Smoke test

```bash
cd /path/to/local-llm-mcp
uv sync
uv run python -c "from server import list_models; print(list_models())"
```

The first run pulls dependencies into `.venv`. Subsequent runs reuse them.

## Multi-agent example

With both `local-llm-mcp` and a cloud MCP wired, you can build a delegation flow where Claude orchestrates and the local models handle bulk mechanical work:

> User: Generate Pydantic models for these 12 JSON schemas.
>
> Claude: Delegating bulk model generation to `qwen3:27b` via `local_chat`. Verifying field types match schemas.
>
> [Calls `mcp__local-llm__local_chat(model="qwen3:27b", prompt="<schemas + spec>", temperature=0.2)`]
>
> Claude: Qwen wrote 12 models. One missing `Optional` repaired. Writing files.

## Building a DXT release

```bash
cd /path/to/local-llm-mcp
python -c "import zipfile, pathlib; src=pathlib.Path('.'); out=pathlib.Path('local-llm.dxt'); files=['manifest.json','server.py','pyproject.toml','README.md']; z=zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED); [z.write(src/f,arcname=f) for f in files]; z.close(); print(out.stat().st_size,'bytes')"
```

## License

MIT — see [LICENSE](LICENSE).
