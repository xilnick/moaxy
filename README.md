# moaxy — Mixture of Agent Proxy

Extensible proxy for routing AI/LLM API requests across multiple providers. Single endpoint, any model, any backend.

Send OpenAI-compatible chat completion requests to moaxy and it routes them to the right backend — Ollama, OpenAI, or any OpenAI-compatible API — based on model name, path, headers, or expression rules.

## Features

- **Plugin-based architecture** — routers, transformers, auth handlers, and middleware as swappable plugins
- **Multi-backend routing** — single, weighted, and round-robin strategies
- **Admin API** — manage backends, routes, and API keys at runtime
- **Request/response transformations** — rewrite models, inject headers, strip fields
- **Docker support** — multi-stage build running as non-root user with health checks
- **${ENV_VAR} substitution** — keep secrets out of config files

## Installation

**Requirements:** Python 3.11+

**Development install (editable):**

```bash
git clone https://github.com/xilnick/moaxy.git
cd moaxy
pip install -e ".[dev]"
```

## Quick Start

1. Copy the example config:

   ```bash
   cp config.example.yaml config.yaml
   ```

2. Set the admin API key:

   ```bash
   export MOAXY_ADMIN_API_KEY="your-secret-key"
   ```

3. Verify the installation:

   ```bash
   python -c "import moaxy; print(moaxy.__version__)"
   ```

4. Run tests:

   ```bash
   pytest
   ```

## Configuration Reference

moaxy discovers its config file automatically, checked in this order:

1. `MOAXY_CONFIG_PATH` environment variable (explicit path)
2. `config.yaml` in the working directory
3. `config.yml` in the working directory
4. `config.json` in the working directory
5. Defaults (empty backends/routes — only health and admin endpoints work)

All string values support `${ENV_VAR}` substitution (e.g. `api_key: "${OPENAI_API_KEY}"`).

### Server

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `listen` | string | `0.0.0.0` | Address to bind |
| `port` | integer | `8000` | Port to bind |
| `log_level` | string | `info` | `debug`, `info`, `warning`, or `error` |

### Backends

Backends define upstream API providers. Each backend has a unique `name`.

```yaml
backends:
  - name: "ollama-local"
    adapter: "ollama"
    base_url: "http://localhost:11434"
    timeout: 30.0
```

**Adapter types:**

| Adapter | Description | Default `base_url` |
|---------|-------------|-------------------|
| `ollama` | Local Ollama server | `http://localhost:11434` |
| `openai` | OpenAI or any OpenAI-compatible API | `https://api.openai.com/v1` |

### Routes

Routes define how incoming requests are matched and dispatched to backends. Evaluated top-to-bottom — first match wins.

```yaml
routes:
  - name: "catch-all"
    match:
      model: "*"
      path: "/*"
    strategy: "single"
    backend: "ollama-local"
```

**Routing strategies:**

| Strategy | Description |
|----------|-------------|
| `single` | Always routes to one backend. Requires `backend`. |
| `weighted` | Randomly selects from `backends` list based on weights. |
| `round_robin` | Cycles through `backends` list in order. |

### Auth

API-key authentication for proxy endpoints. Disabled by default.

```yaml
auth:
  enabled: true
  exempt_paths:
    - "/health"
  header_names:
    - "X-API-Key"
  api_keys:
    - key_id: "admin"
      key_value: "${MOAXY_ADMIN_API_KEY}"
      roles: ["admin"]
      scopes: ["*"]
```

Keys use SHA-256 hashing and timing-safe comparison.

## Plugin System

moaxy is built around a plugin architecture. Each plugin is a Python class extending `moaxy.plugins.Plugin` with a declared `PluginType`:

- **Router** (`PluginType.ROUTER`) — decides which backend handles a request
- **Transformer** (`PluginType.TRANSFORMER`) — modifies requests/responses in-flight
- **Auth** (`PluginType.AUTH`) — authenticates and authorizes requests
- **Middleware** (`PluginType.MIDDLEWARE`) — intercepts the full request context

### Lifecycle hooks

| Hook | When |
|------|------|
| `init()` | Plugin is loaded and registered |
| `validate()` | After init, before first use — returns list of errors |
| `process(context)` | Every request — plugin-specific logic |
| `cleanup()` | Proxy shutdown — release resources |

### Example custom plugin

```python
from typing import Any
from moaxy.plugins import Plugin, PluginType

class HeaderLogger(Plugin):
    name = "header-logger"
    version = "1.0.0"
    plugin_type = PluginType.MIDDLEWARE

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        headers = context.get("headers", {})
        context["logged_headers"] = list(headers.keys())
        return context
```

Place your plugin in `plugins/<name>/plugin.py` — it will be auto-discovered at startup.

### Built-in plugins

Planned built-in plugins:
- **RateLimiter** — token-bucket rate limiting per API key
- **Cors** — CORS header injection
- **RequestLogger** — request ID generation and timing

## Deployment

### Docker

A multi-stage `Dockerfile` is included:

```bash
docker build -t moaxy .
docker run -p 8000:8000 \
  -e MOAXY_ADMIN_API_KEY="your-secret-key" \
  moaxy
```

The Docker image:
- Runs as a non-root `moaxy` user
- Exposes port 8000
- Includes a health check at `http://localhost:8000/health` every 30s

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MOAXY_CONFIG_PATH` | Path to the config file |
| `MOAXY_ADMIN_API_KEY` | Admin API key for `/admin` endpoints |

All config values support `${ENV_VAR}` substitution.

## Architecture

```
Client Request
    │
    ▼
┌──────────────────────────────────┐
│  Auth Middleware (if enabled)     │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Plugin Pipeline                 │
│  - Router plugins                │
│  - Transformer plugins           │
│  - Middleware plugins            │
└──────────────┬───────────────────┘
               ▼
┌──────────────────────────────────┐
│  Adapter.send_request            │
└──────────────┬───────────────────┘
               ▼
         Client Response
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
