# MOAXY — Mixture of Agent Proxy

Extensible, pluggable proxy for routing AI agent requests across multiple providers.

## Overview

MOAXY is a proxy framework built around a plugin architecture. Plugins handle routing, request/response transformation, authentication, and middleware — making it easy to customize and extend the proxy pipeline without modifying core code.

## Plugin System

Plugins declare a type (`router`, `transformer`, `auth`, `middleware`) and implement a `process(context)` method. Plugins are discovered from configured paths and managed through a lifecycle of init → validate → process → cleanup.

### Plugin Types

| Type | Description |
|------|-------------|
| `router` | Decides where to send incoming requests |
| `transformer` | Modifies request or response payloads |
| `auth` | Validates credentials and enforces access |
| `middleware` | Intercepts and modifies the full request context |

### Lifecycle

1. **Init** — Plugin is loaded and resources are set up
2. **Validate** — Plugin checks its configuration and readiness
3. **Process** — Plugin handles requests during runtime
4. **Cleanup** — Plugin releases resources on shutdown

### Built-in Example Plugins

- `plugins/demo_router/` — Demonstrates routing to named backends
- `plugins/demo_transformer/` — Demonstrates request/response transformation

## Quick Start

### Development Setup

```bash
git clone https://github.com/xilnick/moaxy
cd moaxy
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Create a Custom Plugin

```python
from moaxy.plugins.base import Plugin
from moaxy.plugins.types import PluginType
from typing import Any

class MyPlugin(Plugin):
    name = "my-plugin"
    version = "1.0.0"
    plugin_type = PluginType.TRANSFORMER

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        context["x-custom"] = "hello from my-plugin"
        return context
```

## Docker

```bash
docker build -t moaxy .
docker run -p 8000:8000 moaxy
```

## Configuration

Copy the example config and customize:

```bash
cp config.example.yaml config.yaml
```

Configuration supports `${ENV_VAR}` substitution for values like API keys.

## License

MIT
