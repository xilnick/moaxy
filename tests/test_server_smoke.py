"""End-to-end smoke tests for the moaxy FastAPI proxy.

This file groups the M1 smoke tests that exercise the full data plane of
the proxy in two distinct ways:

1. **In-process (always on)** — the FastAPI app is built in-process and
   driven via :class:`httpx.AsyncClient` with :class:`httpx.ASGITransport`.
   The adapter is replaced with a scripted handler so the tests are
   hermetic and never need a running Ollama. These tests cover the basic
   flows (VAL-HTTP-001/003/005/006/007/008/009/011/012/013/014/016/017/024):
   ``/health`` 200, ``/v1/models`` 200, ``/v1/chat/completions`` 200 with a
   valid OpenAI-shaped response, the 4xx error envelopes, content-type
   guarantees, request id propagation, and concurrent request isolation.

2. **Real-Ollama (gated by env var)** — when ``MOAXY_REAL_OLLAMA=1`` is
   set, an additional test boots the proxy via ``uvicorn`` in a background
   subprocess, waits for ``/health`` to come up, then hits
   ``/v1/chat/completions`` with a real Ollama-backed request and asserts
   a non-empty ``choices[0].message.content``. The test is skipped
   automatically when the env var is unset.

The tests run in two phases. The in-process suite runs on every CI run
and is the only required signal for the M1 milestone. The real-Ollama
test is opt-in.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from moaxy.adapters.registry import AdapterRegistry
from moaxy.server.app import create_app
from moaxy.server.middleware import REQUEST_ID_HEADER
from tests.conftest import (
    make_config,
    make_json_response,
    make_ollama_adapter,
    make_ollama_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────────────
# In-process /health smoke (VAL-HTTP-001, 016, 017, 026)
# ────────────────────────────────────────────────────────────────────


class TestHealthSmoke:
    """In-process smoke: ``GET /health`` returns 200 with a stable JSON body."""

    @pytest.mark.asyncio
    async def test_health_200_status_ok_json_body(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body == {"status": "ok"}
        assert "traceback" not in response.text.lower()

    @pytest.mark.asyncio
    async def test_health_request_id_is_present_and_stable(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.get("/health")
            r2 = await client.get("/health")
        assert r1.headers[REQUEST_ID_HEADER] and r2.headers[REQUEST_ID_HEADER]
        assert r1.headers[REQUEST_ID_HEADER] != r2.headers[REQUEST_ID_HEADER]


# ────────────────────────────────────────────────────────────────────
# In-process /v1/models smoke (VAL-HTTP-003, 004, 016)
# ────────────────────────────────────────────────────────────────────


class TestModelsSmoke:
    """In-process smoke: ``GET /v1/models`` returns the OpenAI-shaped list."""

    @pytest.mark.asyncio
    async def test_models_200_openai_shape_with_non_empty_data(self):
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            RouteConfig,
        )
        from moaxy.models.config import (
            RouteMatch as ConfigRouteMatch,
        )

        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="b", adapter="ollama", base_url="http://x")],
            routes=[
                RouteConfig(
                    name="r",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="b",
                    aliases={"coder-pro": "minimax-m3:cloud"},
                )
            ],
        )
        app = create_app(config=cfg)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body["object"] == "list"
        data = body["data"]
        assert isinstance(data, list) and len(data) >= 1
        ids = {entry["id"] for entry in data}
        assert "coder-pro" in ids
        assert "minimax-m3:cloud" in ids
        for entry in data:
            assert entry["object"] == "model"
            assert isinstance(entry["id"], str) and entry["id"]


# ────────────────────────────────────────────────────────────────────
# In-process /v1/chat/completions success (VAL-HTTP-005, 006, 007, 016, 018)
# ────────────────────────────────────────────────────────────────────


class TestChatCompletionsSuccessSmoke:
    """In-process smoke: ``POST /v1/chat/completions`` returns 200 with a valid body."""

    @pytest.mark.asyncio
    async def test_chat_completions_200_valid_ollama_response(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return make_json_response(
                make_ollama_payload(content="pong", model="minimax-m2.7:cloud")
            )

        adapter = make_ollama_adapter(handler)
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=make_config(), adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m2.7:cloud",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body["object"] == "chat.completion"
        assert "id" in body and body["id"]
        assert "created" in body and isinstance(body["created"], int)
        assert body["model"] == "minimax-m2.7:cloud"
        assert isinstance(body["choices"], list) and len(body["choices"]) >= 1
        first_choice = body["choices"][0]
        assert first_choice["message"]["role"] == "assistant"
        assert first_choice["message"]["content"] == "pong"
        usage = body["usage"]
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)
        assert (
            usage["total_tokens"]
            >= max(usage["prompt_tokens"], usage["completion_tokens"])
        )

    @pytest.mark.asyncio
    async def test_chat_completions_with_alias_echoes_original_and_sets_resolved_header(
        self,
    ):
        from moaxy.models.config import (
            AdapterConfig,
            MoaxyConfig,
            RouteConfig,
        )
        from moaxy.models.config import (
            RouteMatch as ConfigRouteMatch,
        )

        async def handler(_request: httpx.Request) -> httpx.Response:
            return make_json_response(
                make_ollama_payload(model="minimax-m3:cloud", content="ack")
            )

        adapter = make_ollama_adapter(handler)
        cfg = MoaxyConfig(
            backends=[AdapterConfig(name="b", adapter="ollama", base_url="http://x")],
            routes=[
                RouteConfig(
                    name="r",
                    match=ConfigRouteMatch(model="*", path="/v1/chat/completions"),
                    backend="b",
                    aliases={"coder-pro": "minimax-m3:cloud"},
                )
            ],
        )
        registry = AdapterRegistry({"b": adapter})
        app = create_app(config=cfg, adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "coder-pro",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "coder-pro"
        assert response.headers.get("x-moaxy-alias-resolved") == "minimax-m3:cloud"
        assert body["choices"][0]["message"]["content"] == "ack"


# ────────────────────────────────────────────────────────────────────
# In-process 4xx error envelopes (VAL-HTTP-008, 009, 011, 012, 013, 014, 026)
# ────────────────────────────────────────────────────────────────────


class TestErrorEnvelopesSmoke:
    """In-process smoke: the proxy returns structured 4xx envelopes."""

    @pytest.mark.asyncio
    async def test_empty_messages_returns_400(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "minimax-m2.7:cloud", "messages": []},
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert "error" in body
        assert "messages" in body["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_missing_model_returns_400_no_traceback(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 400
        body = response.json()
        assert "error" in body
        assert "model" in body["error"]["message"].lower()
        assert "traceback" not in response.text.lower()

    @pytest.mark.asyncio
    async def test_missing_content_type_returns_415(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                content=b'{"model":"x","messages":[{"role":"user","content":"hi"}]}',
            )
        assert response.status_code == 415
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert "error" in body

    @pytest.mark.asyncio
    async def test_malformed_json_returns_400(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                content=b"{not-json",
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert "error" in body
        assert "traceback" not in response.text.lower()

    @pytest.mark.asyncio
    async def test_wrong_method_get_on_chat_completions_returns_405(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/chat/completions")
        assert response.status_code == 405
        body = response.json()
        assert "error" in body
        allow = response.headers.get("allow") or response.headers.get("Allow")
        assert allow and "POST" in allow.upper()

    @pytest.mark.asyncio
    async def test_wrong_method_post_on_health_returns_405(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/health")
        assert response.status_code == 405
        body = response.json()
        assert "error" in body

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404_json(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/nonexistent")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert "error" in body


# ────────────────────────────────────────────────────────────────────
# In-process content-type guarantees (VAL-HTTP-016)
# ────────────────────────────────────────────────────────────────────


class TestContentTypeSmoke:
    """In-process smoke: every response (success AND error) is JSON."""

    @pytest.mark.asyncio
    async def test_success_responses_have_json_content_type(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return make_json_response(make_ollama_payload(content="ok"))

        adapter = make_ollama_adapter(handler)
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=make_config(), adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            for path in ("/health", "/v1/models"):
                r = await client.get(path)
                assert r.headers["content-type"].startswith("application/json"), path
            r = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "minimax-m2.7:cloud",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Content-Type": "application/json"},
            )
            assert r.headers["content-type"].startswith("application/json")

    @pytest.mark.asyncio
    async def test_error_responses_have_json_content_type(self):
        app = create_app(config=make_config())
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1 = await client.get("/nonexistent")
            r2 = await client.post("/health")
            r3 = await client.post(
                "/v1/chat/completions",
                json={"model": "x", "messages": []},
                headers={"Content-Type": "application/json"},
            )
        for r in (r1, r2, r3):
            assert r.headers["content-type"].startswith("application/json"), r.status_code


# ────────────────────────────────────────────────────────────────────
# In-process concurrent request isolation (VAL-HTTP-024)
# ────────────────────────────────────────────────────────────────────


class TestConcurrentRequestsSmoke:
    """In-process smoke: two parallel requests do not cross-contaminate."""

    @pytest.mark.asyncio
    async def test_two_concurrent_requests_have_distinct_ids_and_responses(self):
        seen: list[str] = []
        seen_lock = asyncio.Lock()

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            content = body["messages"][-1]["content"]
            async with seen_lock:
                seen.append(content)
            await asyncio.sleep(0.05)
            return make_json_response(
                make_ollama_payload(content=f"echo:{content}")
            )

        adapter = make_ollama_adapter(handler)
        registry = AdapterRegistry({"olloma-local": adapter})
        app = create_app(config=make_config(), adapters=registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r1, r2 = await asyncio.gather(
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "minimax-m2.7:cloud",
                        "messages": [{"role": "user", "content": "alpha"}],
                    },
                    headers={"Content-Type": "application/json"},
                ),
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "minimax-m2.7:cloud",
                        "messages": [{"role": "user", "content": "beta"}],
                    },
                    headers={"Content-Type": "application/json"},
                ),
            )
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.headers[REQUEST_ID_HEADER] != r2.headers[REQUEST_ID_HEADER]
        body1, body2 = r1.json(), r2.json()
        assert body1["choices"][0]["message"]["content"] != body2["choices"][0]["message"]["content"]
        assert "traceback" not in r1.text.lower()
        assert "traceback" not in r2.text.lower()


# ────────────────────────────────────────────────────────────────────
# Real-Ollama smoke (gated by MOAXY_REAL_OLLAMA=1)
# ────────────────────────────────────────────────────────────────────


def _pick_free_port() -> int:
    """Bind to port 0 to let the kernel pick a free port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ollama_reachable() -> bool:
    """Return True if a local Ollama is reachable on 127.0.0.1:11434."""
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=1.0):
            return True
    except OSError:
        return False


REAL_OLLAMA_ENABLED = os.environ.get("MOAXY_REAL_OLLAMA") == "1"
REAL_OLLAMA_REASON = (
    "set MOAXY_REAL_OLLAMA=1 to run the real-Ollama smoke test "
    "(also requires Ollama running on 127.0.0.1:11434)"
)


@pytest.mark.skipif(not REAL_OLLAMA_ENABLED, reason=REAL_OLLAMA_REASON)
@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama is not reachable on 127.0.0.1:11434")
class TestRealOllamaSmoke:
    """End-to-end smoke against a live uvicorn + local Ollama.

    Boots the proxy in a background subprocess on a free port, waits for
    ``/health`` to return 200, and hits ``/v1/chat/completions`` with a
    real Ollama-backed request. The test is gated by the env var
    ``MOAXY_REAL_OLLAMA=1``.
    """

    @pytest.fixture
    def proxy_url(self, tmp_path: Path) -> str:
        """Boot uvicorn on a free port and tear it down at fixture exit."""
        port = _pick_free_port()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        env["MOAXY_CONFIG_PATH"] = str(tmp_path / "config.yaml")
        # Write a minimal config that points at the local Ollama. The
        # default ``127.0.0.1:8765`` cannot be used (the user-testing
        # validator may already be using it) and we do not want to race
        # with another worker.
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "server:\n"
            "  listen: '127.0.0.1'\n"
            "  port: 8765\n"
            "  log_level: 'warning'\n"
            "  plugins_dir: 'plugins'\n"
            "  request_timeout_s: 60.0\n"
            "backends:\n"
            "  - name: 'ollama-local'\n"
            "    adapter: 'ollama'\n"
            "    base_url: 'http://127.0.0.1:11434'\n"
            "    timeout: 30.0\n"
            "routes:\n"
            "  - name: 'plain'\n"
            "    match:\n"
            "      model: '*'\n"
            "      path: '/v1/chat/completions'\n"
            "    strategy: 'single'\n"
            "    backend: 'ollama-local'\n"
        )
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "moaxy.server.app:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            env=env,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.time() + 30.0
            while time.time() < deadline:
                if proc.poll() is not None:
                    stdout = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
                    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                    raise RuntimeError(
                        f"uvicorn exited prematurely (code {proc.returncode}):\n"
                        f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                    )
                with contextlib.suppress(Exception):
                    r = httpx.get(f"{url}/health", timeout=1.0)
                    if r.status_code == 200:
                        break
                time.sleep(0.2)
            else:
                proc.terminate()
                raise RuntimeError(f"uvicorn did not become healthy at {url}/health within 30s")
            yield url
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)

    def test_real_ollama_health_and_chat_completions(self, proxy_url: str) -> None:
        """Boot the proxy and get a real Ollama chat completion back."""
        with httpx.Client(timeout=30.0) as sync_client:
            health = sync_client.get(f"{proxy_url}/health")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}
            models = sync_client.get(f"{proxy_url}/v1/models")
            assert models.status_code == 200
            assert models.headers["content-type"].startswith("application/json")
            assert models.json()["object"] == "list"
            assert isinstance(models.json()["data"], list)
            # Real chat completion. We use ``kimi-k2.6:cloud`` because it
            # returns clean ``content`` (other cloud models sometimes
            # stream the answer through the reasoning field when the
            # budget is small). max_tokens=200 guarantees a non-empty
            # final answer.
            completion = sync_client.post(
                f"{proxy_url}/v1/chat/completions",
                json={
                    "model": "kimi-k2.6:cloud",
                    "messages": [{"role": "user", "content": "say hi in 1 word"}],
                    "max_tokens": 200,
                },
                headers={"Content-Type": "application/json"},
            )
        assert completion.status_code == 200, completion.text
        assert completion.headers["content-type"].startswith("application/json")
        body = completion.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "kimi-k2.6:cloud"
        assert isinstance(body["choices"], list) and len(body["choices"]) >= 1
        content = body["choices"][0]["message"]["content"]
        assert isinstance(content, str) and content.strip() != "", (
            f"expected non-empty content; got {content!r}"
        )
        assert "traceback" not in completion.text.lower()
        usage = body["usage"]
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)
