"""Tests for the moaxy Pydantic v2 config schema.

Covers: importability of all 12 models, default values, range and literal
rejections, required-field invariants, loopback-only listen, JSON
serialisation, and the canonical example parse.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from moaxy.models.config import (
    AdapterConfig,
    AdvisorConfig,
    ApiKey,
    AuthConfig,
    BackendRef,
    ModelDefaults,
    MoaxyConfig,
    PluginConfig,
    ReflectionConfig,
    RouteConfig,
    RouteMatch,
    ServerConfig,
)


# ── Importability ─────────────────────────────────────────────────────────


class TestImports:
    def test_all_twelve_models_importable(self):
        from moaxy.models.config import (  # noqa: F401
            AdapterConfig,
            AdvisorConfig,
            ApiKey,
            AuthConfig,
            BackendRef,
            ModelDefaults,
            MoaxyConfig,
            PluginConfig,
            ReflectionConfig,
            RouteConfig,
            RouteMatch,
            ServerConfig,
        )

        classes = [
            AdapterConfig,
            AdvisorConfig,
            ApiKey,
            AuthConfig,
            BackendRef,
            ModelDefaults,
            MoaxyConfig,
            PluginConfig,
            ReflectionConfig,
            RouteConfig,
            RouteMatch,
            ServerConfig,
        ]
        assert len(classes) == 12

    def test_models_package_reexports(self):
        import moaxy.models

        for name in (
            "AdapterConfig",
            "AdvisorConfig",
            "ApiKey",
            "AuthConfig",
            "BackendRef",
            "ModelDefaults",
            "MoaxyConfig",
            "PluginConfig",
            "ReflectionConfig",
            "RouteConfig",
            "RouteMatch",
            "ServerConfig",
        ):
            assert hasattr(moaxy.models, name)


# ── ServerConfig ──────────────────────────────────────────────────────────


class TestServerConfig:
    def test_defaults(self):
        s = ServerConfig()
        assert s.listen == "127.0.0.1"
        assert s.port == 8765
        assert s.log_level == "info"
        assert s.plugins_dir == "plugins"
        assert s.request_timeout_s == 60.0

    def test_accepts_explicit_values(self):
        s = ServerConfig(
            listen="127.0.0.1",
            port=9000,
            log_level="debug",
            plugins_dir="/tmp/p",
            request_timeout_s=5.0,
        )
        assert s.port == 9000
        assert s.log_level == "debug"

    def test_rejects_zero_port(self):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(port=0)
        assert "port" in str(exc.value)

    def test_rejects_high_port(self):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(port=99999)
        assert "port" in str(exc.value)

    def test_accepts_boundary_ports(self):
        assert ServerConfig(port=1).port == 1
        assert ServerConfig(port=65535).port == 65535

    def test_rejects_zero_zero_zero_zero_listen(self):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(listen="0.0.0.0")
        msg = str(exc.value)
        assert "listen" in msg
        assert "loopback" in msg.lower() or "127.0.0.1" in msg

    def test_rejects_non_loopback_ipv4(self):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(listen="10.0.0.1")
        assert "loopback" in str(exc.value).lower()

    def test_accepts_loopback_addresses(self):
        assert ServerConfig(listen="127.0.0.1").listen == "127.0.0.1"
        assert ServerConfig(listen="::1").listen == "::1"

    def test_rejects_invalid_ip(self):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(listen="not-an-ip")
        assert "loopback" in str(exc.value).lower()

    def test_rejects_unknown_log_level(self):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(log_level="verbose")
        assert "log_level" in str(exc.value)

    def test_rejects_non_positive_timeout(self):
        with pytest.raises(ValidationError):
            ServerConfig(request_timeout_s=0)


# ── AdapterConfig ─────────────────────────────────────────────────────────


class TestAdapterConfig:
    def test_ollama_minimal(self):
        a = AdapterConfig(name="local", adapter="ollama", base_url="http://127.0.0.1:11434")
        assert a.api_key is None
        assert a.timeout == 30.0

    def test_openai_with_key(self):
        a = AdapterConfig(
            name="openai",
            adapter="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-abc",
        )
        assert a.adapter == "openai"
        assert a.api_key == "sk-abc"

    def test_rejects_unknown_adapter(self):
        with pytest.raises(ValidationError) as exc:
            AdapterConfig(name="x", adapter="anthropic", base_url="http://x")
        assert "adapter" in str(exc.value)

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            AdapterConfig(name="", adapter="ollama", base_url="http://x")

    def test_rejects_empty_base_url(self):
        with pytest.raises(ValidationError):
            AdapterConfig(name="x", adapter="ollama", base_url="")


# ── AuthConfig + ApiKey ───────────────────────────────────────────────────


class TestAuthConfig:
    def test_defaults(self):
        a = AuthConfig()
        assert a.enabled is False
        assert a.exempt_paths == ["/health"]
        assert a.header_names == ["X-API-Key", "Authorization"]
        assert a.api_keys == []

    def test_with_api_keys(self):
        a = AuthConfig(
            enabled=True,
            api_keys=[ApiKey(key_id="k1", key_value="v1", roles=["admin"])],
        )
        assert a.enabled is True
        assert a.api_keys[0].key_id == "k1"
        assert a.api_keys[0].roles == ["admin"]


# ── RouteMatch ────────────────────────────────────────────────────────────


class TestRouteMatch:
    def test_basic(self):
        m = RouteMatch(model="*", path="/v1/chat/completions")
        assert m.model == "*"
        assert m.path == "/v1/chat/completions"


# ── ReflectionConfig ──────────────────────────────────────────────────────


class TestReflectionConfig:
    def test_defaults(self):
        r = ReflectionConfig()
        assert r.turns == 0
        assert r.early_exit is True
        assert r.threshold == 0.85
        assert r.parallel is False
        assert r.system_prompt is None
        assert r.system_prompt_file is None

    @pytest.mark.parametrize("turns", [0, 1, 2, 3])
    def test_accepts_in_range_turns(self, turns):
        ReflectionConfig(turns=turns)

    def test_rejects_turns_above_3(self):
        with pytest.raises(ValidationError) as exc:
            ReflectionConfig(turns=4)
        msg = str(exc.value)
        assert "turns" in msg
        assert "3" in msg

    def test_rejects_negative_turns(self):
        with pytest.raises(ValidationError):
            ReflectionConfig(turns=-1)

    def test_rejects_threshold_above_1(self):
        with pytest.raises(ValidationError) as exc:
            ReflectionConfig(threshold=1.5)
        msg = str(exc.value)
        assert "threshold" in msg

    def test_rejects_threshold_below_0(self):
        with pytest.raises(ValidationError) as exc:
            ReflectionConfig(threshold=-0.1)
        assert "threshold" in str(exc.value)

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 0.85, 1.0])
    def test_accepts_in_range_threshold(self, threshold):
        ReflectionConfig(threshold=threshold)

    def test_rejects_both_system_prompts(self):
        with pytest.raises(ValidationError) as exc:
            ReflectionConfig(system_prompt="x", system_prompt_file="y")
        assert "system_prompt" in str(exc.value).lower()

    def test_accepts_either_system_prompt(self):
        ReflectionConfig(system_prompt="hello")
        ReflectionConfig(system_prompt_file="./p.txt")


# ── AdvisorConfig ─────────────────────────────────────────────────────────


class TestAdvisorConfig:
    def test_defaults(self):
        a = AdvisorConfig()
        assert a.model is None
        assert a.turns == 0
        assert a.parallel is False
        assert a.system_prompt is None
        assert a.system_prompt_file is None

    @pytest.mark.parametrize("turns", [0, 1])
    def test_accepts_in_range_turns(self, turns):
        AdvisorConfig(turns=turns, model="m")

    def test_rejects_turns_above_1(self):
        with pytest.raises(ValidationError) as exc:
            AdvisorConfig(turns=2)
        msg = str(exc.value)
        assert "turns" in msg
        assert "1" in msg

    def test_rejects_negative_turns(self):
        with pytest.raises(ValidationError):
            AdvisorConfig(turns=-1)

    def test_rejects_both_system_prompts(self):
        with pytest.raises(ValidationError):
            AdvisorConfig(system_prompt="x", system_prompt_file="y")


# ── BackendRef ────────────────────────────────────────────────────────────


class TestBackendRef:
    def test_defaults(self):
        b = BackendRef(name="b1")
        assert b.name == "b1"
        assert b.weight == 1

    def test_custom_weight(self):
        b = BackendRef(name="b2", weight=5)
        assert b.weight == 5

    def test_rejects_negative_weight(self):
        with pytest.raises(ValidationError):
            BackendRef(name="b", weight=-1)

    def test_rejects_empty_name(self):
        with pytest.raises(ValidationError):
            BackendRef(name="")


# ── RouteConfig ───────────────────────────────────────────────────────────


class TestRouteConfig:
    def _minimal(self, **overrides):
        data = {
            "name": "r1",
            "match": {"model": "*", "path": "/v1/chat/completions"},
            "backend": "ollama-local",
        }
        data.update(overrides)
        return RouteConfig(**data)

    def test_minimal(self):
        r = self._minimal()
        assert r.name == "r1"
        assert r.strategy == "single"
        assert r.backend == "ollama-local"
        assert r.backends == []
        assert r.aliases == {}
        assert r.fallbacks == []
        assert r.retry == 0
        assert r.reflection.turns == 0
        assert r.advisor.turns == 0

    def test_with_reflection_and_advisor(self):
        r = self._minimal(
            strategy="weighted",
            backends=[BackendRef(name="ollama-local", weight=2)],
            aliases={"coder-pro": "minimax-m3:cloud"},
            fallbacks=["minimax-m2.7:cloud"],
            retry=2,
            reflection=ReflectionConfig(turns=2, threshold=0.9),
            advisor=AdvisorConfig(model="deepseek-v4-pro:cloud", turns=1),
        )
        assert r.strategy == "weighted"
        assert r.backends[0].weight == 2
        assert r.aliases == {"coder-pro": "minimax-m3:cloud"}
        assert r.retry == 2
        assert r.reflection.turns == 2
        assert r.advisor.model == "deepseek-v4-pro:cloud"

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValidationError) as exc:
            self._minimal(strategy="magic")
        assert "strategy" in str(exc.value)

    def test_rejects_retry_above_5(self):
        with pytest.raises(ValidationError) as exc:
            self._minimal(retry=6)
        msg = str(exc.value)
        assert "retry" in msg
        assert "5" in msg

    @pytest.mark.parametrize("retry", [0, 1, 2, 3, 4, 5])
    def test_accepts_in_range_retry(self, retry):
        self._minimal(retry=retry)

    def test_rejects_negative_retry(self):
        with pytest.raises(ValidationError):
            self._minimal(retry=-1)


# ── ModelDefaults & PluginConfig ─────────────────────────────────────────


class TestModelDefaults:
    def test_defaults(self):
        m = ModelDefaults()
        assert m.fallbacks == {}
        assert m.retry == {}

    def test_with_values(self):
        m = ModelDefaults(
            fallbacks={"m1": ["m2"]},
            retry={"m1": 2},
        )
        assert m.fallbacks == {"m1": ["m2"]}
        assert m.retry == {"m1": 2}


class TestPluginConfig:
    def test_defaults(self):
        p = PluginConfig()
        assert p.plugins_dir == "plugins"
        assert p.plugin_config == {}

    def test_custom(self):
        p = PluginConfig(plugins_dir="p", plugin_config={"A": {"x": 1}})
        assert p.plugin_config == {"A": {"x": 1}}


# ── MoaxyConfig ───────────────────────────────────────────────────────────


def _backend(name: str = "ollama-local") -> dict:
    return {
        "name": name,
        "adapter": "ollama",
        "base_url": "http://127.0.0.1:11434",
    }


def _route(name: str = "r1", backend: str | None = "ollama-local") -> dict:
    return {
        "name": name,
        "match": {"model": "*", "path": "/v1/chat/completions"},
        "backend": backend,
    }


class TestMoaxyConfig:
    def test_minimal_with_required_backends(self):
        c = MoaxyConfig(backends=[_backend()])
        assert c.server.listen == "127.0.0.1"
        assert c.server.port == 8765
        assert c.plugins.plugins_dir == "plugins"
        assert c.backends[0].name == "ollama-local"
        assert c.routes == []
        assert c.auth is None
        assert c.models.fallbacks == {}
        assert c.models.retry == {}

    def test_backends_required(self):
        with pytest.raises(ValidationError) as exc:
            MoaxyConfig()
        assert "backends" in str(exc.value)

    def test_routes_default_empty(self):
        c = MoaxyConfig(backends=[_backend()])
        assert c.routes == []

    def test_composes_optional_sections(self):
        c = MoaxyConfig(
            server=ServerConfig(port=9001, log_level="debug"),
            plugins=PluginConfig(plugins_dir="custom_plugins"),
            backends=[_backend()],
            routes=[_route()],
            auth=AuthConfig(enabled=True),
            models=ModelDefaults(fallbacks={"m": ["x"]}),
        )
        assert c.server.port == 9001
        assert c.server.log_level == "debug"
        assert c.plugins.plugins_dir == "custom_plugins"
        assert c.routes[0].name == "r1"
        assert c.auth is not None and c.auth.enabled is True
        assert c.models.fallbacks == {"m": ["x"]}

    def test_rejects_route_referencing_unknown_backend(self):
        with pytest.raises(ValidationError) as exc:
            MoaxyConfig(
                backends=[_backend()],
                routes=[_route(backend="nonexistent")],
            )
        msg = str(exc.value)
        assert "nonexistent" in msg
        assert "backend" in msg.lower()

    def test_rejects_route_backends_ref_unknown_backend(self):
        with pytest.raises(ValidationError) as exc:
            MoaxyConfig(
                backends=[_backend()],
                routes=[
                    {
                        "name": "r1",
                        "match": {"model": "*", "path": "/v1/chat/completions"},
                        "backends": [{"name": "ghost", "weight": 1}],
                    }
                ],
            )
        assert "ghost" in str(exc.value)

    def test_rejects_duplicate_backend_names(self):
        with pytest.raises(ValidationError) as exc:
            MoaxyConfig(backends=[_backend("dup"), _backend("dup")])
        msg = str(exc.value)
        assert "dup" in msg
        assert "backend" in msg.lower()

    def test_rejects_duplicate_route_names(self):
        with pytest.raises(ValidationError) as exc:
            MoaxyConfig(
                backends=[_backend()],
                routes=[_route("dup"), _route("dup")],
            )
        msg = str(exc.value)
        assert "dup" in msg
        assert "route" in msg.lower()

    def test_canonical_example_parses(self):
        c = MoaxyConfig(
            server=ServerConfig(),
            plugins=PluginConfig(),
            backends=[_backend()],
            routes=[
                {
                    "name": "reflective-coder",
                    "match": {"model": "minimax-m3:cloud", "path": "/v1/chat/completions"},
                    "strategy": "single",
                    "backend": "ollama-local",
                    "aliases": {"coder-pro": "minimax-m3:cloud"},
                    "fallbacks": ["minimax-m2.7:cloud", "deepseek-v4-pro:cloud"],
                    "retry": 2,
                    "reflection": {"turns": 1, "early_exit": True, "threshold": 0.85, "parallel": False},
                    "advisor": {"model": "deepseek-v4-pro:cloud", "turns": 1, "parallel": False},
                },
                {
                    "name": "plain",
                    "match": {"model": "*", "path": "/v1/chat/completions"},
                    "strategy": "single",
                    "backend": "ollama-local",
                },
            ],
            auth=AuthConfig(enabled=False),
            models=ModelDefaults(
                fallbacks={"minimax-m3:cloud": ["minimax-m2.7:cloud", "deepseek-v4-pro:cloud"]},
                retry={"minimax-m3:cloud": 2},
            ),
        )
        assert c.routes[0].reflection.turns == 1
        assert c.routes[0].advisor.model == "deepseek-v4-pro:cloud"
        assert c.auth is not None and c.auth.enabled is False
        assert "minimax-m3:cloud" in c.models.fallbacks

    def test_json_round_trip(self):
        c = MoaxyConfig(
            backends=[_backend()],
            routes=[_route()],
        )
        dumped = c.model_dump()
        assert json.dumps(dumped)
        reloaded = json.loads(json.dumps(dumped))
        assert reloaded == dumped

    def test_isinstance_basemodel(self):
        c = MoaxyConfig(backends=[_backend()])
        from pydantic import BaseModel

        assert isinstance(c, BaseModel)
        assert isinstance(c, MoaxyConfig)
        assert c.server.port == 8765


# ── Error message clarity ────────────────────────────────────────────────


class TestErrorMessages:
    def test_server_listen_message_is_clear(self):
        with pytest.raises(ValidationError) as exc:
            ServerConfig(listen="0.0.0.0")
        msg = str(exc.value)
        assert "listen" in msg
        assert "127.0.0.1" in msg or "loopback" in msg.lower()

    def test_reflection_turns_message_is_clear(self):
        with pytest.raises(ValidationError) as exc:
            MoaxyConfig(
                backends=[_backend()],
                routes=[
                    {
                        "name": "r",
                        "match": {"model": "*", "path": "/v1/chat/completions"},
                        "reflection": {"turns": 4},
                    }
                ],
            )
        msg = str(exc.value)
        assert "turns" in msg

    def test_reflection_threshold_message_is_clear(self):
        with pytest.raises(ValidationError) as exc:
            MoaxyConfig(
                backends=[_backend()],
                routes=[
                    {
                        "name": "r",
                        "match": {"model": "*", "path": "/v1/chat/completions"},
                        "reflection": {"threshold": 1.5},
                    }
                ],
            )
        msg = str(exc.value)
        assert "threshold" in msg

    def test_advisor_turns_message_is_clear(self):
        with pytest.raises(ValidationError) as exc:
            MoaxyConfig(
                backends=[_backend()],
                routes=[
                    {
                        "name": "r",
                        "match": {"model": "*", "path": "/v1/chat/completions"},
                        "advisor": {"turns": 2},
                    }
                ],
            )
        msg = str(exc.value)
        assert "turns" in msg

    def test_route_retry_message_is_clear(self):
        with pytest.raises(ValidationError) as exc:
            MoaxyConfig(
                backends=[_backend()],
                routes=[
                    {
                        "name": "r",
                        "match": {"model": "*", "path": "/v1/chat/completions"},
                        "retry": 6,
                    }
                ],
            )
        msg = str(exc.value)
        assert "retry" in msg

    def test_strategy_literal_message_is_clear(self):
        with pytest.raises(ValidationError) as exc:
            MoaxyConfig(
                backends=[_backend()],
                routes=[
                    {
                        "name": "r",
                        "match": {"model": "*", "path": "/v1/chat/completions"},
                        "strategy": "magic",
                    }
                ],
            )
        msg = str(exc.value)
        assert "strategy" in msg

    def test_adapter_literal_message_is_clear(self):
        with pytest.raises(ValidationError) as exc:
            MoaxyConfig(
                backends=[
                    {
                        "name": "x",
                        "adapter": "anthropic",
                        "base_url": "http://x",
                    }
                ],
            )
        msg = str(exc.value)
        assert "adapter" in msg

    def test_backends_required_message_is_clear(self):
        with pytest.raises(ValidationError) as exc:
            MoaxyConfig()
        assert "backends" in str(exc.value)


# ── Aliases: null parses to empty dict ───────────────────────────────────


class TestAliasesNullHandling:
    def test_route_aliases_default_empty(self):
        c = MoaxyConfig(backends=[_backend()], routes=[_route()])
        assert c.routes[0].aliases == {}

    def test_models_section_default(self):
        c = MoaxyConfig(backends=[_backend()])
        assert c.models.fallbacks == {}
        assert c.models.retry == {}


# ── Extra fields are forbidden ───────────────────────────────────────────


class TestStrictSchema:
    def test_server_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            ServerConfig(unknown_field="x")

    def test_adapter_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            AdapterConfig(name="x", adapter="ollama", base_url="http://x", unknown=1)
