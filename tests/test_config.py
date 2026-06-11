"""Tests for the moaxy configuration system (Pydantic v2 schema + loader).

Covers:

* Pydantic schema (all 12 models): importability, default values, range and
  literal rejections, required-field invariants, loopback-only listen, port
  range, JSON serialisation, and the canonical example parse.
* Config loader: path discovery in priority order
  (``MOAXY_CONFIG_PATH`` -> ``config.yaml`` -> ``config.yml`` -> ``config.json``
  -> defaults), ``${ENV_VAR}`` substitution (basic, multi, missing, empty,
  nested), parse errors (YAML/JSON), default fallback, and a typed
  :class:`MoaxyConfig` return.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from moaxy.config import (
    ConfigNotFoundError,
    SubstitutionError,
    find_config_file,
    load_config,
    parse_config_payload,
    substitute_env,
)
from moaxy.config.loader import CONFIG_PATH_ENV_VAR, DEFAULT_CANDIDATE_NAMES
from moaxy.models.config import (
    AdapterConfig,
    AdvisorConfig,
    ApiKey,
    AuthConfig,
    BackendRef,
    MoaxyConfig,
    ModelDefaults,
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
            MoaxyConfig,
            ModelDefaults,
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
        assert r.order == "reflect_first"
        assert r.trust_verbal == 0.6
        assert r.trust_score == 0.4

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


# ── ReflectionConfig M5 delta fields ──────────────────────────────────────


class TestReflectionConfigM5DeltaFields:
    """M5 delta 1: order, trust_verbal, trust_score on ReflectionConfig.

    These fields are backward-compatible: their defaults preserve the
    v1-v4 behavior. The defaults come from the M5 alignment document
    (order='reflect_first', trust_verbal=0.6, trust_score=0.4). See
    `m5-delta-config-fields` in features.json and the
    `m5-delta-pydantic-config-fields` section of architecture.md.
    """

    def test_default_order_is_reflect_first(self):
        # VAL-PIPE-EXTRA-027 invariant: default preserves v1 ordering.
        r = ReflectionConfig()
        assert r.order == "reflect_first"

    def test_default_trust_verbal_is_zero_point_six(self):
        # alignment document default for the verbal-confidence weight.
        r = ReflectionConfig()
        assert r.trust_verbal == 0.6

    def test_default_trust_score_is_zero_point_four(self):
        # alignment document default for the score-weight in the
        # weighted early-exit signal.
        r = ReflectionConfig()
        assert r.trust_score == 0.4

    def test_explicit_order_reflect_first_parses(self):
        r = ReflectionConfig(order="reflect_first")
        assert r.order == "reflect_first"

    def test_explicit_order_advise_first_parses(self):
        r = ReflectionConfig(order="advise_first")
        assert r.order == "advise_first"

    @pytest.mark.parametrize(
        "bad_value",
        [
            "reflect-first",  # hyphens not allowed
            "advise",  # missing suffix
            "REJECT",
            "first",
            "advise_first ",  # trailing whitespace
            " reflect_first",  # leading whitespace
        ],
    )
    def test_rejects_invalid_order_literal(self, bad_value):
        with pytest.raises(ValidationError) as exc:
            ReflectionConfig(order=bad_value)
        assert "order" in str(exc.value)

    def test_rejects_non_string_order(self):
        with pytest.raises(ValidationError) as exc:
            ReflectionConfig(order=42)  # type: ignore[arg-type]
        assert "order" in str(exc.value)

    def test_rejects_none_order(self):
        with pytest.raises(ValidationError) as exc:
            ReflectionConfig(order=None)  # type: ignore[arg-type]
        assert "order" in str(exc.value)

    def test_explicit_trust_verbal_parses(self):
        r = ReflectionConfig(trust_verbal=0.5)
        assert r.trust_verbal == 0.5

    def test_explicit_trust_score_parses(self):
        r = ReflectionConfig(trust_score=0.0)
        assert r.trust_score == 0.0

    @pytest.mark.parametrize("value", [0.0, 0.3, 0.6, 0.4, 1.0, 2.5, 100.0])
    def test_accepts_non_negative_trust_verbal(self, value):
        # No upper bound per the alignment document; weights can be
        # arbitrarily large when the other weight is zero.
        r = ReflectionConfig(trust_verbal=value)
        assert r.trust_verbal == value

    @pytest.mark.parametrize("value", [0.0, 0.3, 0.6, 0.4, 1.0, 2.5, 100.0])
    def test_accepts_non_negative_trust_score(self, value):
        r = ReflectionConfig(trust_score=value)
        assert r.trust_score == value

    def test_rejects_negative_trust_verbal(self):
        with pytest.raises(ValidationError) as exc:
            ReflectionConfig(trust_verbal=-0.1)
        assert "trust_verbal" in str(exc.value)

    def test_rejects_negative_trust_score(self):
        with pytest.raises(ValidationError) as exc:
            ReflectionConfig(trust_score=-0.01)
        assert "trust_score" in str(exc.value)

    def test_rejects_very_negative_trust_verbal(self):
        with pytest.raises(ValidationError):
            ReflectionConfig(trust_verbal=-100.0)

    def test_rejects_very_negative_trust_score(self):
        with pytest.raises(ValidationError):
            ReflectionConfig(trust_score=-50.5)

    def test_rejects_non_numeric_trust_verbal(self):
        with pytest.raises(ValidationError):
            ReflectionConfig(trust_verbal="high")  # type: ignore[arg-type]

    def test_rejects_non_numeric_trust_score(self):
        with pytest.raises(ValidationError):
            ReflectionConfig(trust_score="low")  # type: ignore[arg-type]

    def test_all_three_fields_round_trip(self):
        # Combined: order, trust_verbal, and trust_score all set at once.
        r = ReflectionConfig(
            order="advise_first",
            trust_verbal=0.25,
            trust_score=0.75,
        )
        assert r.order == "advise_first"
        assert r.trust_verbal == 0.25
        assert r.trust_score == 0.75

    def test_yaml_load_with_new_fields(self, tmp_path, monkeypatch):
        # End-to-end: a YAML config carrying the new fields parses into
        # a typed ReflectionConfig.
        monkeypatch.delenv("MOAXY_CONFIG_PATH", raising=False)
        path = tmp_path / "cfg.yaml"
        path.write_text(
            """
backends:
  - name: b
    adapter: ollama
    base_url: http://127.0.0.1:11434
routes:
  - name: r
    match: {model: "*", path: "/v1/chat/completions"}
    reflection:
      turns: 1
      order: advise_first
      trust_verbal: 0.7
      trust_score: 0.3
""",
            encoding="utf-8",
        )
        from moaxy.config import load_config

        cfg = load_config(path=path)
        refl = cfg.routes[0].reflection
        assert refl.order == "advise_first"
        assert refl.trust_verbal == 0.7
        assert refl.trust_score == 0.3

    def test_yaml_load_with_invalid_order_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MOAXY_CONFIG_PATH", raising=False)
        path = tmp_path / "cfg.yaml"
        path.write_text(
            """
backends:
  - name: b
    adapter: ollama
    base_url: http://127.0.0.1:11434
routes:
  - name: r
    match: {model: "*", path: "/v1/chat/completions"}
    reflection:
      turns: 1
      order: reflect-first
""",
            encoding="utf-8",
        )
        from moaxy.config import load_config

        with pytest.raises(ValidationError) as exc:
            load_config(path=path)
        assert "order" in str(exc.value)

    def test_yaml_load_with_negative_trust_score_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MOAXY_CONFIG_PATH", raising=False)
        path = tmp_path / "cfg.yaml"
        path.write_text(
            """
backends:
  - name: b
    adapter: ollama
    base_url: http://127.0.0.1:11434
routes:
  - name: r
    match: {model: "*", path: "/v1/chat/completions"}
    reflection:
      turns: 1
      trust_score: -0.1
""",
            encoding="utf-8",
        )
        from moaxy.config import load_config

        with pytest.raises(ValidationError) as exc:
            load_config(path=path)
        assert "trust_score" in str(exc.value)


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


# ── envsubst: SubstitutionError ──────────────────────────────────────────


class TestSubstitutionError:
    def test_message_mentions_variable_name(self):
        with pytest.raises(SubstitutionError) as exc:
            substitute_env("hello ${MISSING_VAR_XYZ}", env={})
        assert "MISSING_VAR_XYZ" in str(exc.value)
        assert exc.value.name == "MISSING_VAR_XYZ"

    def test_message_mentions_environment_or_substitution(self):
        with pytest.raises(SubstitutionError) as exc:
            substitute_env("${OPENAI_API_KEY}", env={})
        msg = str(exc.value).lower()
        assert "substitution" in msg or "environment variable" in msg

    def test_subclass_of_keyerror(self):
        assert issubclass(SubstitutionError, KeyError)

    def test_can_be_caught_as_keyerror(self):
        with pytest.raises(KeyError):
            substitute_env("${X}", env={})

    def test_nested_dict_missing_var_propagates(self):
        with pytest.raises(SubstitutionError) as exc:
            substitute_env({"a": {"b": "${NOPE}"}}, env={"OTHER": "1"})
        assert "NOPE" in str(exc.value)

    def test_list_missing_var_propagates(self):
        with pytest.raises(SubstitutionError) as exc:
            substitute_env(["ok", "${NOPE}"], env={})
        assert "NOPE" in str(exc.value)


# ── envsubst: basic substitution ─────────────────────────────────────────


class TestSubstitutionBasic:
    def test_single_token(self):
        assert substitute_env("hi ${NAME}", env={"NAME": "world"}) == "hi world"

    def test_token_at_start(self):
        assert substitute_env("${GREETING}!", env={"GREETING": "hi"}) == "hi!"

    def test_token_at_end(self):
        assert substitute_env("say ${WORD}", env={"WORD": "please"}) == "say please"

    def test_whole_string_is_token(self):
        assert substitute_env("${WHOLE}", env={"WHOLE": "all of it"}) == "all of it"

    def test_no_tokens_returns_same_string(self):
        assert substitute_env("plain text", env={}) == "plain text"

    def test_empty_string_returns_empty_string(self):
        assert substitute_env("", env={}) == ""

    def test_preserves_literal_dollar_sign(self):
        # A bare $ not followed by { is treated as a literal character.
        assert substitute_env("price: $5", env={}) == "price: $5"

    def test_preserves_partial_brace(self):
        # ${ alone (no closing brace) is not a token; also $X (no braces).
        assert substitute_env("${unterminated", env={}) == "${unterminated"

    def test_underscore_in_name(self):
        assert substitute_env(
            "${MY_VAR_1}", env={"MY_VAR_1": "ok"}
        ) == "ok"

    def test_digits_in_name(self):
        assert (
            substitute_env("${VAR123}", env={"VAR123": "ok"}) == "ok"
        )

    def test_name_must_start_with_letter_or_underscore(self):
        # $1X is not a valid name; the regex should not match.
        assert substitute_env("${1INVALID}", env={}) == "${1INVALID}"


# ── envsubst: multiple tokens ────────────────────────────────────────────


class TestSubstitutionMultiple:
    def test_two_tokens(self):
        result = substitute_env("${A}-${B}", env={"A": "foo", "B": "bar"})
        assert result == "foo-bar"

    def test_three_tokens(self):
        result = substitute_env(
            "${A}/${B}/${C}",
            env={"A": "x", "B": "y", "C": "z"},
        )
        assert result == "x/y/z"

    def test_token_used_twice(self):
        result = substitute_env(
            "${A}-${A}-${A}",
            env={"A": "x"},
        )
        assert result == "x-x-x"

    def test_unrelated_text_around_tokens(self):
        result = substitute_env(
            "start ${A} middle ${B} end",
            env={"A": "AAA", "B": "BBB"},
        )
        assert result == "start AAA middle BBB end"

    def test_one_missing_among_many_raises(self):
        with pytest.raises(SubstitutionError) as exc:
            substitute_env(
                "${A}-${B}-${C}",
                env={"A": "x", "C": "z"},
            )
        assert "B" in str(exc.value)


# ── envsubst: nested structures ──────────────────────────────────────────


class TestSubstitutionStructures:
    def test_dict_at_root(self):
        out = substitute_env(
            {"api_key": "${KEY}", "name": "x"},
            env={"KEY": "secret"},
        )
        assert out == {"api_key": "secret", "name": "x"}

    def test_list_at_root(self):
        out = substitute_env(
            ["${A}", "${B}", "static"],
            env={"A": "1", "B": "2"},
        )
        assert out == ["1", "2", "static"]

    def test_deeply_nested(self):
        out = substitute_env(
            {
                "backends": [
                    {"name": "${NAME}", "base_url": "http://${HOST}:${PORT}"},
                ],
                "server": {"listen": "${LISTEN}"},
            },
            env={
                "NAME": "ollama-local",
                "HOST": "127.0.0.1",
                "PORT": "11434",
                "LISTEN": "127.0.0.1",
            },
        )
        assert out["backends"][0]["name"] == "ollama-local"
        assert out["backends"][0]["base_url"] == "http://127.0.0.1:11434"
        assert out["server"]["listen"] == "127.0.0.1"

    def test_non_string_passthrough(self):
        out = substitute_env(
            {"port": 8765, "enabled": True, "tags": None, "rate": 0.5},
            env={},
        )
        assert out == {"port": 8765, "enabled": True, "tags": None, "rate": 0.5}

    def test_uses_os_environ_by_default(self, monkeypatch):
        monkeypatch.setenv("MOAXY_TEST_VAR", "from-env")
        assert substitute_env("${MOAXY_TEST_VAR}") == "from-env"

    def test_does_not_mutate_input(self):
        original = {"k": "${A}"}
        substitute_env(original, env={"A": "v"})
        assert original == {"k": "${A}"}


# ── envsubst: empty value handling ───────────────────────────────────────


class TestSubstitutionEmptyValue:
    def test_empty_string_value_raises(self):
        # An env var set to "" must fail loud. This is the consistent behavior
        # the contract allows: either fail-loud (chosen) or treat as empty.
        with pytest.raises(SubstitutionError) as exc:
            substitute_env("${EMPTY_VAR}", env={"EMPTY_VAR": ""})
        assert exc.value.name == "EMPTY_VAR"

    def test_empty_value_message_mentions_name(self):
        with pytest.raises(SubstitutionError) as exc:
            substitute_env("prefix ${EMPTY_VAR} suffix", env={"EMPTY_VAR": ""})
        assert "EMPTY_VAR" in str(exc.value)


# ── loader: discovery ────────────────────────────────────────────────────


class TestFindConfigFile:
    def test_explicit_path_takes_precedence(self, tmp_path, monkeypatch):
        # Even with MOAXY_CONFIG_PATH set, an explicit path wins.
        target = tmp_path / "explicit.yaml"
        target.write_text("server: {port: 9000}\n")
        monkeypatch.setenv(CONFIG_PATH_ENV_VAR, str(tmp_path / "env.yaml"))
        result = find_config_file(path=target, cwd=tmp_path)
        assert result == target

    def test_env_var_path_used_when_set_and_exists(self, tmp_path, monkeypatch):
        env_path = tmp_path / "envpath.yaml"
        env_path.write_text("server: {port: 9000}\n")
        monkeypatch.setenv(CONFIG_PATH_ENV_VAR, str(env_path))
        result = find_config_file(cwd=tmp_path)
        assert result == env_path

    def test_env_var_path_must_exist(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            CONFIG_PATH_ENV_VAR, str(tmp_path / "missing.yaml")
        )
        with pytest.raises(ConfigNotFoundError) as exc:
            find_config_file(cwd=tmp_path)
        assert "MOAXY_CONFIG_PATH" in str(exc.value)

    def test_yaml_preferred_over_yml(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        (tmp_path / "config.yml").write_text("server: {port: 9001}\n")
        (tmp_path / "config.yaml").write_text("server: {port: 9002}\n")
        result = find_config_file(cwd=tmp_path)
        assert result is not None
        assert result.name == "config.yaml"

    def test_yml_used_when_yaml_absent(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        (tmp_path / "config.yml").write_text("server: {port: 9003}\n")
        result = find_config_file(cwd=tmp_path)
        assert result is not None
        assert result.name == "config.yml"

    def test_json_used_when_yaml_and_yml_absent(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        (tmp_path / "config.json").write_text('{"server": {"port": 9004}}\n')
        result = find_config_file(cwd=tmp_path)
        assert result is not None
        assert result.name == "config.json"

    def test_yaml_preferred_over_json(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        (tmp_path / "config.json").write_text('{"server": {"port": 9005}}\n')
        (tmp_path / "config.yaml").write_text("server: {port: 9006}\n")
        result = find_config_file(cwd=tmp_path)
        assert result is not None
        assert result.name == "config.yaml"

    def test_yml_preferred_over_json(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        (tmp_path / "config.json").write_text('{"server": {"port": 9005}}\n')
        (tmp_path / "config.yml").write_text("server: {port: 9006}\n")
        result = find_config_file(cwd=tmp_path)
        assert result is not None
        assert result.name == "config.yml"

    def test_returns_none_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        assert find_config_file(cwd=tmp_path) is None

    def test_default_candidates_include_all_three(self):
        assert "config.yaml" in DEFAULT_CANDIDATE_NAMES
        assert "config.yml" in DEFAULT_CANDIDATE_NAMES
        assert "config.json" in DEFAULT_CANDIDATE_NAMES


# ── loader: load_config happy paths ──────────────────────────────────────


def _write_yaml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _write_json(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


class TestLoadConfigDefaults:
    def test_no_file_returns_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        cfg = load_config(cwd=tmp_path)
        assert isinstance(cfg, MoaxyConfig)
        assert cfg.backends == []
        assert cfg.routes == []

    def test_no_file_default_uses_server_port_8765(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        cfg = load_config(cwd=tmp_path)
        assert cfg.server.port == 8765


class TestLoadConfigExplicitPath:
    def test_explicit_yaml(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
server:
  port: 9100
backends:
  - name: b1
    adapter: ollama
    base_url: http://127.0.0.1:11434
""",
        )
        cfg = load_config(path=path)
        assert cfg.server.port == 9100
        assert cfg.backends[0].name == "b1"

    def test_explicit_json(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        path = _write_json(
            tmp_path / "cfg.json",
            json.dumps(
                {
                    "server": {"port": 9200},
                    "backends": [
                        {
                            "name": "b1",
                            "adapter": "ollama",
                            "base_url": "http://127.0.0.1:11434",
                        }
                    ],
                }
            ),
        )
        cfg = load_config(path=path)
        assert cfg.server.port == 9200
        assert cfg.backends[0].name == "b1"


# ── loader: discovery from cwd ──────────────────────────────────────────


class TestLoadConfigFromCwd:
    def test_config_yaml_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        _write_yaml(
            tmp_path / "config.yaml",
            """
server:
  port: 9300
backends:
  - name: b
    adapter: ollama
    base_url: http://127.0.0.1:11434
""",
        )
        cfg = load_config(cwd=tmp_path)
        assert cfg.server.port == 9300

    def test_config_yml_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        _write_yaml(
            tmp_path / "config.yml",
            """
server:
  port: 9400
backends:
  - name: b
    adapter: ollama
    base_url: http://127.0.0.1:11434
""",
        )
        cfg = load_config(cwd=tmp_path)
        assert cfg.server.port == 9400

    def test_config_json_in_cwd(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        _write_json(
            tmp_path / "config.json",
            json.dumps(
                {
                    "server": {"port": 9500},
                    "backends": [
                        {
                            "name": "b",
                            "adapter": "ollama",
                            "base_url": "http://127.0.0.1:11434",
                        }
                    ],
                }
            ),
        )
        cfg = load_config(cwd=tmp_path)
        assert cfg.server.port == 9500


# ── loader: MOAXY_CONFIG_PATH wins ─────────────────────────────────────


class TestMoaxyConfigPathEnvVar:
    def test_env_var_wins_over_config_yaml(self, tmp_path, monkeypatch):
        # Both files exist; the env var must win.
        _write_yaml(
            tmp_path / "config.yaml",
            """
server:
  port: 1001
backends:
  - name: from_default
    adapter: ollama
    base_url: http://127.0.0.1:11434
""",
        )
        env_path = _write_yaml(
            tmp_path / "env_config.yaml",
            """
server:
  port: 2002
backends:
  - name: from_env
    adapter: ollama
    base_url: http://127.0.0.1:11434
""",
        )
        monkeypatch.setenv(CONFIG_PATH_ENV_VAR, str(env_path))
        cfg = load_config(cwd=tmp_path)
        assert cfg.server.port == 2002
        assert cfg.backends[0].name == "from_env"


# ── loader: envsubst integration ────────────────────────────────────────


class TestLoadConfigEnvSubstitution:
    def test_api_key_quoted_token(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-abc-123")
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: openai
    adapter: openai
    base_url: https://api.openai.com/v1
    api_key: "${OPENAI_API_KEY}"
""",
        )
        cfg = load_config(path=path)
        assert cfg.backends[0].api_key == "sk-abc-123"

    def test_base_url_unquoted_token(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: ollama-local
    adapter: ollama
    base_url: http://${OLLAMA_HOST}/v1
""",
        )
        cfg = load_config(path=path)
        assert cfg.backends[0].base_url == "http://127.0.0.1:11434/v1"

    def test_multiple_tokens_in_one_string(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        monkeypatch.setenv("A", "foo")
        monkeypatch.setenv("B", "bar")
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: combined
    adapter: openai
    base_url: https://x
    api_key: "${A}-${B}"
""",
        )
        cfg = load_config(path=path)
        assert cfg.backends[0].api_key == "foo-bar"

    def test_missing_env_var_raises_substitution_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        monkeypatch.delenv("MISSING_VAR_XYZ", raising=False)
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: openai
    adapter: openai
    base_url: https://x
    api_key: "${MISSING_VAR_XYZ}"
""",
        )
        with pytest.raises(SubstitutionError) as exc:
            load_config(path=path)
        assert "MISSING_VAR_XYZ" in str(exc.value)

    def test_empty_env_var_handled_consistently(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        monkeypatch.setenv("EMPTY_VAR", "")
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: openai
    adapter: openai
    base_url: https://x
    api_key: "${EMPTY_VAR}"
""",
        )
        # Either fail-loud OR substitute "" is acceptable. Our implementation
        # fails loud.
        with pytest.raises(SubstitutionError) as exc:
            load_config(path=path)
        assert exc.value.name == "EMPTY_VAR"

    def test_openai_api_key_cross_area(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-abc-123")
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: openai
    adapter: openai
    base_url: https://api.openai.com/v1
    api_key: "${OPENAI_API_KEY}"
""",
        )
        cfg = load_config(path=path)
        assert cfg.backends[0].api_key == "sk-test-abc-123"

    def test_missing_openai_api_key_fails_loud(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: openai
    adapter: openai
    base_url: https://api.openai.com/v1
    api_key: "${OPENAI_API_KEY}"
""",
        )
        with pytest.raises(SubstitutionError) as exc:
            load_config(path=path)
        assert "OPENAI_API_KEY" in str(exc.value)


# ── loader: YAML and JSON parse errors ──────────────────────────────────


class TestLoadConfigParseErrors:
    def test_malformed_yaml_raises_yamlerror(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        path = _write_yaml(tmp_path / "bad.yaml", "not: valid: yaml: :: :\n")
        with pytest.raises(yaml.YAMLError):
            load_config(path=path)

    def test_malformed_json_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        path = _write_json(tmp_path / "bad.json", "{not-json")
        with pytest.raises(json.JSONDecodeError):
            load_config(path=path)

    def test_top_level_not_mapping_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        path = _write_yaml(tmp_path / "list.yaml", "- one\n- two\n")
        with pytest.raises(yaml.YAMLError):
            load_config(path=path)


# ── loader: schema validation errors propagate ──────────────────────────


class TestLoadConfigSchemaErrors:
    def test_missing_backends_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
server:
  port: 8765
""",
        )
        with pytest.raises(ValidationError) as exc:
            load_config(path=path)
        assert "backends" in str(exc.value)

    def test_reflection_turns_out_of_range(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: b
    adapter: ollama
    base_url: http://127.0.0.1:11434
routes:
  - name: r
    match: {model: "*", path: "/v1/chat/completions"}
    reflection:
      turns: 4
""",
        )
        with pytest.raises(ValidationError) as exc:
            load_config(path=path)
        assert "turns" in str(exc.value)


# ── loader: canonical example ──────────────────────────────────────────


class TestLoadConfigCanonicalExample:
    def test_canonical_example_parses(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
server:
  listen: "127.0.0.1"
  port: 8765
  log_level: "info"
  plugins_dir: "plugins"
  request_timeout_s: 60.0

plugins:
  plugins_dir: "plugins"
  plugin_config: {}

backends:
  - name: "ollama-local"
    adapter: "ollama"
    base_url: "http://127.0.0.1:11434"
    timeout: 30.0

models:
  fallbacks:
    "minimax-m3:cloud": ["minimax-m2.7:cloud", "deepseek-v4-pro:cloud"]
  retry:
    "minimax-m3:cloud": 2

routes:
  - name: "reflective-coder"
    match:
      model: "minimax-m3:cloud"
      path: "/v1/chat/completions"
    strategy: "single"
    backend: "ollama-local"
    aliases:
      "coder-pro": "minimax-m3:cloud"
    fallbacks: ["minimax-m2.7:cloud", "deepseek-v4-pro:cloud"]
    retry: 2
    reflection:
      turns: 1
      early_exit: true
      threshold: 0.85
      parallel: false
    advisor:
      model: "deepseek-v4-pro:cloud"
      turns: 1
      parallel: false

  - name: "plain"
    match: { model: "*", path: "/v1/chat/completions" }
    strategy: "single"
    backend: "ollama-local"

auth:
  enabled: false
""",
        )
        cfg = load_config(path=path)
        assert isinstance(cfg, MoaxyConfig)
        assert cfg.routes[0].reflection.turns == 1
        assert cfg.routes[0].advisor.model == "deepseek-v4-pro:cloud"
        assert cfg.auth is not None
        assert cfg.auth.enabled is False
        assert "minimax-m3:cloud" in cfg.models.fallbacks


# ── loader: aliases: null parses to empty dict ──────────────────────────


class TestLoadConfigAliasesNull:
    def test_route_aliases_null_yields_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: b
    adapter: ollama
    base_url: http://127.0.0.1:11434
routes:
  - name: r
    match: {model: "*", path: "/v1/chat/completions"}
    aliases: null
""",
        )
        cfg = load_config(path=path)
        assert cfg.routes[0].aliases == {}


# ── loader: model_dump() is JSON-serializable ───────────────────────────


class TestLoadConfigJsonSerializable:
    def test_default_is_json_serializable(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        cfg = load_config(cwd=tmp_path)
        dumped = cfg.model_dump()
        encoded = json.dumps(dumped)
        assert json.loads(encoded) == dumped

    def test_loaded_config_is_json_serializable(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
server:
  port: 9001
backends:
  - name: b
    adapter: ollama
    base_url: http://127.0.0.1:11434
routes:
  - name: r
    match: {model: "*", path: "/v1/chat/completions"}
""",
        )
        cfg = load_config(path=path)
        dumped = cfg.model_dump()
        encoded = json.dumps(dumped)
        reloaded = json.loads(encoded)
        assert reloaded == dumped
        assert reloaded["server"]["port"] == 9001


# ── loader: parse_config_payload direct API ─────────────────────────────


class TestParseConfigPayload:
    def test_payload_only_no_file(self):
        cfg = parse_config_payload(
            {"backends": [{"name": "b", "adapter": "ollama", "base_url": "http://x"}]},
            env={},
        )
        assert isinstance(cfg, MoaxyConfig)
        assert cfg.backends[0].name == "b"

    def test_payload_with_env_substitution(self):
        cfg = parse_config_payload(
            {"backends": [{"name": "b", "adapter": "ollama", "base_url": "${HOST}"}]},
            env={"HOST": "http://1.2.3.4:5678"},
        )
        assert cfg.backends[0].base_url == "http://1.2.3.4:5678"


# ── loader: typed return ────────────────────────────────────────────────


class TestLoaderReturnsTyped:
    def test_returns_moaxy_config_instance(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: b
    adapter: ollama
    base_url: http://127.0.0.1:11434
""",
        )
        cfg = load_config(path=path)
        assert isinstance(cfg, MoaxyConfig)

    def test_attribute_access_works(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
server:
  port: 9100
backends:
  - name: b
    adapter: ollama
    base_url: http://127.0.0.1:11434
""",
        )
        cfg = load_config(path=path)
        assert cfg.server.port == 9100
        assert cfg.backends[0].name == "b"


# ── plugin_config default is empty dict ────────────────────────────────


class TestPluginConfigDefault:
    def test_plugins_section_default_is_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        cfg = load_config(cwd=tmp_path)
        assert cfg.plugins.plugins_dir == "plugins"
        assert cfg.plugins.plugin_config == {}

    def test_plugins_section_explicit_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.delenv(CONFIG_PATH_ENV_VAR, raising=False)
        path = _write_yaml(
            tmp_path / "cfg.yaml",
            """
backends:
  - name: b
    adapter: ollama
    base_url: http://127.0.0.1:11434
plugins:
  plugins_dir: "plugins"
""",
        )
        cfg = load_config(path=path)
        assert cfg.plugins.plugins_dir == "plugins"
        assert cfg.plugins.plugin_config == {}
