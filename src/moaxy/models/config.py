"""Pydantic v2 schema for the moaxy configuration tree.

Parses the YAML/JSON config file into a strictly-typed model hierarchy. All
ranges, literals, and structural invariants are enforced at load time. Routes
default to ``[]``; ``backends`` is required. ``server.listen`` must be a
loopback address — the proxy must not bind to a public interface.
"""

from __future__ import annotations

from ipaddress import ip_address
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

AdapterKind = Literal["ollama", "openai", "openrouter"]
StrategyKind = Literal["single", "weighted", "round_robin"]
LogLevel = Literal["debug", "info", "warning", "error"]


class ServerConfig(BaseModel):
    """HTTP server bind address and runtime settings."""

    model_config = ConfigDict(extra="forbid")

    listen: str = "127.0.0.1"
    port: int = Field(8765, ge=1, le=65535)
    log_level: LogLevel = "info"
    plugins_dir: str = "plugins"
    request_timeout_s: float = Field(60.0, gt=0.0)

    @field_validator("listen")
    @classmethod
    def _listen_must_be_loopback(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "server.listen must be a non-empty string (loopback address, e.g. '127.0.0.1' or '::1')"
            )
        try:
            ip = ip_address(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"server.listen must be a loopback IP address (got {value!r}); "
                "binding to a non-loopback interface is rejected for security"
            ) from exc
        if not ip.is_loopback:
            raise ValueError(
                f"server.listen must be a loopback address such as 127.0.0.1 or ::1 "
                f"(got {value!r}, which is not loopback); binding to 0.0.0.0 is rejected"
            )
        return value


class AdapterConfig(BaseModel):
    """Backend adapter (ollama, openai, or openrouter)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    adapter: AdapterKind
    base_url: str = Field(min_length=1)
    api_key: str | None = None
    timeout: float = Field(30.0, gt=0.0)
    http_referer: str | None = None
    x_title: str | None = None
    transforms: list[str] | None = None

    @field_validator("http_referer")
    @classmethod
    def _http_referer_must_be_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "http_referer must be a non-empty URL string when set "
                "(e.g. 'https://example.com'); use null to omit the header"
            )
        try:
            parsed = urlsplit(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"http_referer is not a valid URL (got {value!r}): {exc}"
            ) from exc
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(
                f"http_referer must be an absolute URL with a scheme and host "
                f"(e.g. 'https://example.com'); got {value!r}"
            )
        return value

    @field_validator("transforms")
    @classmethod
    def _transforms_must_be_non_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list) or len(value) == 0:
            raise ValueError(
                "transforms must be a non-empty list of strings when set; "
                "use null to omit the field"
            )
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    "transforms entries must be non-empty strings "
                    f"(got {entry!r})"
                )
        return value


class ApiKey(BaseModel):
    """An issued API key, its roles, and the scopes it grants."""

    model_config = ConfigDict(extra="forbid")

    key_id: str = Field(min_length=1)
    key_value: str = Field(min_length=1)
    roles: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)


class AuthConfig(BaseModel):
    """Optional API-key authentication gate."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    exempt_paths: list[str] = Field(default_factory=lambda: ["/health"])
    header_names: list[str] = Field(default_factory=lambda: ["X-API-Key", "Authorization"])
    api_keys: list[ApiKey] = Field(default_factory=list)


class RouteMatch(BaseModel):
    """Glob matchers for selecting a route by request model and path."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    path: str = Field(min_length=1)


class ReflectionConfig(BaseModel):
    """Self-reflection loop configuration: 0..3 turns with optional early exit."""

    model_config = ConfigDict(extra="forbid")

    turns: int = Field(0, ge=0, le=3)
    early_exit: bool = True
    threshold: float = Field(0.85, ge=0.0, le=1.0)
    parallel: bool = False
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    order: Literal["reflect_first", "advise_first"] = "reflect_first"
    trust_verbal: float = Field(0.6, ge=0.0)
    trust_score: float = Field(0.4, ge=0.0)
    fresh_context: bool = False

    @model_validator(mode="after")
    def _system_prompt_xor(self) -> ReflectionConfig:
        if self.system_prompt is not None and self.system_prompt_file is not None:
            raise ValueError(
                "reflection.system_prompt and reflection.system_prompt_file "
                "are mutually exclusive; set at most one"
            )
        return self


class AdvisorConfig(BaseModel):
    """Advisor stage configuration: optional second model that approves or revises."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    turns: int = Field(0, ge=0, le=1)
    parallel: bool = False
    system_prompt: str | None = None
    system_prompt_file: str | None = None

    @model_validator(mode="after")
    def _system_prompt_xor(self) -> AdvisorConfig:
        if self.system_prompt is not None and self.system_prompt_file is not None:
            raise ValueError(
                "advisor.system_prompt and advisor.system_prompt_file "
                "are mutually exclusive; set at most one"
            )
        return self


class BackendRef(BaseModel):
    """A weighted reference to a configured backend (used by multi-backend routes)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    weight: int = Field(1, ge=0)


class RouteConfig(BaseModel):
    """A single routing rule: matcher, backend selection, reflection, advisor."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    match: RouteMatch
    strategy: StrategyKind = "single"
    backend: str | None = None
    backends: list[BackendRef] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    fallbacks: list[str] = Field(default_factory=list)
    retry: int = Field(0, ge=0, le=5)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    advisor: AdvisorConfig = Field(default_factory=AdvisorConfig)

    @field_validator("aliases", mode="before")
    @classmethod
    def _aliases_none_to_empty(cls, value: Any) -> Any:
        # An explicit YAML ``aliases: null`` is equivalent to omitting the
        # field. Pydantic would otherwise reject ``None`` for a non-Optional
        # ``dict`` type.
        return {} if value is None else value


class ModelDefaults(BaseModel):
    """Per-model fallback and retry defaults shared across routes."""

    model_config = ConfigDict(extra="forbid")

    fallbacks: dict[str, list[str]] = Field(default_factory=dict)
    retry: dict[str, int] = Field(default_factory=dict)


class PluginConfig(BaseModel):
    """Plugin discovery directory and per-plugin config overrides."""

    model_config = ConfigDict(extra="forbid")

    plugins_dir: str = "plugins"
    plugin_config: dict[str, dict[str, Any]] = Field(default_factory=dict)


class MoaxyConfig(BaseModel):
    """Top-level configuration tree."""

    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    backends: list[AdapterConfig]
    routes: list[RouteConfig] = Field(default_factory=list)
    auth: AuthConfig | None = None
    models: ModelDefaults = Field(default_factory=ModelDefaults)

    @field_validator("backends")
    @classmethod
    def _backends_must_have_unique_names(
        cls, value: list[AdapterConfig]
    ) -> list[AdapterConfig]:
        seen: set[str] = set()
        for backend in value:
            if backend.name in seen:
                raise ValueError(
                    f"duplicate backend name {backend.name!r} in moaxy.backends; "
                    "backend names must be unique"
                )
            seen.add(backend.name)
        return value

    @field_validator("routes")
    @classmethod
    def _routes_must_have_unique_names(
        cls, value: list[RouteConfig]
    ) -> list[RouteConfig]:
        seen: set[str] = set()
        for route in value:
            if route.name in seen:
                raise ValueError(
                    f"duplicate route name {route.name!r} in moaxy.routes; "
                    "route names must be unique"
                )
            seen.add(route.name)
        return value

    @model_validator(mode="after")
    def _routes_reference_known_backends(self) -> MoaxyConfig:
        backend_names = {b.name for b in self.backends}
        for route in self.routes:
            if route.backend is not None and route.backend not in backend_names:
                raise ValueError(
                    f"route {route.name!r} references unknown backend {route.backend!r}; "
                    f"known backends: {sorted(backend_names)}"
                )
            for ref in route.backends:
                if ref.name not in backend_names:
                    raise ValueError(
                        f"route {route.name!r} backends[] references unknown backend {ref.name!r}; "
                        f"known backends: {sorted(backend_names)}"
                    )
        return self


__all__ = [
    "AdapterConfig",
    "AdapterKind",
    "AdvisorConfig",
    "ApiKey",
    "AuthConfig",
    "BackendRef",
    "LogLevel",
    "ModelDefaults",
    "MoaxyConfig",
    "PluginConfig",
    "ReflectionConfig",
    "RouteConfig",
    "RouteMatch",
    "ServerConfig",
    "StrategyKind",
    "ValidationError",
]
