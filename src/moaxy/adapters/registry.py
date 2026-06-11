"""AdapterRegistry — constructs backend adapters from a list of AdapterConfig.

The registry is the single place that maps a config-declared
``AdapterConfig.adapter`` literal ("ollama", "openai") to a concrete
:class:`moaxy.adapters.base.Adapter` subclass. The :class:`AdapterRegistry`
exposes a public ``adapters`` dict keyed by the config's ``name`` field
(NOT by the adapter class's class-level ``name`` attribute), and a
``get(name)`` convenience method.

Adapters are constructed with their ``base_url``, ``api_key`` and
``timeout`` values. All other parameters (e.g. ``request_timeout_s``) are
the orchestrator's concern, not the adapter's.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from moaxy.adapters.base import Adapter
from moaxy.adapters.ollama import OllamaAdapter
from moaxy.adapters.openrouter import DEFAULT_BASE_URL as OPENROUTER_DEFAULT_BASE_URL
from moaxy.adapters.openrouter import OpenRouterAdapter
from moaxy.models.config import AdapterConfig

logger = logging.getLogger(__name__)


class UnknownAdapterError(ValueError):
    """Raised when an :class:`AdapterConfig` declares an unimplemented adapter kind."""

    def __init__(self, kind: str, name: str) -> None:
        super().__init__(
            f"unknown adapter kind {kind!r} for backend {name!r}; "
            f"supported kinds: {sorted(_ADAPTER_FACTORIES)}"
        )
        self.kind = kind
        self.name = name


class DuplicateAdapterNameError(ValueError):
    """Raised when two :class:`AdapterConfig` entries share the same ``name``."""

    def __init__(self, name: str) -> None:
        super().__init__(
            f"duplicate backend name {name!r}; backend names must be unique"
        )
        self.name = name


def _build_ollama(config: AdapterConfig) -> Adapter:
    return OllamaAdapter(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
    )


def _build_openrouter(config: AdapterConfig) -> Adapter:
    # ``base_url`` falls back to OpenRouter's canonical default when the
    # config did not set one (i.e. the field is the empty string or
    # whitespace). The ``OpenRouterAdapter`` itself applies the same
    # default, but doing it here keeps the contract explicit at the
    # registry boundary and makes the resolved ``base_url`` observable
    # via the config object passed to the factory.
    base_url = config.base_url.strip() if config.base_url else ""
    if not base_url:
        base_url = OPENROUTER_DEFAULT_BASE_URL
    return OpenRouterAdapter(
        base_url=base_url,
        timeout=config.timeout,
        http_referer=config.http_referer,
        x_title=config.x_title,
        transforms=config.transforms,
    )


# Mapping of config-level adapter kind → factory callable.
# The OpenAI-compatible adapter is not implemented in M1, so the entry
# exists for forward compatibility but raises NotImplementedError when
# actually invoked.
def _build_openai(config: AdapterConfig) -> Adapter:
    raise NotImplementedError(
        "the 'openai' adapter is not yet implemented; "
        "use adapter='ollama' for the M1 milestone"
    )


_ADAPTER_FACTORIES: dict[str, Callable[[AdapterConfig], Adapter]] = {
    "ollama": _build_ollama,
    "openai": _build_openai,
    "openrouter": _build_openrouter,
}


class AdapterRegistry:
    """A name-keyed collection of constructed :class:`Adapter` instances.

    Build the registry once at process startup from the ``backends`` list
    of a parsed :class:`moaxy.models.config.MoaxyConfig` and look up
    adapters by name (matching the ``route.backend`` value) at request
    time. The registry holds no class-level state: two instances are
    fully independent.
    """

    def __init__(self, adapters: dict[str, Adapter] | None = None) -> None:
        self.adapters: dict[str, Adapter] = dict(adapters) if adapters else {}

    @classmethod
    def build(cls, configs: Iterable[AdapterConfig]) -> AdapterRegistry:
        """Construct an :class:`AdapterRegistry` from a sequence of configs.

        Each :class:`AdapterConfig` is mapped to a concrete adapter
        instance via the ``_ADAPTER_FACTORIES`` table. The resulting
        registry's ``adapters`` dict is keyed by the config's ``name``
        field, not by the adapter class.

        Raises:
            DuplicateAdapterNameError: Two configs share the same name.
            UnknownAdapterError: A config declares an adapter kind with
                no registered factory.
            NotImplementedError: A config declares an adapter kind whose
                factory is registered but not yet implemented (e.g.
                ``"openai"`` in M1).
        """
        built: dict[str, Adapter] = {}
        for config in configs:
            if config.name in built:
                raise DuplicateAdapterNameError(config.name)
            factory = _ADAPTER_FACTORIES.get(config.adapter)
            if factory is None:
                raise UnknownAdapterError(config.adapter, config.name)
            adapter = factory(config)
            logger.debug(
                "AdapterRegistry: built adapter %r (kind=%r, base_url=%r)",
                config.name,
                config.adapter,
                config.base_url,
            )
            built[config.name] = adapter
        return cls(built)

    def get(self, name: str) -> Adapter | None:
        """Return the adapter registered under ``name``, or ``None``."""
        return self.adapters.get(name)


def build_registry(configs: Iterable[AdapterConfig]) -> AdapterRegistry:
    """Module-level convenience wrapper for :meth:`AdapterRegistry.build`."""
    return AdapterRegistry.build(configs)


__all__ = [
    "AdapterRegistry",
    "DuplicateAdapterNameError",
    "UnknownAdapterError",
    "build_registry",
]
