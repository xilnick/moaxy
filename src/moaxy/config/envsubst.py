"""Recursive ${ENV_VAR} substitution for arbitrary Python data structures.

Strings are scanned for ``${NAME}`` tokens; every match is replaced with the
value of the corresponding environment variable. Dicts and lists are walked
recursively. Other value types (``int``, ``float``, ``bool``, ``None``) are
returned unchanged.

A missing or unset environment variable raises :class:`SubstitutionError` with
the variable name in the message. The contract treats empty values as
unacceptable: an empty value is treated as "set to empty string" but the loader
also flags the situation. To keep behavior consistent and safe, an empty
environment variable raises :class:`SubstitutionError` — an empty string in
the config is almost always a misconfiguration (e.g. an empty ``api_key``).
Callers that need a different policy can wrap the result in their own logic.
"""

from __future__ import annotations

import os
import re
from typing import Any, Mapping

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class SubstitutionError(KeyError):
    """Raised when a ``${ENV_VAR}`` token cannot be resolved.

    Inherits from :class:`KeyError` for backwards-compatible ``except KeyError``
    handlers. The ``name`` attribute carries the variable name; ``args[0]``
    is the human-readable message.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        message = (
            f"environment variable substitution failed: ${name} is not set "
            "(set it in the environment or fix the config file)"
        )
        super().__init__(message)


def _substitute_string(text: str, env: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in env:
            raise SubstitutionError(name)
        value = env[name]
        if value == "":
            raise SubstitutionError(name)
        return value

    return _ENV_PATTERN.sub(replace, text)


def substitute_env(value: Any, env: Mapping[str, str] | None = None) -> Any:
    """Return ``value`` with every ``${ENV_VAR}`` token substituted.

    Args:
        value: Arbitrary Python object (dict, list, str, primitive).
        env: Mapping to read environment variables from. Defaults to
            :data:`os.environ`. Pass an explicit mapping in tests.

    Returns:
        A new structure of the same shape as ``value`` with all string leaves
        having their ``${ENV_VAR}`` tokens replaced. Non-string scalars and
        ``None`` are returned unchanged.

    Raises:
        SubstitutionError: If any required environment variable is unset or
            set to the empty string.
    """
    mapping = os.environ if env is None else env
    return _walk(value, mapping)


def _walk(value: Any, env: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _substitute_string(value, env)
    if isinstance(value, dict):
        return {key: _walk(item, env) for key, item in value.items()}
    if isinstance(value, list):
        return [_walk(item, env) for item in value]
    if isinstance(value, tuple):
        return tuple(_walk(item, env) for item in value)
    return value


__all__ = ["SubstitutionError", "substitute_env"]
