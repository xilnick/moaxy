"""Configuration file discovery and loading for moaxy.

Discovery order:

1. ``MOAXY_CONFIG_PATH`` environment variable (if set, must point at an
   existing readable file — the caller has explicitly opted in).
2. ``./config.yaml`` in the current working directory.
3. ``./config.yml`` in the current working directory.
4. ``./config.json`` in the current working directory.
5. Built-in defaults: a :class:`MoaxyConfig` with ``backends=[]`` and
   ``routes=[]``.

After loading the file, the raw dict is run through
:func:`moaxy.config.envsubst.substitute_env` to expand ``${ENV_VAR}`` tokens.
The substituted dict is then validated by the Pydantic v2
:class:`MoaxyConfig` schema, which enforces all structural and range
invariants.

The function returns a typed :class:`MoaxyConfig` instance; callers should
not bypass the type system by re-parsing the file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from moaxy.config.envsubst import substitute_env
from moaxy.models.config import MoaxyConfig

CONFIG_PATH_ENV_VAR = "MOAXY_CONFIG_PATH"
DEFAULT_CANDIDATE_NAMES: tuple[str, ...] = ("config.yaml", "config.yml", "config.json")


class ConfigNotFoundError(FileNotFoundError):
    """Raised when ``MOAXY_CONFIG_PATH`` is set but the file does not exist."""


def find_config_file(
    path: str | os.PathLike[str] | None = None,
    *,
    env: os._Environ[str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Return the path of the first existing config file, or ``None``.

    Args:
        path: Explicit override. Takes precedence over every other source.
        env: Mapping to consult for ``MOAXY_CONFIG_PATH``. Defaults to
            :data:`os.environ`. Use ``env={}`` to suppress the env var.
        cwd: Working directory used to resolve the default candidate names.
            Defaults to :data:`os.getcwd`. Exposed for testability.

    Returns:
        Absolute path of the first existing config file, or ``None`` if no
        candidate exists.
    """
    if path is not None:
        return Path(path)

    env_mapping = os.environ if env is None else env
    if CONFIG_PATH_ENV_VAR in env_mapping:
        env_path = Path(env_mapping[CONFIG_PATH_ENV_VAR])
        if env_path.is_file():
            return env_path
        raise ConfigNotFoundError(
            f"{CONFIG_PATH_ENV_VAR}={env_path!s} does not point at an existing file"
        )

    base = Path(cwd) if cwd is not None else Path.cwd()
    for candidate in DEFAULT_CANDIDATE_NAMES:
        full = base / candidate
        if full.is_file():
            return full
    return None


def _read_payload(path: Path) -> dict[str, Any]:
    """Read a config file and return the parsed dict.

    JSON files are detected by suffix. All other suffixes are treated as YAML.
    A YAML mapping at the top level is required; an empty file or a non-mapping
    payload is a config error.
    """
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise yaml.YAMLError(
            f"config file {path!s} must contain a YAML/JSON mapping at the top level; "
            f"got {type(data).__name__}"
        )
    return data


def parse_config_payload(
    payload: dict[str, Any],
    *,
    env: os._Environ[str] | None = None,
) -> MoaxyConfig:
    """Apply ${ENV} substitution to a parsed dict and return a typed config.

    The function does not read any file or consult ``MOAXY_CONFIG_PATH``; it
    operates entirely on the supplied mapping. This is the hook used by tests
    that want to inject a payload directly.
    """
    substituted = substitute_env(payload, env=env)
    return MoaxyConfig.model_validate(substituted)


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    env: os._Environ[str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> MoaxyConfig:
    """Discover, read, expand, and validate the moaxy configuration.

    Args:
        path: Explicit path to a config file. Overrides ``MOAXY_CONFIG_PATH``
            and the default candidate names.
        env: Mapping to consult for both ``MOAXY_CONFIG_PATH`` and any
            ``${ENV_VAR}`` tokens in the file. Defaults to :data:`os.environ`.
        cwd: Working directory for default candidate resolution. Defaults to
            :data:`os.getcwd`.

    Returns:
        A :class:`MoaxyConfig` instance. When no config file is discoverable,
        a default instance with ``backends=[]`` and ``routes=[]`` is returned.

    Raises:
        ConfigNotFoundError: ``MOAXY_CONFIG_PATH`` is set but the file does
            not exist.
        yaml.YAMLError: The YAML config file is malformed.
        json.JSONDecodeError: The JSON config file is malformed.
        SubstitutionError: A ``${ENV_VAR}`` token in the file references an
            unset or empty environment variable.
        pydantic.ValidationError: The expanded payload fails schema
            validation (range, literal, structural, cross-reference).
    """
    env_mapping = os.environ if env is None else env
    found = find_config_file(path=path, env=env_mapping, cwd=cwd)
    if found is None:
        return MoaxyConfig(backends=[])
    payload = _read_payload(found)
    return parse_config_payload(payload, env=env_mapping)


__all__ = [
    "CONFIG_PATH_ENV_VAR",
    "ConfigNotFoundError",
    "DEFAULT_CANDIDATE_NAMES",
    "find_config_file",
    "load_config",
    "parse_config_payload",
]
