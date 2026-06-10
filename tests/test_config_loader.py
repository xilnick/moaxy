"""Tests for the config loader and ${ENV_VAR} substitution.

Covers: envsubst (single token, multiple tokens, nested structures, missing
var, empty var), loader (MOAXY_CONFIG_PATH, default discovery order,
.yaml/.yml/.json, missing file, malformed YAML, JSON parse, default
MoaxyConfig fallback), Pydantic schema round-trip on a loaded config, and the
canonical example parse end-to-end.
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
from moaxy.models.config import MoaxyConfig

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


# ── module surface ─────────────────────────────────────────────────────


class TestModuleSurface:
    def test_config_package_imports(self):
        from moaxy import config  # noqa: F401

    def test_exports(self):
        from moaxy.config import (  # noqa: F401
            ConfigNotFoundError,
            SubstitutionError,
            find_config_file,
            load_config,
            parse_config_payload,
            substitute_env,
        )
