"""Tests for :mod:`moaxy.pipeline.prompts`.

These tests pin the literal substrings that the validation contract
asserts on the default prompts. Editing the prompt text without
updating these substring assertions (and the matching contract
assertions) is a regression.

The contract pins the following invariants:

* ``VAL-PIPE-010`` — the default reflector system prompt contains
  ``REFLECT_CONFIDENCE:`` (v1-v4 invariant; the prompt instructs the
  model to emit it on the last line).
* ``VAL-PIPE-EXTRA-025`` — the default advisor system prompt contains
  ``ADVISOR_DECISION:``, ``ADVISOR_SCORE:``, ``ADVISOR_ISSUES:``,
  ``ADVISOR_SUGGESTIONS:`` (the M5 cross-critique markers).
* ``VAL-PIPE-EXTRA-026`` — the default advisor system prompt still
  contains the legacy ``ADVISOR_APPROVE`` and ``ADVISOR_REVISE:``
  markers so v1-v4 advisor models keep working unchanged.

The M5 deltas also extend ``DEFAULT_REFLECT_PROMPT`` with an
optional ``SCORE:`` line.
"""

from __future__ import annotations

import pytest

from moaxy.pipeline.prompts import DEFAULT_ADVISOR_PROMPT, DEFAULT_REFLECT_PROMPT

# ────────────────────────────────────────────────────────────────────
# DEFAULT_REFLECT_PROMPT — VAL-PIPE-010 (and M5 SCORE: addition)
# ────────────────────────────────────────────────────────────────────


class TestDefaultReflectPrompt:
    """The default reflector prompt contains the contract substrings."""

    def test_is_non_empty_string(self):
        assert isinstance(DEFAULT_REFLECT_PROMPT, str)
        assert len(DEFAULT_REFLECT_PROMPT) > 0

    def test_contains_reflect_confidence_marker(self):
        # VAL-PIPE-010: the literal substring "REFLECT_CONFIDENCE:" must
        # be present in the system prompt sent to the critique LLM call.
        assert "REFLECT_CONFIDENCE:" in DEFAULT_REFLECT_PROMPT

    def test_contains_score_marker(self):
        # M5 delta: the optional SCORE: <0-10> line is requested alongside
        # REFLECT_CONFIDENCE: so models that emit a 0-10 self-score can
        # be parsed by the orchestrator's parse_score helper.
        assert "SCORE:" in DEFAULT_REFLECT_PROMPT

    def test_score_marker_is_uppercase_with_colon(self):
        # The parser is anchored and case-sensitive; pin the exact form
        # the model must emit (uppercase, with a trailing colon).
        assert "SCORE:" in DEFAULT_REFLECT_PROMPT
        # The lowercased prompt must still contain a "score:" substring
        # (case-insensitive check), confirming the marker is present in
        # some case form. The anchored case-sensitive regex requires
        # uppercase "SCORE:" to match.
        assert "score:" in DEFAULT_REFLECT_PROMPT.lower()

    def test_describes_confidence_range(self):
        # The prompt must mention the 0.0-1.0 range so the model knows
        # how to express its confidence.
        assert "0.0-1.0" in DEFAULT_REFLECT_PROMPT

    def test_describes_score_range(self):
        # The M5 addition mentions the 0-10 integer range for the
        # self-score so the model knows the expected format.
        assert "0-10" in DEFAULT_REFLECT_PROMPT

    def test_does_not_drop_legacy_reflect_confidence(self):
        # Backward compatibility: REFLECT_CONFIDENCE: must STILL be
        # present after the M5 SCORE: addition (VAL-PIPE-010 invariant).
        assert DEFAULT_REFLECT_PROMPT.count("REFLECT_CONFIDENCE:") == 1


# ────────────────────────────────────────────────────────────────────
# DEFAULT_ADVISOR_PROMPT — VAL-PIPE-012, VAL-PIPE-013, M5 cross-critique
# ────────────────────────────────────────────────────────────────────


class TestDefaultAdvisorPrompt:
    """The default advisor prompt contains the contract substrings."""

    def test_is_non_empty_string(self):
        assert isinstance(DEFAULT_ADVISOR_PROMPT, str)
        assert len(DEFAULT_ADVISOR_PROMPT) > 0

    def test_contains_advisor_decision_marker(self):
        # M5 cross-critique: the new ADVISOR_DECISION: marker is requested
        # on its own line. The parser (M5) reads APPROVE / REVISE from
        # the value following the colon.
        assert "ADVISOR_DECISION:" in DEFAULT_ADVISOR_PROMPT

    def test_contains_advisor_score_marker(self):
        # M5 cross-critique: ADVISOR_SCORE: <0-10> is requested as an
        # optional integer self-assessment, parsed by parse_advisor_score.
        assert "ADVISOR_SCORE:" in DEFAULT_ADVISOR_PROMPT

    def test_contains_advisor_issues_marker(self):
        # M5 cross-critique: ADVISOR_ISSUES: header is requested followed
        # by a bulleted list of problems (parsed by parse_advisor_issues).
        assert "ADVISOR_ISSUES:" in DEFAULT_ADVISOR_PROMPT

    def test_contains_advisor_suggestions_marker(self):
        # M5 cross-critique: ADVISOR_SUGGESTIONS: header is requested
        # followed by a bulleted list of improvement suggestions.
        assert "ADVISOR_SUGGESTIONS:" in DEFAULT_ADVISOR_PROMPT

    def test_still_contains_legacy_advisor_approve(self):
        # VAL-PIPE-012 / VAL-PIPE-EXTRA-026 backward-compat invariant:
        # v1-v4 advisor models that emit only the legacy ADVISOR_APPROVE
        # marker must still be parseable. The prompt must STILL contain
        # the legacy marker.
        assert "ADVISOR_APPROVE" in DEFAULT_ADVISOR_PROMPT

    def test_still_contains_legacy_advisor_revise(self):
        # VAL-PIPE-013 / VAL-PIPE-EXTRA-026 backward-compat invariant:
        # v1-v4 advisor models that emit only the legacy ADVISOR_REVISE:
        # marker must still be parseable. The prompt must STILL contain
        # the legacy marker.
        assert "ADVISOR_REVISE:" in DEFAULT_ADVISOR_PROMPT


# ────────────────────────────────────────────────────────────────────
# Combined substring pin (single-table test that catches regressions
# in any of the contract invariants in one place)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "prompt_name,prompt_text,substring",
    [
        # VAL-PIPE-010 invariant
        ("DEFAULT_REFLECT_PROMPT", DEFAULT_REFLECT_PROMPT, "REFLECT_CONFIDENCE:"),
        # M5 reflector addition
        ("DEFAULT_REFLECT_PROMPT", DEFAULT_REFLECT_PROMPT, "SCORE:"),
        # VAL-PIPE-012 / VAL-PIPE-EXTRA-026 legacy advisor markers
        ("DEFAULT_ADVISOR_PROMPT", DEFAULT_ADVISOR_PROMPT, "ADVISOR_APPROVE"),
        ("DEFAULT_ADVISOR_PROMPT", DEFAULT_ADVISOR_PROMPT, "ADVISOR_REVISE:"),
        # VAL-PIPE-EXTRA-025 M5 cross-critique markers
        ("DEFAULT_ADVISOR_PROMPT", DEFAULT_ADVISOR_PROMPT, "ADVISOR_DECISION:"),
        ("DEFAULT_ADVISOR_PROMPT", DEFAULT_ADVISOR_PROMPT, "ADVISOR_SCORE:"),
        ("DEFAULT_ADVISOR_PROMPT", DEFAULT_ADVISOR_PROMPT, "ADVISOR_ISSUES:"),
        ("DEFAULT_ADVISOR_PROMPT", DEFAULT_ADVISOR_PROMPT, "ADVISOR_SUGGESTIONS:"),
    ],
)
def test_prompt_substring_invariant(prompt_name, prompt_text, substring):
    """Every contract-pinned substring must be present in the named prompt.

    This is a single table of contract assertions; adding a new
    substring here is the canonical way to pin a new validator
    invariant. The ``prompt_name`` parameter is included in the
    test id for clear failure messages.
    """
    assert substring in prompt_text, (
        f"{prompt_name} is missing required substring {substring!r}; "
        f"the contract asserts this substring and it cannot be removed "
        f"without updating validation-contract.md."
    )


# ────────────────────────────────────────────────────────────────────
# Backward-compat invariant: all v1-v4 markers preserved on M5
# ────────────────────────────────────────────────────────────────────


class TestBackwardCompatMarkers:
    """The M5 prompt extensions must NOT remove any v1-v4 marker.

    This is the explicit backward-compat contract (VAL-PIPE-EXTRA-026
    and the AGENTS.md "Backwards compatibility" rules). If a future
    refactor drops ``ADVISOR_APPROVE`` or ``ADVISOR_REVISE:`` from the
    default prompt, v1-v4 advisor models stop being parseable and
    the contract breaks.
    """

    def test_reflect_prompt_still_requests_reflect_confidence(self):
        # Even after the M5 SCORE: addition, the prompt must still ask
        # the model to emit REFLECT_CONFIDENCE: (VAL-PIPE-010).
        assert "REFLECT_CONFIDENCE:" in DEFAULT_REFLECT_PROMPT
        # The prompt should not just mention the marker incidentally
        # — it should instruct the model to emit it. Pin the phrase
        # "output REFLECT_CONFIDENCE:" so the model's instruction is
        # unambiguous.
        assert "output REFLECT_CONFIDENCE:" in DEFAULT_REFLECT_PROMPT

    def test_advisor_prompt_still_requests_advisor_approve(self):
        # VAL-PIPE-012 backward-compat: the legacy ADVISOR_APPROVE
        # marker is still requested in the default prompt.
        assert "ADVISOR_APPROVE" in DEFAULT_ADVISOR_PROMPT

    def test_advisor_prompt_still_requests_advisor_revise(self):
        # VAL-PIPE-013 backward-compat: the legacy ADVISOR_REVISE:
        # marker is still requested in the default prompt.
        assert "ADVISOR_REVISE:" in DEFAULT_ADVISOR_PROMPT

    def test_advisor_prompt_instructs_advisor_decision(self):
        # The M5 cross-critique prompt must instruct the model to emit
        # ADVISOR_DECISION: APPROVE or REVISE. Pin the literal values
        # so the parser can rely on them.
        assert "APPROVE" in DEFAULT_ADVISOR_PROMPT
        assert "REVISE" in DEFAULT_ADVISOR_PROMPT

    def test_advisor_prompt_describes_advisor_score_range(self):
        # The M5 cross-critique prompt mentions the 0-10 integer range
        # for ADVISOR_SCORE so the model knows the expected format.
        assert "0-10" in DEFAULT_ADVISOR_PROMPT
