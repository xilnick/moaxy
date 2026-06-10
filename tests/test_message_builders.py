"""Tests for :mod:`moaxy.pipeline.message_builders`.

The builders are pure functions: they take the conversation history, the
previous answer (and optionally the critique), and a system prompt, and
they return a NEW messages list shaped for an OpenAI-style chat
completion call. The tests below pin:

* the message list shape (roles, content strings, ordering)
* the substrings the validation contract relies on
  (``"Please critique"``, ``"Please revise"``, ``"advise on this"``)
* the immutability contract — the input ``history`` list and the dicts
  it contains are not modified by any builder
* that the helpers do not depend on the request's sampling parameters;
  the orchestrator (a separate concern) is responsible for forwarding
  ``temperature``, ``top_p``, ``max_tokens`` to the adapter.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from moaxy.pipeline.message_builders import (
    build_advisor_messages,
    build_reflection_messages,
    build_revision_messages,
)
from moaxy.pipeline.prompts import (
    DEFAULT_ADVISOR_PROMPT,
    DEFAULT_REFLECT_PROMPT,
)

_REFLECT_SYSTEM = "You are a strict critic. Output REFLECT_CONFIDENCE: <0-1>."
_ADVISOR_SYSTEM = "You are an advisor. Output ADVISOR_APPROVE or ADVISOR_REVISE: ..."


def _sample_history() -> list[dict[str, str]]:
    """A representative multi-turn conversation history."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Four."},
        {"role": "user", "content": "Are you sure?"},
    ]


# -----------------------------------------------------------------------------
# build_reflection_messages
# -----------------------------------------------------------------------------


class TestBuildReflectionMessages:
    """The critique-call message builder returns a fresh, ordered list."""

    def test_returns_a_list(self):
        result = build_reflection_messages(
            history=[{"role": "user", "content": "hi"}],
            answer="hello",
            system_prompt=_REFLECT_SYSTEM,
        )
        assert isinstance(result, list)

    def test_prepends_system_prompt(self):
        history = [{"role": "user", "content": "hi"}]
        result = build_reflection_messages(history, "hello", _REFLECT_SYSTEM)
        assert result[0] == {"role": "system", "content": _REFLECT_SYSTEM}

    def test_history_follows_system_prompt(self):
        history = _sample_history()
        result = build_reflection_messages(history, "answer", _REFLECT_SYSTEM)
        assert result[1:1 + len(history)] == history

    def test_appends_critique_user_message(self):
        history = _sample_history()
        result = build_reflection_messages(history, "answer text", _REFLECT_SYSTEM)
        last = result[-1]
        assert last["role"] == "user"
        assert "Please critique" in last["content"]
        assert "answer text" in last["content"]

    def test_critique_message_includes_full_answer(self):
        answer = "The capital of France is Paris."
        result = build_reflection_messages([], answer, _REFLECT_SYSTEM)
        last = result[-1]
        assert last["content"].endswith(answer)

    def test_no_system_prompt_when_none(self):
        history = [{"role": "user", "content": "hi"}]
        result = build_reflection_messages(history, "answer", None)
        # The first element should be the history entry, not a system message.
        assert result[0] == {"role": "user", "content": "hi"}
        assert all(msg["role"] != "system" for msg in result)

    def test_no_system_prompt_when_empty(self):
        history = [{"role": "user", "content": "hi"}]
        result = build_reflection_messages(history, "answer", "")
        assert all(msg["role"] != "system" for msg in result)

    def test_empty_history_is_allowed(self):
        result = build_reflection_messages([], "answer", _REFLECT_SYSTEM)
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": _REFLECT_SYSTEM}
        assert result[1]["role"] == "user"

    def test_default_reflect_prompt_is_a_valid_system_prompt(self):
        """The default reflector prompt passes through as a system message."""
        result = build_reflection_messages(
            history=[{"role": "user", "content": "hi"}],
            answer="answer",
            system_prompt=DEFAULT_REFLECT_PROMPT,
        )
        assert result[0] == {"role": "system", "content": DEFAULT_REFLECT_PROMPT}
        # The default prompt still contains the substring the contract pins.
        assert "REFLECT_CONFIDENCE:" in result[0]["content"]


# -----------------------------------------------------------------------------
# build_revision_messages
# -----------------------------------------------------------------------------


class TestBuildRevisionMessages:
    """The revision-call builder includes the critique and the revise ask."""

    def test_returns_a_list(self):
        result = build_revision_messages(
            history=[{"role": "user", "content": "hi"}],
            answer="answer",
            critique="critique",
            system_prompt=_REFLECT_SYSTEM,
        )
        assert isinstance(result, list)

    def test_prepends_system_prompt(self):
        history = [{"role": "user", "content": "hi"}]
        result = build_revision_messages(history, "answer", "crit", _REFLECT_SYSTEM)
        assert result[0] == {"role": "system", "content": _REFLECT_SYSTEM}

    def test_history_follows_system_prompt(self):
        history = _sample_history()
        result = build_revision_messages(history, "a", "c", _REFLECT_SYSTEM)
        assert result[1:1 + len(history)] == history

    def test_contains_critique_user_message(self):
        history = _sample_history()
        result = build_revision_messages(history, "answer", "crit", _REFLECT_SYSTEM)
        # Find the user message that contains "Please critique"
        user_with_critique = next(
            m for m in result if m["role"] == "user" and "Please critique" in m["content"]
        )
        assert "answer" in user_with_critique["content"]

    def test_critique_message_appears_as_assistant_role(self):
        history = _sample_history()
        critique = "The answer is missing the unit."
        result = build_revision_messages(history, "answer", critique, _REFLECT_SYSTEM)
        # The assistant message that holds the critique is in the middle.
        assistant_messages = [m for m in result if m["role"] == "assistant"]
        assert any(m["content"] == critique for m in assistant_messages)

    def test_appends_revise_user_message(self):
        history = _sample_history()
        result = build_revision_messages(history, "a", "c", _REFLECT_SYSTEM)
        last = result[-1]
        assert last["role"] == "user"
        assert "Please revise" in last["content"]
        assert "c" in last["content"]

    def test_revise_message_mentions_critique(self):
        history = _sample_history()
        critique = "needs more detail"
        result = build_revision_messages(history, "a", critique, _REFLECT_SYSTEM)
        last = result[-1]
        assert critique in last["content"]

    def test_no_system_prompt_when_none(self):
        history = [{"role": "user", "content": "hi"}]
        result = build_revision_messages(history, "a", "c", None)
        assert all(m["role"] != "system" for m in result)

    def test_empty_history_is_allowed(self):
        result = build_revision_messages([], "a", "c", _REFLECT_SYSTEM)
        # system + critique-user + critique-assistant + revise-user
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[-1]["role"] == "user"

    def test_default_reflect_prompt_passes_through(self):
        result = build_revision_messages(
            history=[{"role": "user", "content": "hi"}],
            answer="a",
            critique="c",
            system_prompt=DEFAULT_REFLECT_PROMPT,
        )
        assert result[0] == {"role": "system", "content": DEFAULT_REFLECT_PROMPT}


# -----------------------------------------------------------------------------
# build_advisor_messages
# -----------------------------------------------------------------------------


class TestBuildAdvisorMessages:
    """The advisor-call builder produces the expected message sequence."""

    def test_returns_a_list(self):
        result = build_advisor_messages(
            history=[{"role": "user", "content": "hi"}],
            answer="answer",
            system_prompt=_ADVISOR_SYSTEM,
        )
        assert isinstance(result, list)

    def test_prepends_advisor_system_prompt(self):
        history = [{"role": "user", "content": "hi"}]
        result = build_advisor_messages(history, "answer", _ADVISOR_SYSTEM)
        assert result[0] == {"role": "system", "content": _ADVISOR_SYSTEM}

    def test_history_follows_system_prompt(self):
        history = _sample_history()
        result = build_advisor_messages(history, "answer", _ADVISOR_SYSTEM)
        assert result[1:1 + len(history)] == history

    def test_appends_advise_user_message(self):
        history = _sample_history()
        result = build_advisor_messages(history, "answer", _ADVISOR_SYSTEM)
        last = result[-1]
        assert last["role"] == "user"
        assert "advise on this" in last["content"]
        assert "answer" in last["content"]

    def test_advise_message_includes_full_answer(self):
        answer = "My final answer is 42."
        result = build_advisor_messages([], answer, _ADVISOR_SYSTEM)
        last = result[-1]
        assert last["content"].endswith(answer)

    def test_no_system_prompt_when_none(self):
        history = [{"role": "user", "content": "hi"}]
        result = build_advisor_messages(history, "answer", None)
        assert all(m["role"] != "system" for m in result)

    def test_no_system_prompt_when_empty(self):
        history = [{"role": "user", "content": "hi"}]
        result = build_advisor_messages(history, "answer", "")
        assert all(m["role"] != "system" for m in result)

    def test_empty_history_is_allowed(self):
        result = build_advisor_messages([], "answer", _ADVISOR_SYSTEM)
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": _ADVISOR_SYSTEM}
        assert result[1]["role"] == "user"

    def test_default_advisor_prompt_passes_through(self):
        result = build_advisor_messages(
            history=[{"role": "user", "content": "hi"}],
            answer="answer",
            system_prompt=DEFAULT_ADVISOR_PROMPT,
        )
        assert result[0] == {"role": "system", "content": DEFAULT_ADVISOR_PROMPT}
        assert "ADVISOR_APPROVE" in result[0]["content"]
        assert "ADVISOR_REVISE:" in result[0]["content"]


# -----------------------------------------------------------------------------
# Immutability contract
# -----------------------------------------------------------------------------


class TestImmutability:
    """Builders must not mutate the input ``history`` list or its dicts."""

    @pytest.mark.parametrize(
        "builder",
        [
            build_reflection_messages,
            build_advisor_messages,
        ],
    )
    def test_history_list_not_mutated(self, builder):
        history = _sample_history()
        original = deepcopy(history)
        builder(history, "answer", _REFLECT_SYSTEM if builder is build_reflection_messages else _ADVISOR_SYSTEM)
        assert history == original

    def test_revision_history_list_not_mutated(self):
        history = _sample_history()
        original = deepcopy(history)
        build_revision_messages(history, "answer", "critique", _REFLECT_SYSTEM)
        assert history == original

    @pytest.mark.parametrize(
        "builder",
        [
            build_reflection_messages,
            build_advisor_messages,
        ],
    )
    def test_history_dicts_not_mutated(self, builder):
        history = _sample_history()
        original_dicts = [deepcopy(m) for m in history]
        builder(history, "answer", _REFLECT_SYSTEM if builder is build_reflection_messages else _ADVISOR_SYSTEM)
        # Each input dict must still equal its pre-call snapshot.
        for original, current in zip(original_dicts, history, strict=True):
            assert current == original

    def test_revision_history_dicts_not_mutated(self):
        history = _sample_history()
        original_dicts = [deepcopy(m) for m in history]
        build_revision_messages(history, "answer", "critique", _REFLECT_SYSTEM)
        for original, current in zip(original_dicts, history, strict=True):
            assert current == original

    @pytest.mark.parametrize(
        "builder,system",
        [
            (build_reflection_messages, _REFLECT_SYSTEM),
            (build_advisor_messages, _ADVISOR_SYSTEM),
        ],
    )
    def test_output_does_not_share_dicts_with_history(self, builder, system):
        history = _sample_history()
        result = builder(history, "answer", system)
        # The output is a new list; the messages that mirror history must
        # be different dict objects (no aliasing back into the input).
        for original, produced in zip(history, result[1:1 + len(history)], strict=True):
            assert produced is not original

    def test_revision_output_does_not_share_dicts_with_history(self):
        history = _sample_history()
        result = build_revision_messages(history, "answer", "critique", _REFLECT_SYSTEM)
        for original, produced in zip(history, result[1:1 + len(history)], strict=True):
            assert produced is not original

    @pytest.mark.parametrize(
        "builder",
        [
            build_reflection_messages,
            build_advisor_messages,
        ],
    )
    def test_output_list_is_a_new_list(self, builder):
        history = [{"role": "user", "content": "hi"}]
        result = builder(history, "answer", "system")
        assert result is not history


# -----------------------------------------------------------------------------
# Returned-shape invariants
# -----------------------------------------------------------------------------


class TestMessageShape:
    """Every returned message has the required ``role`` and ``content`` keys."""

    @pytest.mark.parametrize(
        "builder,system",
        [
            (build_reflection_messages, _REFLECT_SYSTEM),
            (build_revision_messages, _REFLECT_SYSTEM),
            (build_advisor_messages, _ADVISOR_SYSTEM),
        ],
    )
    def test_every_message_has_role_and_content(self, builder, system):
        history = _sample_history()
        if builder is build_revision_messages:
            result = builder(history, "answer", "critique", system)
        else:
            result = builder(history, "answer", system)
        for message in result:
            assert isinstance(message, dict)
            assert "role" in message
            assert "content" in message
            assert message["role"] in {"system", "user", "assistant"}
            assert isinstance(message["content"], str)

    def test_revision_message_role_ordering(self):
        """The revision call has a fixed role ordering: user (critique),
        assistant (critique), user (revise). The system message (if any)
        is first; the rest of ``history`` precedes the dialogue."""
        history = _sample_history()
        result = build_revision_messages(history, "a", "c", _REFLECT_SYSTEM)
        history_len = len(history)
        # The last three messages (after history) are: user, assistant, user.
        last_three = result[history_len + 1:]
        assert last_three[0]["role"] == "user"
        assert last_three[1]["role"] == "assistant"
        assert last_three[2]["role"] == "user"


# -----------------------------------------------------------------------------
# Sampling parameters are a separate concern (the orchestrator forwards them)
# -----------------------------------------------------------------------------


class TestSamplingParameterForwarding:
    """The builders don't touch sampling parameters; the orchestrator does.

    The validation contract (VAL-PIPE-042) requires that ``temperature``,
    ``top_p``, and ``max_tokens`` (and any other OpenAI body fields) are
    forwarded verbatim by the orchestrator's adapter call. The
    message-builders are deliberately NOT responsible for that: they
    produce the ``messages`` list, not the full request body. These
    tests pin that boundary so a future refactor doesn't quietly start
    dropping the body fields.
    """

    @pytest.mark.parametrize(
        "builder,system",
        [
            (build_reflection_messages, _REFLECT_SYSTEM),
            (build_advisor_messages, _ADVISOR_SYSTEM),
        ],
    )
    def test_does_not_propagate_sampling_params(self, builder, system):
        """A request body with sampling params passes through unchanged
        when the orchestrator merges it back with the built messages."""
        body = {
            "model": "minimax-m3:cloud",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 100,
            "stop": ["END"],
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "n": 1,
        }
        history = body["messages"]
        result = builder(history, "answer", system)
        # The builder returned a list of messages; the orchestrator is
        # expected to re-attach the sampling fields at call time. The
        # builder itself does not add or drop any of these fields.
        assert all(isinstance(m, dict) for m in result)
        for message in result:
            # None of the messages should carry the sampling fields.
            for sampling_field in (
                "temperature",
                "top_p",
                "max_tokens",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "n",
            ):
                assert sampling_field not in message

    def test_orchestrator_style_merge_preserves_sampling_params(self):
        """The orchestrator is expected to merge the built messages back
        into the request body; this test simulates that pattern and
        asserts the sampling fields survive untouched."""
        body = {
            "model": "minimax-m3:cloud",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.42,
            "top_p": 0.88,
            "max_tokens": 256,
        }
        messages = build_reflection_messages(
            history=body["messages"],
            answer="answer",
            system_prompt=_REFLECT_SYSTEM,
        )
        # Simulate orchestrator: build the call body from the original
        # request, substituting the new messages list and keeping the
        # other fields.
        call_body = {k: v for k, v in body.items() if k != "messages"}
        call_body["messages"] = messages
        assert call_body["temperature"] == 0.42
        assert call_body["top_p"] == 0.88
        assert call_body["max_tokens"] == 256
        assert call_body["messages"] is messages
