"""Message-list builders for the reflection and advisor pipeline stages.

The orchestrator calls one of these helpers to materialise the OpenAI-
shaped ``messages`` payload for each LLM call inside the pipeline:

* :func:`build_reflection_messages` — assembles the message list for the
  critique call in a single reflection turn.
* :func:`build_revision_messages` — assembles the message list for the
  revision call, including the prior critique.
* :func:`build_advisor_messages` — assembles the message list for the
  advisor's pass over the post-reflection answer.

Every helper returns a NEW list of message dicts without mutating the
caller's ``history``. The builder deep-copies each message dict so the
caller's payload (and every other reference to those dicts elsewhere in
the pipeline) is untouched. Sampling parameters (``temperature``,
``top_p``, ``max_tokens``, …) are NOT a concern of these helpers; the
adapter call that consumes the returned messages forwards those fields
verbatim from the original request body.

The output schema is the OpenAI ``chat.completion`` ``messages`` shape:
a list of dicts with ``role`` (one of ``"system"``, ``"user"``,
``"assistant"``) and ``content`` (a string). Each builder prepends a
system-role message (the reflector or advisor system prompt) when
``system_prompt`` is truthy and non-empty.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

_REFLECT_PROMPT_PREFIX = "Please critique"
_REVISE_PROMPT_PREFIX = "Please revise"
_ADVISE_PROMPT_PREFIX = "advise on this"


def _copy_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deep-copy ``history`` so the caller's list and dicts are untouched.

    The result is a brand-new list of brand-new dicts. Future mutations
    (e.g. the orchestrator appending an ``assistant`` role message) do
    not leak back into the request payload.
    """
    return [deepcopy(msg) for msg in history]


def _system_message(system_prompt: str | None) -> list[dict[str, Any]]:
    """Return a single-element system-message list, or an empty list.

    An empty or ``None`` system prompt yields no system message; the
    orchestrator's downstream adapter call inherits whatever system
    messages the client already supplied in ``history``.
    """
    if not system_prompt:
        return []
    return [{"role": "system", "content": system_prompt}]


def build_reflection_messages(
    history: list[dict[str, Any]],
    answer: str,
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    """Build the message list for the reflection critique LLM call.

    The returned list contains, in order:

    1. The reflector ``system_prompt`` (when non-empty), as a system-role
       message.
    2. A deep copy of the client's ``history``.
    3. A user-role message that names the previous answer and asks the
       model to critique it (``"Please critique the following answer:\\n"
       + answer``). The literal prefix ``"Please critique"`` is the
       pinning the validation contract expects.

    The input ``history`` list and every dict it contains are NOT
    mutated. The returned list is safe for the orchestrator to mutate
    further (e.g. by appending a ``role`` change for the next stage).
    """
    messages: list[dict[str, Any]] = []
    messages.extend(_system_message(system_prompt))
    messages.extend(_copy_history(history))
    messages.append(
        {
            "role": "user",
            "content": f"{_REFLECT_PROMPT_PREFIX} the following answer:\n{answer}",
        }
    )
    return messages


def build_revision_messages(
    history: list[dict[str, Any]],
    answer: str,
    critique: str,
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    """Build the message list for the reflection revision LLM call.

    The returned list contains, in order:

    1. The reflector ``system_prompt`` (when non-empty).
    2. A deep copy of the client's ``history``.
    3. A user-role message asking the model to critique the previous
       answer (the same content used by
       :func:`build_reflection_messages`).
    4. An assistant-role message holding the model's critique so the
       revision call sees the full dialogue: the original answer, the
       model's own critique, and the request to revise.
    5. A user-role message asking the model to revise its previous
       answer in light of the critique
       (``"Please revise your previous answer based on this critique:\\n"
       + critique``). The literal prefix ``"Please revise"`` matches the
       validation contract.

    The input ``history`` list and its dicts are not mutated.
    """
    messages: list[dict[str, Any]] = []
    messages.extend(_system_message(system_prompt))
    messages.extend(_copy_history(history))
    messages.append(
        {
            "role": "user",
            "content": f"{_REFLECT_PROMPT_PREFIX} the following answer:\n{answer}",
        }
    )
    messages.append({"role": "assistant", "content": critique})
    messages.append(
        {
            "role": "user",
            "content": (
                f"{_REVISE_PROMPT_PREFIX} your previous answer based on this critique:\n{critique}"
            ),
        }
    )
    return messages


def build_advisor_messages(
    history: list[dict[str, Any]],
    answer: str,
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    """Build the message list for the advisor LLM call.

    The returned list contains, in order:

    1. The advisor ``system_prompt`` (when non-empty).
    2. A deep copy of the client's ``history``.
    3. A user-role message that names the post-reflection answer and
       asks the advisor to weigh in
       (``"Please advise on this answer:\\n" + answer``). The literal
       substring ``"advise on this"`` matches the validation contract.

    The input ``history`` list and its dicts are not mutated.
    """
    messages: list[dict[str, Any]] = []
    messages.extend(_system_message(system_prompt))
    messages.extend(_copy_history(history))
    messages.append(
        {
            "role": "user",
            "content": f"Please {_ADVISE_PROMPT_PREFIX} answer:\n{answer}",
        }
    )
    return messages


__all__ = [
    "build_advisor_messages",
    "build_reflection_messages",
    "build_revision_messages",
]
