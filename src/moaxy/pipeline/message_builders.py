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
_ADVISOR_REVISE_PROMPT_PREFIX = "incorporate the advisor's feedback"

# Cold-grading rubric used by the M8 ``fresh_context: true`` reflection
# mode. The rubric asks the model to grade the candidate answer in
# isolation — without the original system prompt, user request, or chat
# history. The literal substring ``REFLECT_CONFIDENCE:`` is preserved
# so the downstream parser (VAL-PIPE-010) still works; ``SCORE:`` is
# also kept so the M5 weighted-signal code path is unaffected.
_FRESH_CONTEXT_RUBRIC: str = (
    "You are a critical reviewer grading the following answer in "
    "isolation. Treat the answer as if you have never seen the "
    "original question, system instructions, or any prior "
    "conversation. Identify any factual errors, missing edge cases, "
    "or unclear explanations purely on the merits of the answer "
    "text. Output your critique, then output REFLECT_CONFIDENCE: "
    "<0.0-1.0> on the last line, where 1.0 means the previous "
    "answer is correct and complete. Optionally, you may also "
    "output SCORE: <0-10> on a separate line to give an integer "
    "self-assessment of the previous answer's overall quality "
    "(0 = worst, 10 = best)."
)

# Cold-revision instruction used by the M8 ``fresh_context: true``
# reflection mode. The model is asked to revise the candidate answer
# in light of the critique, still without the original system
# prompt, user request, or chat history. The literal substring
# ``Please revise`` is preserved so the existing revision-path
# grep tests still find the marker.
_FRESH_CONTEXT_REVISE: str = (
    "Please revise your previous answer in light of this critique. "
    "Stay in the isolated grading context: do not invent any "
    "constraints that are not present in the answer or the "
    "critique."
)


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
    *,
    fresh_context: bool = False,
) -> list[dict[str, Any]]:
    """Build the message list for the reflection critique LLM call.

    The returned list contains, in order:

    1. The reflector ``system_prompt`` (when non-empty), as a system-role
       message. *Skipped* when ``fresh_context: true``; the
       :data:`_FRESH_CONTEXT_RUBRIC` is used instead so the critique
       call is genuinely isolated from the original request context.
    2. A deep copy of the client's ``history``. *Skipped* when
       ``fresh_context: true``; the original user request and chat
       history are NOT included.
    3. A user-role message that names the previous answer and asks the
       model to critique it (``"Please critique the following answer:\\n"
       + answer``). The literal prefix ``"Please critique"`` is the
       pinning the validation contract expects. The trailing
       ``REFLECT_CONFIDENCE:`` / ``SCORE:`` instructions are part of
       the system rubric in ``fresh_context`` mode (so the parser
       contract VAL-PIPE-010 still sees the marker in the system
       message) and part of the user-role request otherwise.

    The input ``history`` list and every dict it contains are NOT
    mutated. The returned list is safe for the orchestrator to mutate
    further (e.g. by appending a ``role`` change for the next stage).
    """
    messages: list[dict[str, Any]] = []
    if fresh_context:
        # M8: in fresh-context mode the critique sees only the rubric
        # and the candidate answer. No client system prompt, no client
        # history, no user request. The rubric itself is a system
        # message so the model's behaviour is constrained even without
        # the original system prompt.
        messages.append({"role": "system", "content": _FRESH_CONTEXT_RUBRIC})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{_REFLECT_PROMPT_PREFIX} the following answer:\n{answer}"
                ),
            }
        )
        return messages
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
    *,
    fresh_context: bool = False,
) -> list[dict[str, Any]]:
    """Build the message list for the reflection revision LLM call.

    The returned list contains, in order:

    1. The reflector ``system_prompt`` (when non-empty). *Skipped* when
       ``fresh_context: true``; the :data:`_FRESH_CONTEXT_RUBRIC` is
       used instead so the revision call stays in the isolated
       grading context.
    2. A deep copy of the client's ``history``. *Skipped* when
       ``fresh_context: true``.
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
       validation contract. In ``fresh_context`` mode the message
       uses the cold-revision wording
       (:data:`_FRESH_CONTEXT_REVISE`) so the model stays in
       isolated-grading mode.

    The input ``history`` list and its dicts are not mutated.
    """
    messages: list[dict[str, Any]] = []
    if fresh_context:
        # M8: in fresh-context mode the revision sees only the rubric,
        # the candidate answer, the critique, and a cold-revision
        # instruction. No client system prompt, no client history, no
        # user request.
        messages.append({"role": "system", "content": _FRESH_CONTEXT_RUBRIC})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{_REFLECT_PROMPT_PREFIX} the following answer:\n{answer}"
                ),
            }
        )
        messages.append({"role": "assistant", "content": critique})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{_FRESH_CONTEXT_REVISE}\n\nCritique:\n{critique}"
                ),
            }
        )
        return messages
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


def build_advisor_revision_messages(
    history: list[dict[str, Any]],
    answer: str,
    advisor_feedback: str,
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    """Build the message list for the primary model's post-advisor revision.

    The returned list contains, in order:

    1. The reflector ``system_prompt`` (when non-empty). The
       post-advisor revision is done by the primary model in the
       reflector role, so the reflector system prompt is the
       appropriate framing.
    2. A deep copy of the client's ``history``.
    3. A user-role message naming the post-reflection answer and
       asking the primary model to critique it (the same content
       used by :func:`build_reflection_messages`).
    4. An assistant-role message holding the primary model's previous
       answer so the revision call sees the full dialogue.
    5. A user-role message asking the primary model to incorporate
       the advisor's feedback into its previous answer
       (``"Please incorporate the advisor's feedback into your
       previous answer:\\n" + advisor_feedback``). The literal
       prefix ``"incorporate the advisor's feedback"`` matches the
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
    messages.append({"role": "assistant", "content": answer})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Please {_ADVISOR_REVISE_PROMPT_PREFIX} into your previous answer:\n{advisor_feedback}"
            ),
        }
    )
    return messages


__all__ = [
    "build_advisor_messages",
    "build_advisor_revision_messages",
    "build_reflection_messages",
    "build_revision_messages",
]
