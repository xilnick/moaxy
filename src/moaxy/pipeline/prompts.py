"""Default system prompts for the reflection and advisor stages.

These prompts are the fallback used when a route does not configure a
custom ``reflection.system_prompt`` or ``advisor.system_prompt``. They
are intentionally small and stable: the validator (VAL-PIPE-010)
asserts that the reflector prompt contains the literal substring
``REFLECT_CONFIDENCE:`` and that the advisor prompt contains both
``ADVISOR_APPROVE`` and ``ADVISOR_REVISE:``. Editing the text? Update
the validator contract and the tests that pin the substring.
"""

from __future__ import annotations

DEFAULT_REFLECT_PROMPT: str = (
    "You are a critical reviewer of your own previous answer. "
    "Identify any factual errors, missing edge cases, or unclear "
    "explanations. Output your critique, then output "
    "REFLECT_CONFIDENCE: <0.0-1.0> on the last line, where 1.0 means "
    "the previous answer is correct and complete."
)

DEFAULT_ADVISOR_PROMPT: str = (
    "You are an advisor reviewing a previous assistant's answer. If the "
    "previous assistant's answer is correct, complete, and well-reasoned, "
    "respond with ADVISOR_APPROVE. Otherwise, produce an improved answer "
    "prefixed with ADVISOR_REVISE: followed by the full revised response."
)


__all__ = [
    "DEFAULT_ADVISOR_PROMPT",
    "DEFAULT_REFLECT_PROMPT",
]
