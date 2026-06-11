"""Default system prompts for the reflection and advisor stages.

These prompts are the fallback used when a route does not configure a
custom ``reflection.system_prompt`` or ``advisor.system_prompt``. They
are intentionally small and stable: the validator (VAL-PIPE-010)
asserts that the reflector prompt contains the literal substring
``REFLECT_CONFIDENCE:`` and that the advisor prompt contains both
``ADVISOR_APPROVE`` and ``ADVISOR_REVISE:`` (VAL-PIPE-012/013). The
M5 deltas extend the prompts with additional cross-critique markers
(``SCORE:``, ``ADVISOR_DECISION:``, ``ADVISOR_SCORE:``,
``ADVISOR_ISSUES:``, ``ADVISOR_SUGGESTIONS:``) without removing the
v1-v4 markers. Editing the text? Update the validator contract and
the tests that pin the substring.
"""

from __future__ import annotations

DEFAULT_REFLECT_PROMPT: str = (
    "You are a critical reviewer of your own previous answer. "
    "Identify any factual errors, missing edge cases, or unclear "
    "explanations. Output your critique, then output "
    "REFLECT_CONFIDENCE: <0.0-1.0> on the last line, where 1.0 means "
    "the previous answer is correct and complete. "
    "Optionally, you may also output SCORE: <0-10> on a separate line "
    "to give an integer self-assessment of the previous answer's "
    "overall quality (0 = worst, 10 = best)."
)

DEFAULT_ADVISOR_PROMPT: str = (
    "You are an advisor reviewing a previous assistant's answer. "
    "First, decide whether the previous assistant's answer is "
    "acceptable or needs revision, then output ADVISOR_DECISION: "
    "APPROVE or ADVISOR_DECISION: REVISE on its own line. "
    "Optionally output ADVISOR_SCORE: <0-10> on a separate line to "
    "give an integer self-assessment of the previous answer's "
    "quality. Then, if the answer is acceptable, output "
    "ADVISOR_APPROVE on its own line. Otherwise, output the issues "
    "found as a bulleted list under ADVISOR_ISSUES: (one bullet per "
    "line, using '- ' or '* ' or '\u2022 ' markers), followed by "
    "improvement suggestions as a bulleted list under "
    "ADVISOR_SUGGESTIONS: (same bullet format), and finally the "
    "revised answer prefixed with ADVISOR_REVISE: followed by the "
    "full revised response."
)


__all__ = [
    "DEFAULT_ADVISOR_PROMPT",
    "DEFAULT_REFLECT_PROMPT",
]
