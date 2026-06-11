"""Deterministic scorers for the M7 benchmark harness.

The :mod:`moaxy.benchmark.scoring.deterministic` module owns the
three deterministic scorers that score a model response against
the corresponding :class:`~moaxy.benchmark.prompts.CodingPrompt`
subclass:

* :func:`score_function_from_docstring` — scores the
  ``function_from_docstring`` category. Parses the model's code
  output (tolerating markdown code fences and prose wrappers),
  extracts the function definition, and runs the hidden test
  cases in a hermetic, sandboxed namespace. Returns ``1.0`` when
  every test case passes and ``0.0`` otherwise.
* :func:`score_bug_fix` — scores the ``bug_fix`` category. Uses
  :class:`difflib.SequenceMatcher` to compute a similarity ratio
  between the model's patched code and the
  :attr:`~moaxy.benchmark.prompts.BugFixPrompt.reference_patch`.
  Returns ``1.0`` when the ratio is at least ``0.9`` (the
  M7-specified threshold) and ``0.0`` otherwise.
* :func:`score_refactor` — scores the ``refactor`` category.
  Regex-searches the model's response for the
  :attr:`~moaxy.benchmark.prompts.RefactorPrompt.target_pattern`.
  Returns ``1.0`` on a match and ``0.0`` on a miss.

The contract (VAL-BENCH-004, VAL-BENCH-005) requires:

* The function-from-docstring scorer returns ``1.0`` for a
  known-correct function and ``0.0`` for a known-wrong function
  (e.g. an off-by-one bug). The score is binary: any test-case
  failure (or any unhandled exception during test execution) is
  ``0.0``; all tests passing is ``1.0``.
* The bug_fix scorer returns ``1.0`` for a known-correct patch
  and ``0.0`` for a known-wrong patch (e.g. the unfixed
  original). The threshold is hardcoded at ``0.9``; values
  below the threshold are ``0.0``.
* All three scorers return a float in ``{0.0, 1.0}``. Empty
  model output is handled gracefully and returns ``0.0`` — the
  scorers never raise on empty input.

The scorers are pure functions: they take a
:class:`~moaxy.benchmark.prompts.CodingPrompt` and the model's
output string, and return a float. They do not call any LLM and
do not perform I/O. The :func:`score_function_from_docstring`
function is the only one that uses :func:`exec`; the namespace is
explicitly constructed and the test cases are run in that
namespace. The exec'd code cannot reach the host process's
globals; the namespace is a fresh dict that only contains the
extracted function and the :mod:`builtins` module.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from moaxy.benchmark.prompts import (
    BugFixPrompt,
    CodingPrompt,
    FunctionFromDocstringPrompt,
    RefactorPrompt,
)

# The fuzzy-match similarity threshold for the ``bug_fix`` scorer.
# Values >= 0.9 score 1.0; values < 0.9 score 0.0. The threshold
# is fixed at the M7-specified value (see ``VAL-BENCH-005``); a
# future edit that wants a different threshold should add a
# contract assertion and update this constant in lockstep.
_BUG_FIX_SIMILARITY_THRESHOLD: float = 0.9
"""The bug_fix scorer's similarity threshold (fuzzy match)."""

# The regex used to extract a Python function definition from a
# model response. The pattern is intentionally permissive — it
# matches ``def name(...):`` followed by an indented body — and is
# used by both the markdown-fence extraction and the plain-text
# extraction paths. The ``re.DOTALL`` flag lets ``.*?`` match
# newlines, so multi-line function bodies are captured correctly.
# The first capture group is the function name; the second is the
# body.
_FUNCTION_DEF_REGEX: re.Pattern[str] = re.compile(
    r"def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*(?:->[^:]+)?:\s*(?P<body>(?:\n[ \t]+[^\n]*)+)",
    re.MULTILINE,
)
"""Regex that matches a Python function definition with a multi-line indented body."""


def _extract_function_source(
    model_output: str,
    entry_point: str,
) -> str | None:
    """Extract a Python function definition from the model's response.

    The function tolerates two common model output shapes:

    * A markdown code fence (``\\`\\`\\`python ... \\`\\`\\```). The
      function content is captured verbatim, including the
      ``def`` line and the indented body.
    * Plain text — a function definition with no surrounding
      fence. The function looks for a ``def <entry_point>(...):``
      line in the output and captures from that line to the
      last indented line of the body.

    Args:
        model_output: The raw text the model returned. May be
            empty; the function returns ``None`` in that case.
        entry_point: The expected function name. The function
            returns the first matching definition whose name
            equals ``entry_point``. If the model emitted a
            function with a different name, the function still
            tries to recover by returning the first match (the
            contract does not require name verification on the
            extracted source — the test cases verify behaviour).

    Returns:
        The function source as a string (including the ``def``
        line and the body), or ``None`` when no function
        definition can be found in the model output.
    """
    if not model_output or not model_output.strip():
        return None
    # First pass: extract a python-fenced code block and search
    # for the function inside it. The fenced path is the
    # contract-pinned happy case (the prompts ask the model to
    # output a single fenced block).
    fence_pattern = re.compile(
        r"```(?:python|py)?\s*\n(?P<code>.*?)\n```",
        re.DOTALL | re.IGNORECASE,
    )
    for match in fence_pattern.finditer(model_output):
        candidate = match.group("code")
        extracted = _find_function_in_source(candidate, entry_point)
        if extracted is not None:
            return extracted
    # Second pass: search the raw output for the function. The
    # function finder handles plain-text definitions whose
    # ``def`` line is at the start of the output (or after a
    # short prose lead-in).
    extracted = _find_function_in_source(model_output, entry_point)
    if extracted is not None:
        return extracted
    return None


def _find_function_in_source(
    source: str,
    entry_point: str,
) -> str | None:
    """Return the source of the first matching function definition.

    The function searches the source for a function named
    ``entry_point`` first (the contract-pinned case), and falls
    back to the first function definition in the source when
    the name does not match. The returned string includes the
    ``def`` line and the indented body.

    Args:
        source: The text to search. May be a fenced code block's
            body or the raw model output.
        entry_point: The expected function name.

    Returns:
        The function source as a string, or ``None`` when no
        function definition is found.
    """
    if not source:
        return None
    # First pass: look for a definition whose name matches
    # ``entry_point``. The function ``_FUNCTION_DEF_REGEX``
    # captures the full ``def`` line plus the body. We use a
    # separate regex pattern here that captures the named
    # function with its name in a backreference so the
    # full-source reconstruction is automatic.
    name_pattern = re.compile(
        rf"def\s+(?P<name>{re.escape(entry_point)})\s*\([^)]*\)\s*(?:->[^:]+)?:\s*(?P<body>(?:\n[ \t]+[^\n]*)+)"
    )
    name_match = name_pattern.search(source)
    if name_match is not None:
        # Reconstruct the full source by prepending the ``def``
        # line to the body. The named-pattern regex captures
        # the body but not the ``def`` line itself, so we
        # find the start of the ``def`` line and re-emit it.
        match_start = name_match.start()
        # Walk backwards to the start of the line so the
        # returned source begins at column 0 (or at whatever
        # indentation the model emitted).
        line_start = source.rfind("\n", 0, match_start) + 1
        return source[line_start:name_match.end()]
    # Second pass: any function definition. The contract does
    # not require name verification; test cases verify the
    # function's behaviour.
    any_match = _FUNCTION_DEF_REGEX.search(source)
    if any_match is not None:
        return any_match.group(0)
    return None


def score_function_from_docstring(
    prompt: CodingPrompt, model_output: str
) -> float:
    """Score a ``function_from_docstring`` response.

    The scorer:

    1. Asserts (via :func:`isinstance`) that ``prompt`` is a
       :class:`FunctionFromDocstringPrompt`; if not, returns ``0.0``
       (the contract requires a binary result, so a wrong-category
       invocation is treated as a failure rather than raising).
    2. Calls :func:`_extract_function_source` to extract the
       function definition from ``model_output``. Returns
       ``0.0`` when the extraction fails (empty input, no
       ``def`` line, etc.).
    3. Builds a fresh, hermetic namespace with the builtins
       module (so the function can call e.g. ``len``) and
       ``exec``s the extracted source in that namespace.
       A :class:`SyntaxError` or :class:`NameError` from the
       exec is caught and reported as ``0.0``.
    4. Iterates over :attr:`FunctionFromDocstringPrompt.test_cases`,
       ``exec``ing each one in a copy of the namespace (so a
       test case that defines a local variable does not leak
       into the next case). The first failing test case
       (any unhandled exception, including ``AssertionError``)
       returns ``0.0``; the loop completing without error
       returns ``1.0``.

    Args:
        prompt: The :class:`FunctionFromDocstringPrompt` whose
            ``test_cases`` and ``entry_point`` drive the score.
        model_output: The raw text the model returned.

    Returns:
        ``1.0`` when every test case passes against the
        extracted function, ``0.0`` otherwise (including
        extraction failure, exec failure, or any test case
        raising an unhandled exception).
    """
    if not isinstance(prompt, FunctionFromDocstringPrompt):
        return 0.0
    if not model_output or not model_output.strip():
        return 0.0
    function_source = _extract_function_source(model_output, prompt.entry_point)
    if function_source is None:
        return 0.0
    # Build a fresh namespace. We seed it with ``__builtins__``
    # so the extracted function can call built-in functions
    # (``len``, ``range``, ``int``, etc.) without ``NameError``.
    # The namespace is a plain dict; the host process's
    # globals are NOT merged in. This is the sandbox: the
    # extracted function cannot reach ``os``, ``subprocess``,
    # or any other host-side module unless it ``import``s one
    # itself, and a prompt that asks for an ``import`` is
    # already a contract violation (the curated prompts do
    # not).
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(compile(function_source, "<model_output>", "exec"), namespace)
    except (SyntaxError, ValueError, TypeError, NameError):
        return 0.0
    entry = prompt.entry_point
    if entry not in namespace:
        return 0.0
    if not callable(namespace[entry]):
        return 0.0
    # Run every test case in a fresh copy of the namespace.
    # This isolates cases from each other (a test that defines
    # a local variable does not leak into the next test) but
    # preserves the extracted function.
    for case in prompt.test_cases:
        case_namespace: dict[str, Any] = dict(namespace)
        try:
            exec(compile(case, "<test_case>", "exec"), case_namespace)
        except BaseException:
            # Any failure — ``AssertionError``, ``NameError``,
            # ``TypeError``, etc. — is treated as a 0.0 score.
            # Catching ``BaseException`` is intentional: the
            # contract requires a binary result, so a test
            # case that calls ``sys.exit`` or raises
            # ``KeyboardInterrupt`` should be reported as a
            # failure, not propagate out of the scorer.
            return 0.0
    return 1.0


def score_bug_fix(prompt: CodingPrompt, model_output: str) -> float:
    """Score a ``bug_fix`` response.

    The scorer:

    1. Asserts (via :func:`isinstance`) that ``prompt`` is a
       :class:`BugFixPrompt`; if not, returns ``0.0``.
    2. Computes the :class:`difflib.SequenceMatcher` ratio
       between ``model_output`` and
       :attr:`BugFixPrompt.reference_patch`. A ratio of
       ``1.0`` means the two strings are identical; a ratio
       of ``0.0`` means they share no common substring.
    3. Returns ``1.0`` when the ratio is at least
       :data:`_BUG_FIX_SIMILARITY_THRESHOLD` (``0.9``);
       otherwise ``0.0``.

    Empty model output yields a ratio of ``0.0`` and is
    reported as ``0.0`` (the empty string shares no common
    substring with the reference patch).

    Args:
        prompt: The :class:`BugFixPrompt` whose
            ``reference_patch`` is the known-correct patched
            code.
        model_output: The raw text the model returned.

    Returns:
        ``1.0`` when the similarity ratio is at least ``0.9``,
        ``0.0`` otherwise.
    """
    if not isinstance(prompt, BugFixPrompt):
        return 0.0
    if not model_output or not model_output.strip():
        return 0.0
    reference = prompt.reference_patch
    if not reference:
        return 0.0
    matcher = difflib.SequenceMatcher(None, reference, model_output)
    ratio = matcher.ratio()
    if ratio >= _BUG_FIX_SIMILARITY_THRESHOLD:
        return 1.0
    return 0.0


def score_refactor(prompt: CodingPrompt, model_output: str) -> float:
    """Score a ``refactor`` response.

    The scorer:

    1. Asserts (via :func:`isinstance`) that ``prompt`` is a
       :class:`RefactorPrompt`; if not, returns ``0.0``.
    2. Compiles :attr:`RefactorPrompt.target_pattern` (a
       :mod:`re` pattern) and searches ``model_output`` for
       any match. A match scores ``1.0``; a miss scores
       ``0.0``.
    3. A malformed pattern (e.g. unbalanced brackets) raises
       :class:`re.error`; the scorer catches the error and
       returns ``0.0`` (a malformed pattern is a configuration
       bug, but the contract requires a binary result rather
       than an exception).

    Args:
        prompt: The :class:`RefactorPrompt` whose
            ``target_pattern`` is the regex the response
            must contain.
        model_output: The raw text the model returned.

    Returns:
        ``1.0`` when the target pattern matches anywhere in
        the model's response, ``0.0`` otherwise.
    """
    if not isinstance(prompt, RefactorPrompt):
        return 0.0
    if not model_output or not model_output.strip():
        return 0.0
    pattern = prompt.target_pattern
    if not pattern:
        return 0.0
    try:
        compiled = re.compile(pattern)
    except re.error:
        return 0.0
    if compiled.search(model_output):
        return 1.0
    return 0.0


__all__ = [
    "score_bug_fix",
    "score_function_from_docstring",
    "score_refactor",
]
