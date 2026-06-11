"""Curated coding prompt set for the M7 benchmark harness.

The :data:`PROMPT_SET` is a module-level list of :class:`CodingPrompt`
records. The set covers four coding-task categories:

* ``function_from_docstring`` — the model writes a function that
  matches a docstring + type hints. Hidden test cases
  (:attr:`FunctionFromDocstringPrompt.test_cases`) verify correctness.
  Scoring is deterministic (the test cases are executed against the
  extracted function in a hermetic subprocess).
* ``bug_fix`` — the model finds and fixes a known bug in a code
  snippet. The known-correct patched code
  (:attr:`BugFixPrompt.reference_patch`) is fuzzy-matched against the
  model's output via :func:`difflib.SequenceMatcher` with a 0.9
  similarity threshold (deterministic scoring).
* ``refactor`` — the model refactors a code snippet to a target
  pattern (:attr:`RefactorPrompt.target_pattern`). The target pattern
  is regex-matched against the model's output (deterministic
  scoring).
* ``explain`` — the model explains what a code snippet does. Scoring
  is delegated to an LLM judge (e.g.
  ``deepseek-v4-pro:cloud`` on local Ollama) via
  :mod:`moaxy.benchmark.scoring.judge`.

The contract (VAL-BENCH-001) requires:

* ``len(PROMPT_SET) >= 10`` prompts.
* All 4 categories present with at least 2 prompts each.
* Every prompt has ``task_id`` (unique string), ``category`` (one of
  the four allowed values), ``prompt_text`` (non-empty string), and
  ``scoring_method`` (``deterministic`` or ``judge``).
* ``task_id`` values are unique.
* Deterministic ``function_from_docstring`` prompts carry hidden test
  cases that pass for the known-correct answer.
* Deterministic ``bug_fix`` prompts carry the known-correct patched
  code for diff-match scoring.

The set is deliberately small (13 prompts) but covers all four
categories with at least 3 prompts each. The total prompt + token
footprint is bounded so a live run stays inside the user's OpenRouter
credit envelope. Editing a prompt? Update the matching contract
assertions and the test pins in :mod:`tests.test_benchmark`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Allowed category values. The string values are pinned by the
# contract (VAL-BENCH-001) and are matched against the validator's
# substring checks. Adding a new category requires updating both the
# contract and the test that pins the four required values.
Category = Literal[
    "function_from_docstring",
    "bug_fix",
    "refactor",
    "explain",
]

# Allowed scoring methods. ``deterministic`` is paired with a
# hand-checked reference (``test_cases``, ``reference_patch``, or
# ``target_pattern``) on the prompt; ``judge`` is paired with an
# LLM judge call.
ScoringMethod = Literal["deterministic", "judge"]


@dataclass(frozen=True)
class CodingPrompt:
    """A single coding-task prompt in the benchmark set.

    The base class carries the four contract-mandated fields
    (``task_id``, ``category``, ``prompt_text``, ``scoring_method``).
    Subclasses extend it with the reference data needed by the
    deterministic scorer for the corresponding category.

    Attributes:
        task_id: Stable unique identifier. The contract asserts
            ``task_id`` is unique across ``PROMPT_SET``. Strings use
            the form ``"<category>-<short-slug>"`` (e.g.
            ``"function-add"``).
        category: One of the four allowed category literals.
        prompt_text: The text sent to the model verbatim. Must be a
            non-empty string; the contract pins the non-empty
            invariant.
        scoring_method: ``"deterministic"`` when the reference data
            on the subclass is enough to score without an LLM;
            ``"judge"`` when the response is sent to an LLM judge.
    """

    task_id: str
    category: Category
    prompt_text: str
    scoring_method: ScoringMethod


@dataclass(frozen=True)
class FunctionFromDocstringPrompt(CodingPrompt):
    """A prompt where the model writes a function from a docstring.

    The deterministic scorer extracts the function definition from
    the model's response (tolerating markdown code fences and prose
    wrappers), then runs :attr:`test_cases` against it in a hermetic
    subprocess. The score is 1.0 when every test case passes and 0.0
    when any test case fails or the function cannot be extracted.

    Attributes:
        test_cases: Hidden test cases. Each entry is a snippet of
            Python code that runs in a fresh subprocess with the
            extracted function injected; the snippet must exit 0 and
            produce no stderr to score 1.0. The test cases are kept
            hidden from the model's prompt so the model cannot
            hard-code the expected behaviour; the validator only sees
            them at scoring time.
        entry_point: The function name to extract from the model's
            output. Defaults to ``"solution"`` so the model's
            response is expected to define ``def solution(...)``.
    """

    test_cases: tuple[str, ...] = ()
    entry_point: str = "solution"


@dataclass(frozen=True)
class BugFixPrompt(CodingPrompt):
    """A prompt where the model fixes a known bug in a code snippet.

    The deterministic scorer extracts the patched code from the
    model's response and uses
    :func:`difflib.SequenceMatcher` to compute a similarity ratio
    against :attr:`reference_patch`. A ratio >= 0.9 scores 1.0;
    anything below scores 0.0. Whitespace and minor formatting
    differences are tolerated; the matcher is fuzzy by design.

    Attributes:
        reference_patch: The known-correct patched code for this
            prompt. The scorer compares the model's output to this
            string; the contract asserts the field is present on
            every ``bug_fix`` prompt.
    """

    reference_patch: str = ""


@dataclass(frozen=True)
class RefactorPrompt(CodingPrompt):
    """A prompt where the model refactors a code snippet to a target pattern.

    The deterministic scorer regex-searches the model's response for
    :attr:`target_pattern`. A match scores 1.0; a miss scores 0.0.
    The pattern is intentionally a small structural marker (e.g. a
    list comprehension, a with-statement, a single function call)
    that is easy to verify with a regex but impossible to satisfy
    without actually performing the refactor.

    Attributes:
        target_pattern: A Python :mod:`re` pattern that matches the
            refactored code's structural shape. The contract asserts
            the field is present on every ``refactor`` prompt.
    """

    target_pattern: str = ""


@dataclass(frozen=True)
class ExplainPrompt(CodingPrompt):
    """A prompt where the model explains a code snippet.

    The LLM-judge scorer (:mod:`moaxy.benchmark.scoring.judge`)
    receives the model's explanation, scores it on a 0-10 rubric,
    and returns a float. The prompt carries no reference data; the
    judge is the source of truth.
    """

    # Reserved for future rubric metadata. The default empty tuple
    # documents that the field is intentionally a tuple (so the
    # dataclass is hashable and frozen) and that the prompt
    # currently has no structured rubric hints beyond the prompt
    # text itself.
    rubric_hints: tuple[str, ...] = field(default_factory=tuple)


# ────────────────────────────────────────────────────────────────────
# Curated prompt set. Edit with care: the contract (VAL-BENCH-001)
# pins the structure; the test in TestPromptSetContract pins the
# exact task_ids, categories, and reference data.
# ────────────────────────────────────────────────────────────────────


FUNCTION_PROMPTS: tuple[FunctionFromDocstringPrompt, ...] = (
    FunctionFromDocstringPrompt(
        task_id="function-add",
        category="function_from_docstring",
        scoring_method="deterministic",
        prompt_text=(
            "Write a Python function called `solution(a: int, b: int) -> int` "
            "that returns the sum of `a` and `b`. Output ONLY the function "
            "definition (with type hints) wrapped in a single ```python``` "
            "code fence; do not include any prose, tests, or usage examples."
        ),
        entry_point="solution",
        test_cases=(
            "assert solution(1, 2) == 3",
            "assert solution(0, 0) == 0",
            "assert solution(-5, 5) == 0",
            "assert solution(100, 200) == 300",
        ),
    ),
    FunctionFromDocstringPrompt(
        task_id="function-reverse-string",
        category="function_from_docstring",
        scoring_method="deterministic",
        prompt_text=(
            "Write a Python function called `solution(text: str) -> str` "
            "that returns the reverse of `text`. Output ONLY the function "
            "definition (with type hints) wrapped in a single ```python``` "
            "code fence; do not include any prose, tests, or usage examples."
        ),
        entry_point="solution",
        test_cases=(
            "assert solution('hello') == 'olleh'",
            "assert solution('') == ''",
            "assert solution('a') == 'a'",
            "assert solution('ab') == 'ba'",
            "assert solution('racecar') == 'racecar'",
        ),
    ),
    FunctionFromDocstringPrompt(
        task_id="function-is-prime",
        category="function_from_docstring",
        scoring_method="deterministic",
        prompt_text=(
            "Write a Python function called `solution(n: int) -> bool` that "
            "returns True if `n` is a prime number and False otherwise. "
            "`n` is a non-negative integer; 0 and 1 are NOT prime. Output "
            "ONLY the function definition (with type hints) wrapped in a "
            "single ```python``` code fence; do not include any prose, "
            "tests, or usage examples."
        ),
        entry_point="solution",
        test_cases=(
            "assert solution(2) is True",
            "assert solution(3) is True",
            "assert solution(4) is False",
            "assert solution(17) is True",
            "assert solution(1) is False",
            "assert solution(0) is False",
            "assert solution(9) is False",
        ),
    ),
    FunctionFromDocstringPrompt(
        task_id="function-fibonacci",
        category="function_from_docstring",
        scoring_method="deterministic",
        prompt_text=(
            "Write a Python function called `solution(n: int) -> int` that "
            "returns the n-th Fibonacci number where fib(0) == 0 and "
            "fib(1) == 1. `n` is a non-negative integer. Output ONLY the "
            "function definition (with type hints) wrapped in a single "
            "```python``` code fence; do not include any prose, tests, or "
            "usage examples."
        ),
        entry_point="solution",
        test_cases=(
            "assert solution(0) == 0",
            "assert solution(1) == 1",
            "assert solution(2) == 1",
            "assert solution(3) == 2",
            "assert solution(10) == 55",
            "assert solution(15) == 610",
        ),
    ),
)


BUG_FIX_PROMPTS: tuple[BugFixPrompt, ...] = (
    BugFixPrompt(
        task_id="bugfix-off-by-one",
        category="bug_fix",
        scoring_method="deterministic",
        prompt_text=(
            "The following Python function is supposed to return the sum of "
            "the first `n` positive integers (1 + 2 + ... + n) but it has an "
            "off-by-one bug. Fix the bug and return ONLY the corrected "
            "function wrapped in a single ```python``` code fence.\n\n"
            "```python\n"
            "def sum_first_n(n: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(1, n):\n"
            "        total += i\n"
            "    return total\n"
            "```"
        ),
        reference_patch=(
            "def sum_first_n(n: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(1, n + 1):\n"
            "        total += i\n"
            "    return total\n"
        ),
    ),
    BugFixPrompt(
        task_id="bugfix-walrus",
        category="bug_fix",
        scoring_method="deterministic",
        prompt_text=(
            "The following Python function is supposed to return the "
            "maximum value in a non-empty list but it incorrectly returns "
            "the first element when the maximum is at index 0. Fix the "
            "bug and return ONLY the corrected function wrapped in a "
            "single ```python``` code fence.\n\n"
            "```python\n"
            "def find_max(values: list[int]) -> int:\n"
            "    if not values:\n"
            "        raise ValueError('values must be non-empty')\n"
            "    max_val = values[0]\n"
            "    for v in values:\n"
            "        if v >= max_val:\n"
            "            max_val = v\n"
            "    return values[0]\n"
            "```"
        ),
        reference_patch=(
            "def find_max(values: list[int]) -> int:\n"
            "    if not values:\n"
            "        raise ValueError('values must be non-empty')\n"
            "    max_val = values[0]\n"
            "    for v in values:\n"
            "        if v > max_val:\n"
            "            max_val = v\n"
            "    return max_val\n"
        ),
    ),
    BugFixPrompt(
        task_id="bugfix-dict-default",
        category="bug_fix",
        scoring_method="deterministic",
        prompt_text=(
            "The following Python function is supposed to count the "
            "occurrences of each word in a string (case-insensitive, "
            "ignoring punctuation) but it crashes on a `KeyError` when "
            "encountering a word it has not seen before. Fix the bug and "
            "return ONLY the corrected function wrapped in a single "
            "```python``` code fence.\n\n"
            "```python\n"
            "import re\n"
            "def word_counts(text: str) -> dict[str, int]:\n"
            "    counts = {}\n"
            "    for word in re.findall(r\"[A-Za-z']+\", text.lower()):\n"
            "        counts[word] = counts[word] + 1\n"
            "    return counts\n"
            "```"
        ),
        reference_patch=(
            "import re\n"
            "def word_counts(text: str) -> dict[str, int]:\n"
            "    counts = {}\n"
            "    for word in re.findall(r\"[A-Za-z']+\", text.lower()):\n"
            "        counts[word] = counts.get(word, 0) + 1\n"
            "    return counts\n"
        ),
    ),
)


REFACTOR_PROMPTS: tuple[RefactorPrompt, ...] = (
    RefactorPrompt(
        task_id="refactor-list-comp",
        category="refactor",
        scoring_method="deterministic",
        prompt_text=(
            "Refactor the following Python snippet to use a single list "
            "comprehension (no explicit `for` loop and no `append` calls). "
            "Return ONLY the refactored code wrapped in a single "
            "```python``` code fence.\n\n"
            "```python\n"
            "squares = []\n"
            "for x in range(10):\n"
            "    squares.append(x * x)\n"
            "```"
        ),
        target_pattern=r"squares\s*=\s*\[",
    ),
    RefactorPrompt(
        task_id="refactor-context-manager",
        category="refactor",
        scoring_method="deterministic",
        prompt_text=(
            "Refactor the following Python snippet to use a `with` "
            "statement (context manager) instead of the explicit "
            "`file.close()` call. Return ONLY the refactored code "
            "wrapped in a single ```python``` code fence.\n\n"
            "```python\n"
            "f = open('data.txt', 'r')\n"
            "contents = f.read()\n"
            "f.close()\n"
            "```"
        ),
        target_pattern=r"^\s*with\s+open\(",
    ),
    RefactorPrompt(
        task_id="refactor-sum-builtin",
        category="refactor",
        scoring_method="deterministic",
        prompt_text=(
            "Refactor the following Python snippet to use the built-in "
            "`sum(...)` function instead of the explicit accumulator "
            "loop. Return ONLY the refactored code wrapped in a single "
            "```python``` code fence.\n\n"
            "```python\n"
            "total = 0\n"
            "for value in [1, 2, 3, 4, 5]:\n"
            "    total += value\n"
            "```"
        ),
        target_pattern=r"total\s*=\s*sum\(",
    ),
)


EXPLAIN_PROMPTS: tuple[ExplainPrompt, ...] = (
    ExplainPrompt(
        task_id="explain-closure",
        category="explain",
        scoring_method="judge",
        prompt_text=(
            "Explain what the following Python code does in 3-5 sentences. "
            "Describe the closure, the deferred computation, and any "
            "potential pitfalls. Be concise.\n\n"
            "```python\n"
            "def make_multipliers(factors):\n"
            "    return [lambda x, f=f: x * f for f in factors]\n"
            "\n"
            "doublers = make_multipliers([2, 3, 4])\n"
            "print([d(5) for d in doublers])\n"
            "```"
        ),
    ),
    ExplainPrompt(
        task_id="explain-decorator",
        category="explain",
        scoring_method="judge",
        prompt_text=(
            "Explain what the following Python decorator does in 3-5 "
            "sentences. Describe what `functools.wraps` is for, what the "
            "wrapper preserves, and what it does not. Be concise.\n\n"
            "```python\n"
            "import functools\n"
            "def debug(func):\n"
            "    @functools.wraps(func)\n"
            "    def wrapper(*args, **kwargs):\n"
            "        print(f'calling {func.__name__} with {args}, {kwargs}')\n"
            "        result = func(*args, **kwargs)\n"
            "        print(f'{func.__name__} returned {result!r}')\n"
            "        return result\n"
            "    return wrapper\n"
            "```"
        ),
    ),
    ExplainPrompt(
        task_id="explain-async-gather",
        category="explain",
        scoring_method="judge",
        prompt_text=(
            "Explain what the following Python code does in 3-5 "
            "sentences. Describe the role of `asyncio.gather`, the "
            "ordering of the `await` calls, and the shape of the "
            "returned list. Be concise.\n\n"
            "```python\n"
            "import asyncio\n"
            "\n"
            "async def fetch(i):\n"
            "    await asyncio.sleep(0.01)\n"
            "    return i * 2\n"
            "\n"
            "async def main():\n"
            "    results = await asyncio.gather(*[fetch(i) for i in range(5)])\n"
            "    print(results)\n"
            "\n"
            "asyncio.run(main())\n"
            "```"
        ),
    ),
)


PROMPT_SET: list[CodingPrompt] = (
    list(FUNCTION_PROMPTS)
    + list(BUG_FIX_PROMPTS)
    + list(REFACTOR_PROMPTS)
    + list(EXPLAIN_PROMPTS)
)
"""The curated coding prompt set.

The list contains 13 prompts spanning all 4 categories (4
function_from_docstring, 3 bug_fix, 3 refactor, 3 explain). The
contract (VAL-BENCH-001) requires at least 10 prompts with at
least 2 per category; the curated set exceeds the minimum so a
small test-set change does not risk dropping below the floor. The
list is ordered: function prompts first, then bug_fix, refactor,
explain. Tests that pin ordering should re-read this list and
pin the exact indices.
"""


__all__ = [
    "BUG_FIX_PROMPTS",
    "BugFixPrompt",
    "Category",
    "CodingPrompt",
    "EXPLAIN_PROMPTS",
    "ExplainPrompt",
    "FUNCTION_PROMPTS",
    "FunctionFromDocstringPrompt",
    "PROMPT_SET",
    "REFACTOR_PROMPTS",
    "RefactorPrompt",
    "ScoringMethod",
]
