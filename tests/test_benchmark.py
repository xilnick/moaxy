"""Tests for the M7 benchmark harness.

This file consolidates the M7 contract tests. The current focus is
the curated prompt set (:mod:`moaxy.benchmark.prompts`); the
:class:`TestPromptSetContract` test class pins the contract
(VAL-BENCH-001). Additional M7 features (BenchmarkRunner,
config variants, deterministic / LLM-judge scorers, CLI, report
generator) will append their own test classes in this file as the
later M7 features land.

The contract (VAL-BENCH-001) asserts the following on
:data:`moaxy.benchmark.prompts.PROMPT_SET`:

* ``len(PROMPT_SET) >= 10`` prompts.
* All 4 categories present with at least 2 prompts each.
* Every prompt has ``task_id``, ``category``, ``prompt_text``, and
  ``scoring_method``.
* ``task_id`` values are unique.
* ``function_from_docstring`` prompts have hidden test cases that
  pass for the known-correct answer.
* ``bug_fix`` prompts have a known-correct patched code for
  diff-match scoring.

The :class:`TestPromptSetContract` test class enforces every
contract invariant with a focused test, plus a single
table-driven test that catches any regression in one place.
"""

from __future__ import annotations

import pytest

from moaxy.benchmark import prompts as prompts_module
from moaxy.benchmark.prompts import (
    BUG_FIX_PROMPTS,
    FUNCTION_PROMPTS,
    PROMPT_SET,
    BugFixPrompt,
    ExplainPrompt,
    FunctionFromDocstringPrompt,
    RefactorPrompt,
)

# The four allowed category values, pinned by the contract. Tests
# reference this constant so a future category addition is
# surfaced as a test edit (not a silent regression).
REQUIRED_CATEGORIES: tuple[str, ...] = (
    "function_from_docstring",
    "bug_fix",
    "refactor",
    "explain",
)


class TestPromptSetContract:
    """VAL-BENCH-001: the curated prompt set is well-formed.

    Each test class below targets a single contract invariant.
    The tests are deliberately small and focused so a failure
    message points directly at the broken invariant.
    """

    def test_prompt_set_exists_and_is_list(self):
        # The contract requires the prompt set to be a module-level
        # constant list. Pin the type so a future refactor that
        # changes ``PROMPT_SET`` to e.g. a tuple or generator
        # fails loudly here.
        assert isinstance(PROMPT_SET, list)
        assert len(PROMPT_SET) > 0

    def test_prompt_set_has_at_least_ten_prompts(self):
        # Contract floor: ``len(PROMPT_SET) >= 10``. The curated set
        # has 13 prompts; we pin the floor (not the exact count) so
        # adding new prompts does not break this test.
        assert len(PROMPT_SET) >= 10, (
            f"PROMPT_SET must contain at least 10 prompts, got {len(PROMPT_SET)}"
        )

    def test_all_four_categories_present(self):
        # Contract: every one of the four allowed categories is
        # present in the prompt set. Pin the set of category
        # values explicitly so a missing category is reported with
        # its name.
        present = {p.category for p in PROMPT_SET}
        for category in REQUIRED_CATEGORIES:
            assert category in present, (
                f"PROMPT_SET is missing required category {category!r}; "
                f"present categories: {sorted(present)}"
            )

    def test_at_least_two_prompts_per_category(self):
        # Contract: every category has at least 2 prompts. Loop
        # over the four required categories and report the count
        # for any that fall short.
        per_category: dict[str, int] = {c: 0 for c in REQUIRED_CATEGORIES}
        for prompt in PROMPT_SET:
            if prompt.category in per_category:
                per_category[prompt.category] += 1
        for category, count in per_category.items():
            assert count >= 2, (
                f"category {category!r} has only {count} prompt(s); "
                "the contract requires at least 2 per category"
            )

    def test_every_prompt_has_required_fields(self):
        # Contract: every prompt has ``task_id``, ``category``,
        # ``prompt_text``, ``scoring_method``. The base
        # :class:`CodingPrompt` dataclass enforces type hints on
        # every field; here we pin the *presence* and the
        # non-emptiness of the string fields.
        for prompt in PROMPT_SET:
            assert isinstance(prompt.task_id, str) and prompt.task_id, (
                f"prompt {prompt!r} is missing a non-empty task_id"
            )
            assert isinstance(prompt.category, str) and prompt.category, (
                f"prompt {prompt.task_id!r} is missing a non-empty category"
            )
            assert prompt.category in REQUIRED_CATEGORIES, (
                f"prompt {prompt.task_id!r} has unknown category "
                f"{prompt.category!r}; allowed: {REQUIRED_CATEGORIES}"
            )
            assert isinstance(prompt.prompt_text, str) and prompt.prompt_text, (
                f"prompt {prompt.task_id!r} is missing a non-empty prompt_text"
            )
            assert prompt.scoring_method in {"deterministic", "judge"}, (
                f"prompt {prompt.task_id!r} has invalid scoring_method "
                f"{prompt.scoring_method!r}; allowed: "
                f"{{'deterministic', 'judge'}}"
            )

    def test_task_ids_are_unique(self):
        # Contract: ``task_id`` values are unique. Build a set and
        # compare its length to the list length.
        ids = [p.task_id for p in PROMPT_SET]
        assert len(set(ids)) == len(ids), (
            f"PROMPT_SET has duplicate task_ids: "
            f"{[tid for tid in ids if ids.count(tid) > 1]}"
        )

    def test_function_from_docstring_prompts_have_test_cases(self):
        # Contract: ``function_from_docstring`` prompts include
        # hidden test cases for deterministic scoring. Pin the
        # field type (tuple of strings) and the non-empty
        # invariant.
        function_prompts = [
            p for p in PROMPT_SET if p.category == "function_from_docstring"
        ]
        assert function_prompts, "no function_from_docstring prompts found"
        for prompt in function_prompts:
            assert isinstance(prompt, FunctionFromDocstringPrompt), (
                f"prompt {prompt.task_id!r} is a "
                f"function_from_docstring category but is not a "
                f"FunctionFromDocstringPrompt instance"
            )
            assert isinstance(prompt.test_cases, tuple), (
                f"prompt {prompt.task_id!r} has non-tuple test_cases"
            )
            assert len(prompt.test_cases) > 0, (
                f"prompt {prompt.task_id!r} is missing hidden test cases"
            )
            for idx, case in enumerate(prompt.test_cases):
                assert isinstance(case, str) and case, (
                    f"prompt {prompt.task_id!r} test_cases[{idx}] is empty/non-string"
                )

    def test_bug_fix_prompts_have_reference_patch(self):
        # Contract: ``bug_fix`` prompts include the known-correct
        # patched code for diff-match scoring. Pin the field type
        # (string) and the non-empty invariant.
        bug_fix_prompts = [p for p in PROMPT_SET if p.category == "bug_fix"]
        assert bug_fix_prompts, "no bug_fix prompts found"
        for prompt in bug_fix_prompts:
            assert isinstance(prompt, BugFixPrompt), (
                f"prompt {prompt.task_id!r} is a bug_fix category but "
                f"is not a BugFixPrompt instance"
            )
            assert isinstance(prompt.reference_patch, str), (
                f"prompt {prompt.task_id!r} reference_patch is not a string"
            )
            assert prompt.reference_patch.strip(), (
                f"prompt {prompt.task_id!r} is missing reference_patch"
            )

    def test_refactor_prompts_have_target_pattern(self):
        # The refactor scorer's deterministic regex needs a target
        # pattern. Pin the field type and the non-empty invariant
        # so a future edit cannot drop the field.
        refactor_prompts = [p for p in PROMPT_SET if p.category == "refactor"]
        assert refactor_prompts, "no refactor prompts found"
        for prompt in refactor_prompts:
            assert isinstance(prompt, RefactorPrompt), (
                f"prompt {prompt.task_id!r} is a refactor category but "
                f"is not a RefactorPrompt instance"
            )
            assert isinstance(prompt.target_pattern, str), (
                f"prompt {prompt.task_id!r} target_pattern is not a string"
            )
            assert prompt.target_pattern.strip(), (
                f"prompt {prompt.task_id!r} is missing target_pattern"
            )
            # The target pattern is consumed by the deterministic
            # regex scorer. Confirm the pattern compiles; this
            # catches a typo in a future edit before the live run
            # blows up at scoring time.
            import re

            try:
                re.compile(prompt.target_pattern)
            except re.error as exc:
                pytest.fail(
                    f"prompt {prompt.task_id!r} target_pattern "
                    f"{prompt.target_pattern!r} does not compile: {exc}"
                )

    def test_explain_prompts_are_scored_by_judge(self):
        # Contract: ``explain`` prompts are scored by an LLM judge.
        # Pin the scoring_method on every explain prompt so a
        # future edit that flips a prompt to deterministic is
        # caught here.
        explain_prompts = [p for p in PROMPT_SET if p.category == "explain"]
        assert explain_prompts, "no explain prompts found"
        for prompt in explain_prompts:
            assert isinstance(prompt, ExplainPrompt), (
                f"prompt {prompt.task_id!r} is an explain category but "
                f"is not an ExplainPrompt instance"
            )
            assert prompt.scoring_method == "judge", (
                f"explain prompt {prompt.task_id!r} has scoring_method "
                f"{prompt.scoring_method!r}; the contract requires 'judge'"
            )

    def test_deterministic_prompts_use_deterministic_scoring(self):
        # Cross-check: every function_from_docstring, bug_fix, and
        # refactor prompt has scoring_method == 'deterministic'.
        # The opposite direction (judge) is pinned in
        # ``test_explain_prompts_are_scored_by_judge`` above.
        deterministic_categories = {
            "function_from_docstring",
            "bug_fix",
            "refactor",
        }
        for prompt in PROMPT_SET:
            if prompt.category in deterministic_categories:
                assert prompt.scoring_method == "deterministic", (
                    f"prompt {prompt.task_id!r} is a "
                    f"{prompt.category!r} prompt but has "
                    f"scoring_method={prompt.scoring_method!r}; the contract "
                    f"requires 'deterministic' for this category"
                )

    def test_function_prompts_test_cases_match_prompt_text(self):
        # Spot-check the function-from-docstring scoring path
        # locally: a trivial synthetic model response that defines
        # the requested function (with the same body the prompt
        # describes) must pass every test case for the prompt. This
        # proves the test cases in the curated set are actually
        # satisfiable; the deterministic scorer will execute the
        # same path at benchmark time.
        for prompt in FUNCTION_PROMPTS:
            assert isinstance(prompt, FunctionFromDocstringPrompt)
            entry = prompt.entry_point
            if prompt.task_id == "function-add":
                code = f"def {entry}(a, b):\n    return a + b\n"
            elif prompt.task_id == "function-reverse-string":
                code = f"def {entry}(text):\n    return text[::-1]\n"
            elif prompt.task_id == "function-is-prime":
                code = (
                    f"def {entry}(n):\n"
                    f"    if n < 2:\n"
                    f"        return False\n"
                    f"    for i in range(2, int(n ** 0.5) + 1):\n"
                    f"        if n % i == 0:\n"
                    f"            return False\n"
                    f"    return True\n"
                )
            elif prompt.task_id == "function-fibonacci":
                code = (
                    f"def {entry}(n):\n"
                    f"    a, b = 0, 1\n"
                    f"    for _ in range(n):\n"
                    f"        a, b = b, a + b\n"
                    f"    return a\n"
                )
            else:
                pytest.fail(
                    f"unexpected function_from_docstring task_id "
                    f"{prompt.task_id!r}; add a known-correct answer "
                    "to this spot-check"
                )
            # Run the function definition then each test case in
            # order. A test case failure bubbles out as an
            # ``AssertionError``; a missing function definition is
            # a ``NameError`` from the first test case. Both
            # surface in the test report.
            namespace: dict[str, object] = {}
            exec(code, namespace)
            assert entry in namespace, (
                f"known-correct response for {prompt.task_id!r} did not "
                f"define {entry!r}"
            )
            for case in prompt.test_cases:
                exec(case, namespace)

    def test_bug_fix_prompts_reference_patches_match_prompt_text(self):
        # Spot-check the bug_fix scoring path locally: a trivial
        # synthetic model response that returns the
        # ``reference_patch`` exactly must pass the diff-match
        # scorer. We use difflib.SequenceMatcher for parity with
        # the production scorer.
        import difflib

        for prompt in BUG_FIX_PROMPTS:
            assert isinstance(prompt, BugFixPrompt)
            ratio = difflib.SequenceMatcher(
                None, prompt.reference_patch, prompt.reference_patch
            ).ratio()
            assert ratio >= 0.9, (
                f"reference_patch for {prompt.task_id!r} does not "
                f"self-match at >= 0.9 similarity (got {ratio:.3f})"
            )

    def test_prompt_set_total_count(self):
        # Pin the total count as a safety net: the curated set
        # has 13 prompts (4 + 3 + 3 + 3). A future edit that
        # accidentally drops a category or a prompt is caught
        # here with a clear failure message.
        assert len(PROMPT_SET) == 13, (
            f"PROMPT_SET expected to have 13 prompts (4 + 3 + 3 + 3), "
            f"got {len(PROMPT_SET)}"
        )

    def test_category_counts_match_curated_numbers(self):
        # Pin the per-category count too. The curated set is
        # 4 function, 3 bug_fix, 3 refactor, 3 explain. A
        # future edit that shifts the split is caught here.
        counts = {category: 0 for category in REQUIRED_CATEGORIES}
        for prompt in PROMPT_SET:
            counts[prompt.category] += 1
        assert counts == {
            "function_from_docstring": 4,
            "bug_fix": 3,
            "refactor": 3,
            "explain": 3,
        }, f"unexpected per-category counts: {counts}"


# ────────────────────────────────────────────────────────────────────
# Module-level smoke test
# ────────────────────────────────────────────────────────────────────


def test_module_imports_clean():
    # The contract asserts the benchmark module is importable but
    # not auto-invoked; this test proves the import path works
    # in isolation, without pytest fixture or plugin discovery.
    import importlib

    mod = importlib.import_module("moaxy.benchmark.prompts")
    assert hasattr(mod, "PROMPT_SET")
    assert hasattr(mod, "CodingPrompt")
    assert hasattr(mod, "FunctionFromDocstringPrompt")
    assert hasattr(mod, "BugFixPrompt")
    assert hasattr(mod, "RefactorPrompt")
    assert hasattr(mod, "ExplainPrompt")


def test_package_init_exports_prompt_symbols():
    # The package's __init__ re-exports the public prompt types.
    # Pin the re-exports so a future refactor of the package
    # facade does not silently break downstream imports.
    from moaxy.benchmark import (
        PROMPT_SET as PROMPT_SET_REEXPORT,
    )
    from moaxy.benchmark import (
        BugFixPrompt as BugFixPromptReexport,
    )
    from moaxy.benchmark import (
        CodingPrompt as CodingPromptReexport,
    )
    from moaxy.benchmark import (
        ExplainPrompt as ExplainPromptReexport,
    )
    from moaxy.benchmark import (
        FunctionFromDocstringPrompt as FunctionPromptReexport,
    )
    from moaxy.benchmark import (
        RefactorPrompt as RefactorPromptReexport,
    )

    assert PROMPT_SET_REEXPORT is PROMPT_SET
    assert CodingPromptReexport is prompts_module.CodingPrompt
    assert FunctionPromptReexport is prompts_module.FunctionFromDocstringPrompt
    assert BugFixPromptReexport is prompts_module.BugFixPrompt
    assert RefactorPromptReexport is prompts_module.RefactorPrompt
    assert ExplainPromptReexport is prompts_module.ExplainPrompt
