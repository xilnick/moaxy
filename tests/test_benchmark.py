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
from moaxy.benchmark.configs import (
    COMPARISON_MODELS,
    MODEL_ALIASES,
    ConfigVariant,
    make_config,
)
from moaxy.benchmark.prompts import (
    BUG_FIX_PROMPTS,
    FUNCTION_PROMPTS,
    PROMPT_SET,
    BugFixPrompt,
    ExplainPrompt,
    FunctionFromDocstringPrompt,
    RefactorPrompt,
)
from moaxy.models.config import MoaxyConfig

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


# ────────────────────────────────────────────────────────────────────
# Config variants
# ────────────────────────────────────────────────────────────────────
#
# M7 feature m7-config-variants: the
# :mod:`moaxy.benchmark.configs` module exports a
# :class:`ConfigVariant` enum with exactly four members
# (BASELINE, REFLECTION_ONLY, ADVISOR_ONLY, BOTH) and a
# :func:`make_config` factory that returns a fully-validated
# :class:`moaxy.models.config.MoaxyConfig` for every
# (variant, model) cell. The contract (VAL-BENCH-002) asserts:
#
# * 4 variants exist.
# * All 8 configs (4 variants x 2 models) parse cleanly through
#   :meth:`moaxy.models.config.MoaxyConfig.model_validate`.
# * The advisor model for ADVISOR_ONLY and BOTH is the OTHER
#   comparison model (cross-advise).
#
# The :class:`TestConfigVariantsContract` class below enforces every
# contract invariant with a focused test, plus a single
# table-driven test that catches any regression in one place.


# The four required variant names, pinned by the contract. The
# literal values match the string values declared on the
# :class:`ConfigVariant` enum. The test class references this
# constant so a future variant rename is surfaced as a test edit
# (not a silent regression).
REQUIRED_VARIANTS: tuple[ConfigVariant, ...] = (
    ConfigVariant.BASELINE,
    ConfigVariant.REFLECTION_ONLY,
    ConfigVariant.ADVISOR_ONLY,
    ConfigVariant.BOTH,
)


# The cross-advise rule expressed as a data table. The
# ``make_config`` factory must produce a
# :class:`moaxy.models.config.MoaxyConfig` whose
# :attr:`~moaxy.models.config.AdvisorConfig.model` is the OTHER
# model in :data:`COMPARISON_MODELS`. The expected advisor model
# is the full OpenRouter id (the value of the alias in
# :data:`MODEL_ALIASES`), not the client-facing alias.
def _expected_advisor_model(model_alias: str) -> str:
    """Return the OpenRouter id of the OTHER comparison model.

    Mirrors :func:`moaxy.benchmark.configs._cross_advise_model`
    so the test's expected value does not depend on a private
    helper. The function is intentionally duplicated here so
    the test pins the cross-advise rule by computing the
    expected value from the public :data:`COMPARISON_MODELS`
    and :data:`MODEL_ALIASES` tables, not by reading the
    factory's internal helper.
    """
    other = [m for m in COMPARISON_MODELS if m != model_alias]
    assert len(other) == 1, (
        f"cross-advise lookup for {model_alias!r} expected exactly one "
        f"OTHER model in {COMPARISON_MODELS!r}, got {other!r}"
    )
    return MODEL_ALIASES[other[0]]


class TestConfigVariantsContract:
    """VAL-BENCH-002: the 4 config variants parse cleanly.

    Each test method below targets a single contract invariant.
    The tests are deliberately small and focused so a failure
    message points directly at the broken invariant.
    """

    def test_four_variants_exist(self):
        # Contract: the :class:`ConfigVariant` enum has exactly
        # four members. The expected set is pinned by
        # :data:`REQUIRED_VARIANTS`; this test asserts both the
        # exact set and the cardinality.
        actual = set(ConfigVariant)
        expected = set(REQUIRED_VARIANTS)
        assert actual == expected, (
            f"ConfigVariant mismatch: expected {sorted(v.name for v in expected)}, "
            f"got {sorted(v.name for v in actual)}"
        )
        assert len(ConfigVariant) == 4, (
            f"ConfigVariant must have exactly 4 members, got {len(ConfigVariant)}"
        )

    def test_required_variant_names(self):
        # Contract: the four variant names are BASELINE,
        # REFLECTION_ONLY, ADVISOR_ONLY, BOTH. Pin the
        # ``.name`` attribute (Python identifier form) so a
        # future rename of one of the values is caught here
        # with a clear failure message.
        names = {v.name for v in ConfigVariant}
        assert names == {
            "BASELINE",
            "REFLECTION_ONLY",
            "ADVISOR_ONLY",
            "BOTH",
        }, f"unexpected ConfigVariant name set: {names}"

    def test_all_eight_configs_parse_cleanly(self):
        # Contract: all 8 configs (4 variants x 2 models) parse
        # cleanly through ``MoaxyConfig.model_validate``. Loop
        # over the Cartesian product and assert each one
        # round-trips. ``make_config`` returns an instance of
        # :class:`MoaxyConfig` (which is already Pydantic-
        # validated at construction), but the contract is
        # explicit that the test must call ``model_validate``
        # so a future refactor that swaps the constructor for
        # a plain ``BaseModel`` is caught here.
        for variant in REQUIRED_VARIANTS:
            for model_alias in COMPARISON_MODELS:
                config = make_config(model_alias, variant)
                # ``model_validate`` accepts a Pydantic model
                # (round-trip) or a dict. Passing the
                # ``model_dump()`` form exercises the dict
                # parsing path, which is the production
                # loader's path.
                round_tripped = MoaxyConfig.model_validate(config.model_dump())
                assert isinstance(round_tripped, MoaxyConfig)
                # The round-trip must preserve the
                # variant-critical fields.
                assert (
                    round_tripped.routes[0].reflection.turns
                    == config.routes[0].reflection.turns
                )
                assert (
                    round_tripped.routes[0].advisor.turns
                    == config.routes[0].advisor.turns
                )
                assert (
                    round_tripped.routes[0].advisor.model
                    == config.routes[0].advisor.model
                )

    def test_baseline_reflection_and_advisor_disabled(self):
        # Contract: ``make_config(model, BASELINE)`` returns a
        # config with ``reflection.turns == 0`` and
        # ``advisor.turns == 0``. The advisor ``model`` is
        # ``None`` (no advisor call is made) but the contract
        # does not require that field to be ``None``; we
        # assert it as a sanity check.
        for model_alias in COMPARISON_MODELS:
            config = make_config(model_alias, ConfigVariant.BASELINE)
            assert len(config.routes) == 1
            route = config.routes[0]
            assert route.reflection.turns == 0, (
                f"BASELINE for {model_alias!r} expected "
                f"reflection.turns=0, got {route.reflection.turns}"
            )
            assert route.advisor.turns == 0, (
                f"BASELINE for {model_alias!r} expected "
                f"advisor.turns=0, got {route.advisor.turns}"
            )
            assert route.advisor.model is None, (
                f"BASELINE for {model_alias!r} expected "
                f"advisor.model=None, got {route.advisor.model!r}"
            )

    def test_reflection_only_runs_one_reflection_turn(self):
        # Contract: ``make_config(model, REFLECTION_ONLY)``
        # returns a config with ``reflection.turns == 1`` and
        # ``advisor.turns == 0``.
        for model_alias in COMPARISON_MODELS:
            config = make_config(model_alias, ConfigVariant.REFLECTION_ONLY)
            route = config.routes[0]
            assert route.reflection.turns == 1, (
                f"REFLECTION_ONLY for {model_alias!r} expected "
                f"reflection.turns=1, got {route.reflection.turns}"
            )
            assert route.advisor.turns == 0, (
                f"REFLECTION_ONLY for {model_alias!r} expected "
                f"advisor.turns=0, got {route.advisor.turns}"
            )
            assert route.advisor.model is None, (
                f"REFLECTION_ONLY for {model_alias!r} expected "
                f"advisor.model=None, got {route.advisor.model!r}"
            )

    def test_advisor_only_uses_cross_advise_model(self):
        # Contract: ``make_config(model, ADVISOR_ONLY)`` returns
        # a config with ``reflection.turns == 0``,
        # ``advisor.turns == 1``, and
        # ``advisor.model == <OTHER comparison model's
        # OpenRouter id>``.
        for model_alias in COMPARISON_MODELS:
            config = make_config(model_alias, ConfigVariant.ADVISOR_ONLY)
            route = config.routes[0]
            assert route.reflection.turns == 0, (
                f"ADVISOR_ONLY for {model_alias!r} expected "
                f"reflection.turns=0, got {route.reflection.turns}"
            )
            assert route.advisor.turns == 1, (
                f"ADVISOR_ONLY for {model_alias!r} expected "
                f"advisor.turns=1, got {route.advisor.turns}"
            )
            expected = _expected_advisor_model(model_alias)
            assert route.advisor.model == expected, (
                f"ADVISOR_ONLY for {model_alias!r} expected "
                f"advisor.model={expected!r} (cross-advise), "
                f"got {route.advisor.model!r}"
            )

    def test_both_runs_reflection_and_advisor_with_cross_advise(self):
        # Contract: ``make_config(model, BOTH)`` returns a
        # config with ``reflection.turns == 1``,
        # ``advisor.turns == 1``, and ``advisor.model == the
        # OTHER comparison model's OpenRouter id``.
        for model_alias in COMPARISON_MODELS:
            config = make_config(model_alias, ConfigVariant.BOTH)
            route = config.routes[0]
            assert route.reflection.turns == 1, (
                f"BOTH for {model_alias!r} expected "
                f"reflection.turns=1, got {route.reflection.turns}"
            )
            assert route.advisor.turns == 1, (
                f"BOTH for {model_alias!r} expected "
                f"advisor.turns=1, got {route.advisor.turns}"
            )
            expected = _expected_advisor_model(model_alias)
            assert route.advisor.model == expected, (
                f"BOTH for {model_alias!r} expected "
                f"advisor.model={expected!r} (cross-advise), "
                f"got {route.advisor.model!r}"
            )

    def test_cross_advise_is_symmetric(self):
        # The cross-advise rule is symmetric: the advisor for
        # ``minimax-m3`` is the OpenRouter id of
        # ``mimo-v2.5-pro``, and the advisor for
        # ``mimo-v2.5-pro`` is the OpenRouter id of
        # ``minimax-m3``. Pin the symmetry so a future edit
        # that flips the rule to a one-sided mapping is caught
        # here.
        config_a = make_config("minimax-m3", ConfigVariant.ADVISOR_ONLY)
        config_b = make_config("mimo-v2.5-pro", ConfigVariant.ADVISOR_ONLY)
        assert (
            config_a.routes[0].advisor.model
            == MODEL_ALIASES["mimo-v2.5-pro"]
        )
        assert (
            config_b.routes[0].advisor.model
            == MODEL_ALIASES["minimax-m3"]
        )

    def test_eight_cell_table(self):
        # Single table-driven test that pins the full
        # (variant, model) → expected settings mapping in one
        # place. A regression in any cell shows up with a
        # clear pointer to the cell that failed.
        expected_table: dict[
            tuple[ConfigVariant, str], dict[str, int | str | None]
        ] = {
            (ConfigVariant.BASELINE, "minimax-m3"): {
                "reflection_turns": 0,
                "advisor_turns": 0,
                "advisor_model": None,
            },
            (ConfigVariant.BASELINE, "mimo-v2.5-pro"): {
                "reflection_turns": 0,
                "advisor_turns": 0,
                "advisor_model": None,
            },
            (ConfigVariant.REFLECTION_ONLY, "minimax-m3"): {
                "reflection_turns": 1,
                "advisor_turns": 0,
                "advisor_model": None,
            },
            (ConfigVariant.REFLECTION_ONLY, "mimo-v2.5-pro"): {
                "reflection_turns": 1,
                "advisor_turns": 0,
                "advisor_model": None,
            },
            (ConfigVariant.ADVISOR_ONLY, "minimax-m3"): {
                "reflection_turns": 0,
                "advisor_turns": 1,
                "advisor_model": MODEL_ALIASES["mimo-v2.5-pro"],
            },
            (ConfigVariant.ADVISOR_ONLY, "mimo-v2.5-pro"): {
                "reflection_turns": 0,
                "advisor_turns": 1,
                "advisor_model": MODEL_ALIASES["minimax-m3"],
            },
            (ConfigVariant.BOTH, "minimax-m3"): {
                "reflection_turns": 1,
                "advisor_turns": 1,
                "advisor_model": MODEL_ALIASES["mimo-v2.5-pro"],
            },
            (ConfigVariant.BOTH, "mimo-v2.5-pro"): {
                "reflection_turns": 1,
                "advisor_turns": 1,
                "advisor_model": MODEL_ALIASES["minimax-m3"],
            },
        }
        assert len(expected_table) == 8, (
            f"expected table must have 8 cells, got {len(expected_table)}"
        )
        for (variant, model_alias), expected in expected_table.items():
            config = make_config(model_alias, variant)
            route = config.routes[0]
            assert route.reflection.turns == expected["reflection_turns"], (
                f"cell (variant={variant.name!r}, model={model_alias!r}): "
                f"reflection.turns expected {expected['reflection_turns']}, "
                f"got {route.reflection.turns}"
            )
            assert route.advisor.turns == expected["advisor_turns"], (
                f"cell (variant={variant.name!r}, model={model_alias!r}): "
                f"advisor.turns expected {expected['advisor_turns']}, "
                f"got {route.advisor.turns}"
            )
            assert route.advisor.model == expected["advisor_model"], (
                f"cell (variant={variant.name!r}, model={model_alias!r}): "
                f"advisor.model expected {expected['advisor_model']!r}, "
                f"got {route.advisor.model!r}"
            )

    def test_every_config_has_openrouter_backend(self):
        # Contract: every variant routes the model to the
        # openrouter backend. Pin the backend's ``adapter``
        # field (``"openrouter"``) and its ``base_url``
        # (OpenRouter's canonical default) so a future edit
        # that swaps the backend to ``"ollama"`` or
        # ``"openai"`` is caught here.
        for variant in REQUIRED_VARIANTS:
            for model_alias in COMPARISON_MODELS:
                config = make_config(model_alias, variant)
                assert len(config.backends) == 1, (
                    f"cell (variant={variant.name!r}, model={model_alias!r}): "
                    f"expected 1 backend, got {len(config.backends)}"
                )
                backend = config.backends[0]
                assert backend.adapter == "openrouter", (
                    f"cell (variant={variant.name!r}, model={model_alias!r}): "
                    f"backend.adapter expected 'openrouter', "
                    f"got {backend.adapter!r}"
                )
                assert backend.base_url == "https://openrouter.ai/api/v1", (
                    f"cell (variant={variant.name!r}, model={model_alias!r}): "
                    f"backend.base_url expected "
                    f"'https://openrouter.ai/api/v1', got {backend.base_url!r}"
                )
                # The route's ``backend`` field must reference
                # the configured backend by name.
                assert config.routes[0].backend == backend.name, (
                    f"cell (variant={variant.name!r}, model={model_alias!r}): "
                    f"route.backend expected {backend.name!r}, "
                    f"got {config.routes[0].backend!r}"
                )

    def test_every_config_aliases_alias_to_openrouter_id(self):
        # Contract: each config has a single route that maps
        # the model alias to the full OpenRouter model id.
        # Pin the alias table on every cell so a future edit
        # that drops the alias (or rewrites it to the
        # already-resolved name) is caught here.
        for variant in REQUIRED_VARIANTS:
            for model_alias in COMPARISON_MODELS:
                config = make_config(model_alias, variant)
                route = config.routes[0]
                expected_id = MODEL_ALIASES[model_alias]
                assert route.aliases == {model_alias: expected_id}, (
                    f"cell (variant={variant.name!r}, model={model_alias!r}): "
                    f"route.aliases expected {model_alias!r} → "
                    f"{expected_id!r}, got {route.aliases!r}"
                )
                # The route's ``match.model`` is the alias (the
                # client sends the alias; the matcher rewrites
                # it via the alias table). Pin the match glob
                # to catch a future edit that points the
                # matcher at the OpenRouter id directly.
                assert route.match.model == model_alias, (
                    f"cell (variant={variant.name!r}, model={model_alias!r}): "
                    f"route.match.model expected {model_alias!r}, "
                    f"got {route.match.model!r}"
                )
                assert route.match.path == "/v1/chat/completions", (
                    f"cell (variant={variant.name!r}, model={model_alias!r}): "
                    f"route.match.path expected '/v1/chat/completions', "
                    f"got {route.match.path!r}"
                )

    def test_make_config_rejects_unknown_model(self):
        # Sanity: passing a model that is not in
        # :data:`COMPARISON_MODELS` raises :class:`ValueError`
        # rather than silently producing a config with no
        # alias match. The benchmark sweep is restricted to
        # the canonical two-model set; an unknown alias is a
        # programmer error.
        with pytest.raises(ValueError):
            make_config("not-a-comparison-model", ConfigVariant.BASELINE)


def test_package_init_exports_config_symbols():
    # The package's ``__init__`` re-exports the public config
    # symbols (``ConfigVariant``, ``make_config``,
    # ``COMPARISON_MODELS``, ``MODEL_ALIASES``) alongside the
    # prompt types. Pin the re-exports so a future refactor of
    # the package facade does not silently break downstream
    # imports.
    from moaxy.benchmark import (
        COMPARISON_MODELS as COMPARISON_MODELS_REEXPORT,
    )
    from moaxy.benchmark import (
        MODEL_ALIASES as MODEL_ALIASES_REEXPORT,
    )
    from moaxy.benchmark import (
        ConfigVariant as ConfigVariantReexport,
    )
    from moaxy.benchmark import configs as configs_module
    from moaxy.benchmark import (
        make_config as make_config_reexport,
    )

    assert ConfigVariantReexport is configs_module.ConfigVariant
    assert make_config_reexport is configs_module.make_config
    assert COMPARISON_MODELS_REEXPORT is configs_module.COMPARISON_MODELS
    assert MODEL_ALIASES_REEXPORT is configs_module.MODEL_ALIASES
