"""M7 benchmark scoring modules.

The :mod:`moaxy.benchmark.scoring` package owns the two scoring
strategies the benchmark harness uses to score a model response
against a :class:`~moaxy.benchmark.prompts.CodingPrompt`:

* :mod:`moaxy.benchmark.scoring.deterministic` — the three
  deterministic scorers for the ``function_from_docstring``,
  ``bug_fix``, and ``refactor`` prompt categories. Each scorer is a
  pure function that takes a prompt and a model output and returns
  a score in ``{0.0, 1.0}``. No LLM call is involved.
* :mod:`moaxy.benchmark.scoring.judge` — the LLM-as-judge scorer
  for the ``explain`` category. Uses an external model (the
  cheapest local Ollama model, ``deepseek-v4-pro:cloud``) to score
  the model's explanation on a 0-10 rubric; the result is
  normalised to ``[0.0, 1.0]`` by the report generator.

The package is importable but not auto-invoked. The benchmark
harness imports the scorers it needs from this package at run
time; the rest of moaxy is unchanged.
"""

from moaxy.benchmark.scoring.deterministic import (
    score_bug_fix,
    score_function_from_docstring,
    score_refactor,
)

__all__ = [
    "score_bug_fix",
    "score_function_from_docstring",
    "score_refactor",
]
