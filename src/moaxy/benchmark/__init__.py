"""M7 benchmark harness for moaxy.

The benchmark module owns the curated coding prompt set
(:mod:`moaxy.benchmark.prompts`), the harness that drives the proxy
with scripted (model, configuration) cells
(:mod:`moaxy.benchmark.harness`), the four configuration variants
(:mod:`moaxy.benchmark.configs`), the deterministic and LLM-judge
scorers (:mod:`moaxy.benchmark.scoring`), the CLI entry point
(:mod:`moaxy.benchmark.run`), and the markdown report generator
(:mod:`moaxy.benchmark.report`). The benchmark is importable but NOT
auto-invoked; the moaxy proxy runs unchanged without it. The live
benchmark run output is committed under ``.benchmarks/results/`` and
the API key is NEVER committed.
"""

from moaxy.benchmark.configs import (
    COMPARISON_MODELS,
    MODEL_ALIASES,
    ConfigVariant,
    make_config,
)
from moaxy.benchmark.prompts import (
    PROMPT_SET,
    BugFixPrompt,
    CodingPrompt,
    ExplainPrompt,
    FunctionFromDocstringPrompt,
    RefactorPrompt,
)
from moaxy.benchmark.scoring import (
    DEFAULT_FALLBACK_SCORE,
    DEFAULT_JUDGE_MODEL,
    LLMJudgeScorer,
    parse_judge_score,
    score_bug_fix,
    score_function_from_docstring,
    score_refactor,
    score_with_judge,
)

__all__ = [
    "COMPARISON_MODELS",
    "DEFAULT_FALLBACK_SCORE",
    "DEFAULT_JUDGE_MODEL",
    "LLMJudgeScorer",
    "MODEL_ALIASES",
    "BugFixPrompt",
    "CodingPrompt",
    "ConfigVariant",
    "ExplainPrompt",
    "FunctionFromDocstringPrompt",
    "PROMPT_SET",
    "RefactorPrompt",
    "make_config",
    "parse_judge_score",
    "score_bug_fix",
    "score_function_from_docstring",
    "score_refactor",
    "score_with_judge",
]

