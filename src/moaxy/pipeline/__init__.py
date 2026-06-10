"""Pipeline orchestration for moaxy.

The pipeline module owns the typed context that flows through the
reflection + advisor stages, the default system prompts, the message
builders, the orchestrator itself, and the fallback walker used to retry
and walk the per-model fallback list. Adapters stay backend-agnostic;
the pipeline composes them per request.
"""

from moaxy.pipeline.context import PipelineContext, PipelineEvent
from moaxy.pipeline.prompts import (
    DEFAULT_ADVISOR_PROMPT,
    DEFAULT_REFLECT_PROMPT,
)

__all__ = [
    "DEFAULT_ADVISOR_PROMPT",
    "DEFAULT_REFLECT_PROMPT",
    "PipelineContext",
    "PipelineEvent",
]
