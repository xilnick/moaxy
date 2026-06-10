"""Pipeline orchestration for moaxy.

The pipeline module owns the typed context that flows through the
reflection + advisor stages, the default system prompts, the message
builders, the orchestrator itself, and the fallback walker used to retry
and walk the per-model fallback list. Adapters stay backend-agnostic;
the pipeline composes them per request.
"""

from moaxy.pipeline.context import PipelineContext, PipelineEvent
from moaxy.pipeline.fallback import UpstreamExhaustedError, call_with_fallbacks
from moaxy.pipeline.message_builders import (
    build_advisor_messages,
    build_reflection_messages,
    build_revision_messages,
)
from moaxy.pipeline.prompts import (
    DEFAULT_ADVISOR_PROMPT,
    DEFAULT_REFLECT_PROMPT,
)
from moaxy.pipeline.reflector import parse_confidence, reflect_turn

__all__ = [
    "DEFAULT_ADVISOR_PROMPT",
    "DEFAULT_REFLECT_PROMPT",
    "PipelineContext",
    "PipelineEvent",
    "UpstreamExhaustedError",
    "build_advisor_messages",
    "build_reflection_messages",
    "build_revision_messages",
    "call_with_fallbacks",
    "parse_confidence",
    "reflect_turn",
]
