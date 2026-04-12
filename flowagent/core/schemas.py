"""Pydantic models for all LLM input/output contracts.

These schemas are used for structured-output calls (JSON mode) across
all providers, eliminating fragile ``_clean_llm_response`` regex parsing.

OpenAI strict mode requirements:
  - Every object must have ``additionalProperties: false``
  - No ``anyOf``/``oneOf`` with ``$ref`` + ``null``
  - No ``default`` values on optional fields
  - All fields must be ``required``
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Workflow planning ──────────────────────────────────────────

class WorkflowStepSchema(BaseModel):
    """A single step in a workflow plan."""
    name: str = Field(..., description="Short unique name for the step")
    command: str = Field(..., description="Shell command to execute")
    dependencies: List[str] = Field(default_factory=list, description="Names of prerequisite steps")
    outputs: List[str] = Field(default_factory=list, description="Expected output file patterns")
    description: str = Field("", description="Brief description of the step")

    model_config = {"extra": "forbid"}


class WorkflowPlanSchema(BaseModel):
    """Complete workflow plan returned by the LLM."""
    workflow_type: str = Field(..., description="Type of workflow e.g. rna_seq_kallisto")
    steps: List[WorkflowStepSchema]

    model_config = {"extra": "forbid"}


# ── File pattern extraction ────────────────────────────────────

class PatternGroup(BaseModel):
    pattern: str
    group: str


class FileRelationships(BaseModel):
    type: str = Field("single", description="single or paired")
    pattern_groups: List[PatternGroup] = Field(default_factory=list)


class FilePatternResponse(BaseModel):
    """File pattern extraction response."""
    patterns: List[str]
    relationships: FileRelationships


# ── Prompt routing ─────────────────────────────────────────────

class PromptRouting(BaseModel):
    """Determine whether a user prompt is a 'run' or 'analyze' request."""
    action: str = Field(..., description="'run' or 'analyze'")
    paths: List[str] = Field(default_factory=list)
    resume: bool = False
    analysis_dir: Optional[str] = None


# ── Resource suggestion ───────────────────────────────────────

class ResourceSuggestion(BaseModel):
    """LLM-suggested resource profile for an unknown tool."""
    profile: str = Field("default")
    memory: str = Field("4G")
    cpus: int = Field(1)
    time_min: int = Field(60)


# ── Pipeline code refinement ──────────────────────────────────

class PipelineRefinement(BaseModel):
    """LLM-refined pipeline code with validation notes."""
    code: str = Field(..., description="Refined pipeline source code")
    validation_notes: List[str] = Field(default_factory=list)
    container_images: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of tool name → container image URI",
    )


# ── Analysis report ───────────────────────────────────────────

class AnalysisSection(BaseModel):
    heading: str
    content: str


class AnalysisReport(BaseModel):
    """Structured analysis report from LLM."""
    summary: str
    sections: List[AnalysisSection] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


# ── Helper: convert Pydantic model → JSON Schema dict ─────────

def _make_strict(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively enforce OpenAI strict-mode constraints on a JSON Schema."""
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        props = schema.get("properties", {})
        # Strict mode requires every property in "required"
        schema["required"] = list(props.keys())
        for prop in props.values():
            _make_strict(prop)
            prop.pop("default", None)
    if "items" in schema:
        _make_strict(schema["items"])
    if "$ref" in schema:
        pass  # $ref is resolved at the top level via $defs
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            for sub in schema[key]:
                _make_strict(sub)
    # Process $defs
    for defn in schema.get("$defs", {}).values():
        _make_strict(defn)
    return schema


def to_json_schema(model_class: type[BaseModel]) -> Dict[str, Any]:
    """Return the JSON Schema dict suitable for ``response_format`` / ``response_schema``.

    Applies strict-mode transformations so the schema is accepted by
    OpenAI's ``json_schema`` response format with ``"strict": true``.
    """
    schema = model_class.model_json_schema()
    return _make_strict(schema)
