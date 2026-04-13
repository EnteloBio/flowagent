"""Pydantic models for all LLM input/output contracts.

These schemas are used for structured-output calls (JSON mode) across
all providers, eliminating fragile ``_clean_llm_response`` regex parsing.

OpenAI strict mode requirements:
  - Every object must have ``additionalProperties: false``
  - No ``anyOf``/``oneOf`` with ``$ref`` + ``null``
  - No ``default`` values on optional fields
  - All fields must be ``required``
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Pipeline planning context ─────────────────────────────────

@dataclass
class PipelineContext:
    """Everything the planning phase discovers or asks the user.

    Produced by ``gather_pipeline_context()`` and consumed by
    ``generate_workflow_plan()`` so the LLM receives concrete reference
    paths / download URLs instead of guessing.
    """
    input_files: List[str] = field(default_factory=list)
    paired_end: bool = True
    organism: str = "human"
    genome_build: str = "GRCh38"
    reference_fasta: Optional[str] = None
    annotation_gtf: Optional[str] = None
    reference_source: str = "ensembl"
    reference_url: Optional[str] = None
    annotation_url: Optional[str] = None
    workflow_type: str = ""
    extra_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def needs_reference_download(self) -> bool:
        return self.reference_fasta is None

    @property
    def needs_annotation_download(self) -> bool:
        return self.annotation_gtf is None


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
