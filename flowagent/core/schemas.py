"""Pydantic models for all LLM input/output contracts.

These schemas are used for structured-output calls (JSON mode) across
all providers, eliminating fragile ``_clean_llm_response`` regex parsing.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Workflow planning ──────────────────────────────────────────

class ResourceSpec(BaseModel):
    """Resource requirements for a single workflow step."""
    memory: str = Field("4G", description="Memory allocation e.g. '8G'")
    cpus: int = Field(1, description="Number of CPU cores")
    time_min: int = Field(60, description="Time limit in minutes")
    profile: str = Field("default", description="Resource profile name")


class WorkflowStepSchema(BaseModel):
    """A single step in a workflow plan."""
    name: str = Field(..., description="Short unique name for the step")
    command: str = Field(..., description="Shell command to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list, description="Names of prerequisite steps")
    outputs: List[str] = Field(default_factory=list, description="Expected output file patterns")
    resources: Optional[ResourceSpec] = None


class WorkflowPlanSchema(BaseModel):
    """Complete workflow plan returned by the LLM."""
    workflow_type: str = Field(..., description="Type of workflow e.g. rna_seq_kallisto")
    steps: List[WorkflowStepSchema]


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

def to_json_schema(model_class: type[BaseModel]) -> Dict[str, Any]:
    """Return the JSON Schema dict suitable for ``response_format`` / ``response_schema``."""
    schema = model_class.model_json_schema()
    schema.setdefault("additionalProperties", False)
    return schema
