from pydantic import BaseModel, ValidationError
from typing import Dict, Type, Any, List
import logging

logger = logging.getLogger(__name__)

class ValidationService:
    def __init__(self, schema_registry: Dict[str, Type[BaseModel]]):
        self.schemas = schema_registry
        
    def validate_inputs(self, step_name: str, inputs: Dict[str, Any]) -> bool:
        if schema := self.schemas.get(step_name):
            try:
                schema(**inputs)
                return True
            except ValidationError as e:
                logger.error(f"Validation failed for {step_name}: {e}")
                return False
        return True

class BioValidationSchemas:
    class KallistoQuantInput(BaseModel):
        index: str
        output_dir: str
        threads: int = 4
        single_end: bool = False
        bootstrap_samples: int = 100
        
    class FastQCInput(BaseModel):
        input_files: List[str]
        output_dir: str
        threads: int = 4
        
    class MultiQCInput(BaseModel):
        input_dir: str
        output_dir: str
        force: bool = True
        
    class DESeq2Input(BaseModel):
        count_matrix: str
        metadata: str
        design_formula: str
        output_dir: str
        
    @classmethod
    def get_schema(cls, tool_name: str) -> Type[BaseModel]:
        return getattr(cls, f"{tool_name}Input", None)
        
    @classmethod
    def validate_tool_output(cls, tool_name: str, output: Dict[str, Any]) -> bool:
        """Validate tool output against expected schema"""
        try:
            if tool_name == "kallisto_quant":
                required_files = ["abundance.h5", "abundance.tsv", "run_info.json"]
                return all(output.get(f) for f in required_files)
            elif tool_name == "fastqc":
                return output.get("html_report") and output.get("zip_report")
            return True
        except Exception as e:
            logger.error(f"Output validation failed for {tool_name}: {e}")
            return False
