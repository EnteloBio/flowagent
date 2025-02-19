"""Single-cell RNA-seq workflow implementation."""

from typing import Dict, Any, List
import os
from pathlib import Path
from pydantic import BaseModel

from .base import WorkflowBase, WorkflowRegistry
from ..utils.logging import get_logger

logger = get_logger(__name__)

class SingleCellParams(BaseModel):
    """Parameters for SingleCell workflow."""
    
    input_files: List[str]
    reference_transcriptome: str
    output_dir: str
    chemistry: str = "10xv3"  # 10x chemistry version
    expect_cells: int = 3000  # Expected number of cells
    force_cells: int = None  # Force pipeline to use this number of cells
    threads: int = 4
    memory: str = "32G"  # Single-cell typically needs more memory

class SingleCellWorkflow(WorkflowBase):
    """Workflow for single-cell RNA-seq analysis."""
    
    name = "single_cell"
    description = "Single-cell RNA-seq analysis using Cell Ranger and Seurat"
    required_tools = ["cellranger", "R"]
    
    def __init__(self, params: Dict[str, Any], task_agent: Any):
        """Initialize workflow with parameters."""
        super().__init__(params, task_agent)
        self.params = SingleCellParams(**params)
        
        # Initialize execution agents
        self.agents = {
            'cellranger_mkref': CellRangerMkrefAgent(task_agent),
            'cellranger_count': CellRangerCountAgent(task_agent),
            'seurat_qc': SeuratQCAgent(task_agent),
            'seurat_analysis': SeuratAnalysisAgent(task_agent)
        }
    
    async def validate_params(self) -> bool:
        """Validate workflow parameters."""
        # Check input files exist
        for file in self.params.input_files:
            if not os.path.exists(file):
                self.logger.error(f"Input file not found: {file}")
                return False
        
        # Check reference exists
        if not os.path.exists(self.params.reference_transcriptome):
            self.logger.error(f"Reference not found: {self.params.reference_transcriptome}")
            return False
            
        # Validate chemistry version
        valid_chemistry = ["10xv2", "10xv3", "10xv3.1"]
        if self.params.chemistry not in valid_chemistry:
            self.logger.error(f"Invalid chemistry version. Must be one of: {valid_chemistry}")
            return False
            
        # Create output directory
        os.makedirs(self.params.output_dir, exist_ok=True)
        
        return True
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the workflow."""
        results = {}
        
        # 1. Create Cell Ranger reference
        ref_results = await self.agents['cellranger_mkref'].execute({
            "reference": self.params.reference_transcriptome,
            "output_dir": os.path.join(self.params.output_dir, "reference")
        })
        results['reference'] = ref_results
        
        # 2. Run Cell Ranger count
        count_results = await self.agents['cellranger_count'].execute({
            "input_files": self.params.input_files,
            "reference": ref_results['result']['reference_path'],
            "output_dir": os.path.join(self.params.output_dir, "counts"),
            "chemistry": self.params.chemistry,
            "expect_cells": self.params.expect_cells,
            "force_cells": self.params.force_cells,
            "threads": self.params.threads,
            "memory": self.params.memory
        })
        results['counts'] = count_results
        
        # 3. Run Seurat QC
        qc_results = await self.agents['seurat_qc'].execute({
            "input_path": count_results['result']['filtered_feature_bc_matrix'],
            "output_dir": os.path.join(self.params.output_dir, "qc")
        })
        results['qc'] = qc_results
        
        # 4. Run Seurat Analysis
        analysis_results = await self.agents['seurat_analysis'].execute({
            "input_path": qc_results['result']['filtered_data'],
            "output_dir": os.path.join(self.params.output_dir, "analysis")
        })
        results['analysis'] = analysis_results
        
        return results
    
    async def validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate workflow results."""
        # Check reference results
        if not await self.agents['cellranger_mkref'].validate(results.get('reference', {})):
            return False
            
        # Check count results
        if not await self.agents['cellranger_count'].validate(results.get('counts', {})):
            return False
            
        # Check QC results
        if not await self.agents['seurat_qc'].validate(results.get('qc', {})):
            return False
            
        # Check analysis results
        if not await self.agents['seurat_analysis'].validate(results.get('analysis', {})):
            return False
            
        return True

# Register the workflow
WorkflowRegistry.register(SingleCellWorkflow)
