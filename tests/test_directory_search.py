"""Test the directory search functionality in the report generator and analysis system."""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from flowagent.agents.agentic.analysis_system import AgenticAnalysisSystem
from flowagent.analysis.report_generator import ReportGenerator


@pytest.fixture
def test_directory_structure():
    """Create a test directory structure that mimics our RNA-seq workflow output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create main workflow output directory
        workflow_dir = Path(tmp_dir) / "workflow_output"
        workflow_dir.mkdir()
        
        # Create results directory next to the workflow directory
        results_dir = Path(tmp_dir) / "results" / "rna_seq_kallisto"
        results_dir.mkdir(parents=True)
        
        # Create FastQC output in results directory
        fastqc_dir = results_dir / "fastqc"
        fastqc_dir.mkdir()
        fastqc_file = fastqc_dir / "sample1_fastqc.html"
        with open(fastqc_file, "w") as f:
            f.write("<html><body>FastQC Report</body></html>")
        
        # Create Kallisto output in results directory
        kallisto_dir = results_dir / "kallisto_quant" / "sample1"
        kallisto_dir.mkdir(parents=True)
        
        # Create abundance and run info files
        abundance_file = kallisto_dir / "abundance.tsv"
        with open(abundance_file, "w") as f:
            f.write("target_id\tlength\teff_length\test_counts\ttpm\n")
            f.write("ENST1\t1000\t900\t500\t10.5\n")
        
        run_info_file = kallisto_dir / "run_info.json"
        with open(run_info_file, "w") as f:
            json.dump({
                "n_processed": 1000000,
                "n_pseudoaligned": 800000,
                "n_unique": 750000,
                "p_pseudoaligned": 80.0,
                "kallisto_version": "0.46.1"
            }, f)
            
        # Create a log file in the workflow directory
        log_file = workflow_dir / "workflow.log"
        with open(log_file, "w") as f:
            f.write("INFO: Workflow started\n")
            f.write("INFO: Running FastQC\n")
            f.write("INFO: Running Kallisto\n")
        
        # Create a workflow.json file that points to the results directory
        workflow_json = workflow_dir / "workflow.json"
        with open(workflow_json, "w") as f:
            json.dump({
                "steps": [
                    {
                        "command": f"mkdir -p {results_dir}"
                    }
                ]
            }, f)
        
        yield {
            "workflow_dir": workflow_dir,
            "results_dir": results_dir,
            "fastqc_file": fastqc_file,
            "abundance_file": abundance_file,
            "run_info_file": run_info_file
        }


@pytest.mark.asyncio
async def test_analysis_system_directory_search(test_directory_structure):
    """Test that the analysis system can find files in different directories."""
    # Initialize the analysis system
    analysis_system = AgenticAnalysisSystem()
    
    # Run the analysis
    analysis = await analysis_system.analyze_results(test_directory_structure["workflow_dir"])
    
    # Verify that it found the files
    assert analysis["quality_analysis"]["fastqc_available"] is True
    assert "kallisto" in analysis["workflow_info"]["tools_used"]
    assert analysis["quantification_analysis"]["tools"]["kallisto"]["sample_count"] > 0


@pytest.mark.asyncio
@patch('flowagent.analysis.report_generator.LLMInterface')
async def test_report_generator_directory_search(mock_llm, test_directory_structure):
    """Test that the report generator can find files in different directories."""
    # Setup the mock LLM
    mock_llm_instance = MagicMock()
    mock_llm.return_value = mock_llm_instance
    
    # Initialize the report generator
    report_generator = ReportGenerator()
    
    # Make sure the mock is being used
    assert report_generator.llm is mock_llm_instance
    
    # Collect the tool outputs directly
    outputs = await report_generator._collect_tool_outputs(test_directory_structure["workflow_dir"])
    
    # Verify that it found the files
    assert len(outputs["quality_control"]["fastqc"]) > 0
    assert len(outputs["expression"]["samples"]) > 0
    assert outputs["expression"]["samples"]["sample1"]["n_processed"] == 1000000


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
