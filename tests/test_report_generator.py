import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from flowagent.analysis.report_generator import ReportGenerator

@pytest.fixture
def report_generator():
    return ReportGenerator()

@pytest.mark.asyncio
async def test_report_generation(report_generator, tmp_path):
    """Test basic report generation."""
    # Create test output directory
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()

    # Create test workflow files
    (output_dir / "workflow_dag.png").touch()

    # Create kallisto output
    kallisto_dir = output_dir / "kallisto_output"
    kallisto_dir.mkdir()
    (kallisto_dir / "abundance.h5").touch()
    (kallisto_dir / "abundance.tsv").write_text("target_id\tlength\teff_length\test_counts\ttpm\n")
    (kallisto_dir / "run_info.json").write_text('{"n_targets": 100, "n_processed": 1000}')

    # Create FastQC output
    fastqc_dir = output_dir / "fastqc_output"
    fastqc_dir.mkdir()
    (fastqc_dir / "sample_fastqc.html").touch()
    (fastqc_dir / "sample_fastqc.zip").touch()

    # Create log files
    (output_dir / "workflow.log").write_text("Started workflow\nCompleted successfully\n")
    (output_dir / "kallisto.log").write_text("Processed reads\nQuantification complete\n")

    # Mock LLM analysis response
    mock_response = {"status": "success", "analysis": "Test analysis completed"}
    
    with patch('flowagent.core.llm.LLMInterface.generate_analysis', return_value=mock_response):
        # Generate report
        report = await report_generator.generate_report(output_dir)

        assert report is not None
        assert isinstance(report, dict)
        assert report.get("status") == "success"
        assert "analysis" in report

@pytest.mark.asyncio
async def test_report_error_handling(report_generator, tmp_path):
    """Test report generation error handling."""
    # Create empty directory
    output_dir = tmp_path / "empty_output"
    output_dir.mkdir()

    # Try to generate report from empty directory
    report = await report_generator.generate_report(output_dir)

    assert report is not None
    assert isinstance(report, dict)
    assert report.get("status") == "error"
    assert "message" in report
