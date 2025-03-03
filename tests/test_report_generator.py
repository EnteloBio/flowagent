import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from flowagent.analysis.report_generator import ReportGenerator
import json

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock()
    with patch('flowagent.core.llm.AsyncOpenAI', return_value=mock_client):
        yield mock_client

@pytest.fixture
def report_generator(mock_openai_client):
    """Create report generator with mocked OpenAI client."""
    with patch('pathlib.Path.exists', return_value=True), \
         patch('flowagent.core.llm.settings') as mock_settings:
        mock_settings.OPENAI_API_KEY = 'test-key'
        mock_settings.OPENAI_MODEL = 'gpt-4'
        mock_settings.OPENAI_FALLBACK_MODEL = 'gpt-3.5-turbo'
        return ReportGenerator()

@pytest.fixture
def mock_openai_response():
    """Create mock OpenAI API response."""
    mock_response = AsyncMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "status": "success",
            "analysis": "Test analysis completed",
            "metrics": {
                "total_reads": 1000,
                "mapped_reads": 800,
                "alignment_rate": 80.0
            }
        })))
    ]
    return mock_response

@pytest.mark.asyncio
async def test_report_generation(report_generator, mock_openai_client, tmp_path):
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
    (kallisto_dir / "kallisto.log").write_text("Processed reads\nQuantification complete\n")

    # Mock LLM response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(
            content=json.dumps({
                "status": "success",
                "analysis": "Test analysis completed",
                "metrics": {
                    "total_reads": 1000,
                    "mapped_reads": 800,
                    "alignment_rate": 80.0
                }
            })
        ))
    ]
    mock_openai_client.chat.completions.create.return_value = mock_response

    # Generate report
    with patch.object(report_generator.llm, 'generate_analysis') as mock_analyze:
        mock_analyze.return_value = {
            "status": "success",
            "analysis": "Test analysis completed",
            "metrics": {
                "total_reads": 1000,
                "mapped_reads": 800,
                "alignment_rate": 80.0
            }
        }
        report = await report_generator.generate_report(output_dir)

    assert report is not None
    assert isinstance(report, dict)
    assert "success" in report.get("status", "")

@pytest.mark.asyncio
async def test_report_error_handling(report_generator, mock_openai_client, tmp_path):
    """Test report generation error handling."""
    # Test with non-existent directory
    nonexistent_dir = tmp_path / "nonexistent"
    result = await report_generator.generate_report(nonexistent_dir)
    assert result["success"] == False
    assert "does not exist" in result["message"]

    # Test with empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = await report_generator.analyze_tool_outputs(empty_dir)
    assert result["status"] == "error"
    assert "No files found in directory" in result["message"]

    # Test with invalid file format
    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "abundance.tsv").write_text("invalid format")
    
    # Mock LLM response for invalid format
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(
            content=json.dumps({
                "status": "success",
                "analysis": "Analysis with warnings",
                "metrics": {
                    "warning": "Invalid file format detected"
                }
            })
        ))
    ]
    mock_openai_client.chat.completions.create.return_value = mock_response

    # Mock analysis for invalid format
    with patch.object(report_generator.llm, 'generate_analysis') as mock_analyze:
        mock_analyze.return_value = {
            "status": "success",
            "analysis": "Analysis with warnings",
            "metrics": {
                "warning": "Invalid file format detected"
            }
        }
        result = await report_generator.generate_report(invalid_dir)
        assert result["status"] == "success"  # Should still generate report with warnings
