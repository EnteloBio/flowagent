import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from flowagent.analysis.report_generator import ReportGenerator
from flowagent.core.providers.base import ProviderResponse
import json


@pytest.fixture
def mock_provider():
    """Mock LLM provider."""
    provider = AsyncMock()
    provider.chat = AsyncMock(return_value=ProviderResponse(content="{}", model="gpt-4.1"))
    return provider


@pytest.fixture
def report_generator(mock_provider):
    """Create report generator with mocked provider."""
    with patch('flowagent.core.llm.create_provider', return_value=mock_provider), \
         patch('flowagent.core.llm.AsyncOpenAI'), \
         patch('flowagent.core.llm.settings') as mock_settings, \
         patch('pathlib.Path.exists', return_value=True):
        mock_settings.OPENAI_API_KEY = 'test-key'
        mock_settings.OPENAI_MODEL = 'gpt-4.1'
        mock_settings.OPENAI_FALLBACK_MODEL = 'gpt-4.1-mini'
        mock_settings.LLM_PROVIDER = 'openai'
        mock_settings.LLM_MODEL = 'gpt-4.1'
        mock_settings.LLM_FALLBACK_MODEL = 'gpt-4.1-mini'
        mock_settings.LLM_BASE_URL = None
        mock_settings.OPENAI_BASE_URL = None
        mock_settings.active_api_key = 'test-key'
        return ReportGenerator()


@pytest.mark.asyncio
async def test_report_generation(report_generator, tmp_path):
    """Test basic report generation."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()

    (output_dir / "workflow_dag.png").touch()

    kallisto_dir = output_dir / "kallisto_output"
    kallisto_dir.mkdir()
    (kallisto_dir / "abundance.h5").touch()
    (kallisto_dir / "abundance.tsv").write_text("target_id\tlength\teff_length\test_counts\ttpm\n")
    (kallisto_dir / "run_info.json").write_text('{"n_targets": 100, "n_processed": 1000}')

    fastqc_dir = output_dir / "fastqc_output"
    fastqc_dir.mkdir()
    (fastqc_dir / "sample_fastqc.html").touch()
    (fastqc_dir / "sample_fastqc.zip").touch()

    (output_dir / "workflow.log").write_text("Started workflow\nCompleted successfully\n")
    (kallisto_dir / "kallisto.log").write_text("Processed reads\nQuantification complete\n")

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
async def test_report_error_handling(report_generator, tmp_path):
    """Test report generation error handling."""
    nonexistent_dir = tmp_path / "nonexistent"
    result = await report_generator.generate_report(nonexistent_dir)
    assert result["success"] == False
    assert "does not exist" in result["message"]

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = await report_generator.analyze_tool_outputs(empty_dir)
    assert result["status"] == "error"
    assert "No files found in directory" in result["message"]

    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    (invalid_dir / "abundance.tsv").write_text("invalid format")

    with patch.object(report_generator.llm, 'generate_analysis') as mock_analyze:
        mock_analyze.return_value = {
            "status": "success",
            "analysis": "Analysis with warnings",
            "metrics": {
                "warning": "Invalid file format detected"
            }
        }
        result = await report_generator.generate_report(invalid_dir)
        assert result["status"] == "success"
