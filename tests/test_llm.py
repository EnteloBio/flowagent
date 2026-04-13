import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
from flowagent.core.llm import LLMInterface
from flowagent.core.providers.base import ProviderResponse


@pytest.fixture
def mock_provider():
    """Mock LLM provider that replaces the real provider inside LLMInterface."""
    provider = AsyncMock()
    provider.chat = AsyncMock()
    provider.chat_with_tools = AsyncMock()
    provider.chat_structured = AsyncMock()
    provider.stream = AsyncMock()
    return provider


@pytest.fixture
def mock_settings():
    """Mock settings module with provider-based config."""
    with patch('flowagent.core.llm.settings') as ms:
        ms.OPENAI_API_KEY = 'test-key'
        ms.OPENAI_MODEL = 'gpt-4'
        ms.OPENAI_FALLBACK_MODEL = 'gpt-4.1-mini'
        ms.LLM_PROVIDER = 'openai'
        ms.LLM_MODEL = 'gpt-4'
        ms.LLM_FALLBACK_MODEL = 'gpt-4.1-mini'
        ms.LLM_BASE_URL = None
        ms.OPENAI_BASE_URL = None
        ms.active_api_key = 'test-key'
        yield ms


@pytest.fixture
def llm_interface(mock_provider, mock_settings):
    """Create LLM interface with mocked provider."""
    with patch('flowagent.core.llm.create_provider', return_value=mock_provider), \
         patch('flowagent.core.llm.AsyncOpenAI'), \
         patch('pathlib.Path.exists', return_value=True):
        llm = LLMInterface()
        llm.provider = mock_provider
        return llm


@pytest.fixture
def mock_file_pattern_response():
    """ProviderResponse for file pattern extraction."""
    return ProviderResponse(
        content=json.dumps({
            "patterns": ["*.fastq.gz"],
            "relationships": {
                "type": "single",
                "pattern_groups": [{"pattern": "*.fastq.gz", "group": "reads"}]
            }
        }),
        model="gpt-4",
    )


@pytest.fixture
def mock_workflow_response():
    """ProviderResponse for workflow plan generation."""
    return ProviderResponse(
        content=json.dumps({
            "workflow_type": "rna_seq",
            "steps": [
                {
                    "name": "create_dirs",
                    "command": "mkdir -p raw_data processed_data",
                    "dependencies": [],
                    "outputs": ["raw_data/", "processed_data/"]
                }
            ]
        }),
        model="gpt-4",
    )


@pytest.mark.asyncio
async def test_generate_workflow_plan_basic(llm_interface, mock_file_pattern_response, mock_workflow_response):
    """Test workflow plan generation via provider."""
    llm_interface.provider.chat = AsyncMock(
        side_effect=[mock_file_pattern_response, mock_workflow_response]
    )
    llm_interface.provider.chat_structured = AsyncMock(
        side_effect=NotImplementedError("structured output not available")
    )
    with patch('glob.glob', return_value=['test1.fastq.gz', 'test2.fastq.gz']):
        result = await llm_interface.generate_workflow_plan("Process RNA-seq data")
        assert result is not None
        assert isinstance(result, dict)
        assert "workflow_type" in result
        assert result["workflow_type"] == "rna_seq"


@pytest.mark.asyncio
async def test_call_openai_via_provider(llm_interface):
    """Test _call_openai delegates to provider.chat."""
    response_data = {
        "workflow_type": "rna_seq_kallisto",
        "steps": [{"name": "test", "command": "echo test", "dependencies": []}]
    }
    llm_interface.provider.chat = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps(response_data),
            model="gpt-4",
        )
    )

    messages = [{"role": "user", "content": "test prompt"}]
    result = await llm_interface._call_openai(messages)
    parsed_result = json.loads(result)
    assert isinstance(parsed_result, dict)
    assert parsed_result["workflow_type"] == "rna_seq_kallisto"
    llm_interface.provider.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_error_handling_invalid_type(llm_interface):
    """Test error handling for invalid prompt type."""
    with pytest.raises(TypeError):
        await llm_interface.generate_workflow_plan(123)  # type: ignore


@pytest.mark.asyncio
async def test_error_handling_no_files(llm_interface, mock_file_pattern_response):
    """Test error when no files match patterns."""
    llm_interface.provider.chat = AsyncMock(return_value=mock_file_pattern_response)
    with patch('glob.glob', return_value=[]):
        with pytest.raises(ValueError, match="No files found matching patterns"):
            await llm_interface.generate_workflow_plan("test prompt")


@pytest.mark.asyncio
async def test_tool_characteristics(llm_interface):
    """Test tool characteristics extraction."""
    with patch.object(llm_interface, '_get_tool_characteristics') as mock_get_chars:
        mock_get_chars.return_value = {
            "memory_weight": 0.5,
            "time_weight": 0.7
        }
        characteristics = llm_interface._get_tool_characteristics("kallisto")
        assert isinstance(characteristics, dict)
        assert "memory_weight" in characteristics
        assert "time_weight" in characteristics
        assert all(isinstance(v, float) for v in characteristics.values())


@pytest.mark.asyncio
async def test_generate_workflow_plan_with_steps(llm_interface, mock_file_pattern_response, mock_workflow_response):
    """Test workflow plan generation returns expected steps."""
    llm_interface.provider.chat = AsyncMock(
        side_effect=[mock_file_pattern_response, mock_workflow_response]
    )
    llm_interface.provider.chat_structured = AsyncMock(
        side_effect=NotImplementedError("structured output not available")
    )
    with patch('glob.glob', return_value=['test1.fastq.gz', 'test2.fastq.gz']):
        result = await llm_interface.generate_workflow_plan("Process RNA-seq data")
        assert result is not None
        assert isinstance(result, dict)
        assert "workflow_type" in result
        assert "steps" in result
        assert len(result["steps"]) > 0
