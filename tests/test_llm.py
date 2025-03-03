import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
from flowagent.core.llm import LLMInterface

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock()
    with patch('flowagent.core.llm.AsyncOpenAI', return_value=mock_client):
        yield mock_client

@pytest.fixture
def mock_settings():
    """Mock settings module."""
    with patch('flowagent.core.llm.settings') as mock_settings:
        mock_settings.OPENAI_API_KEY = 'test-key'
        mock_settings.OPENAI_MODEL = 'gpt-4'
        mock_settings.OPENAI_FALLBACK_MODEL = 'gpt-3.5-turbo'
        yield mock_settings

@pytest.fixture
def llm_interface(mock_openai_client, mock_settings):
    """Create LLM interface with mocked OpenAI client."""
    with patch('pathlib.Path.exists', return_value=True):
        return LLMInterface()

@pytest.fixture
def mock_openai_response():
    """Create mock OpenAI API response."""
    # First response for file pattern extraction
    file_pattern_response = AsyncMock()
    file_pattern_response.choices = [
        MagicMock(message=MagicMock(
            content=json.dumps({
                "patterns": ["*.fastq.gz"],
                "relationships": {
                    "type": "single",
                    "pattern_groups": [{"pattern": "*.fastq.gz", "group": "reads"}]
                }
            })
        ))
    ]
    
    # Second response for workflow plan generation
    workflow_response = AsyncMock()
    workflow_response.choices = [
        MagicMock(message=MagicMock(
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
            })
        ))
    ]
    
    return [file_pattern_response, workflow_response]

@pytest.mark.asyncio
async def test_generate_workflow_plan(llm_interface, mock_openai_response):
    """Test workflow plan generation."""
    with patch.object(llm_interface, 'client') as mock_client, \
         patch('glob.glob', return_value=['test1.fastq.gz', 'test2.fastq.gz']):
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_openai_response)
        
        result = await llm_interface.generate_workflow_plan("Process RNA-seq data")
        assert result is not None
        assert isinstance(result, dict)
        assert "workflow_type" in result
        assert result["workflow_type"] == "rna_seq"

@pytest.mark.asyncio
async def test_openai_call_mocked(llm_interface, mock_openai_response):
    """Test OpenAI API call with mocked response."""
    llm = llm_interface
    
    mock_response = AsyncMock()
    response_data = {
        "workflow_type": "rna_seq_kallisto",
        "steps": [{"name": "test", "command": "echo test", "dependencies": []}]
    }
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps(response_data)))
    ]
    
    with patch.object(llm, 'client') as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        messages = [{"role": "user", "content": "test prompt"}]
        result = await llm._call_openai(messages)
        parsed_result = json.loads(result)
        assert isinstance(parsed_result, dict)
        assert parsed_result["workflow_type"] == "rna_seq_kallisto"

@pytest.mark.asyncio
async def test_error_handling(llm_interface, mock_openai_response):
    """Test error handling in LLM interface."""
    # Test with invalid prompt type
    with pytest.raises(TypeError):
        await llm_interface.generate_workflow_plan(123)  # type: ignore

    # Test with empty prompt
    with patch.object(llm_interface, 'client') as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=AsyncMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        ))
        with pytest.raises(json.decoder.JSONDecodeError):
            await llm_interface.generate_workflow_plan("")

    # Test with no fastq files
    with patch.object(llm_interface, 'client') as mock_client, \
         patch('glob.glob', return_value=[]):
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response[0])
        with pytest.raises(ValueError, match="No files found matching patterns"):
            await llm_interface.generate_workflow_plan("test prompt")

@pytest.mark.asyncio
async def test_no_matching_files(llm_interface, mock_openai_client):
    """Test error when no files match patterns."""
    # Mock response content
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(
            content=json.dumps({
                "patterns": ["*.fastq.gz"],
                "relationships": {
                    "type": "single",
                    "pattern_groups": [{"pattern": "*.fastq.gz", "group": "reads"}]
                }
            })
        ))
    ]
    mock_openai_client.chat.completions.create.return_value = mock_response

    # Mock glob to return no files
    with patch('glob.glob', return_value=[]):
        with pytest.raises(ValueError, match="No files found matching patterns"):
            await llm_interface.generate_workflow_plan("test prompt")

@pytest.mark.asyncio
async def test_tool_characteristics(llm_interface):
    """Test tool characteristics extraction."""
    llm = llm_interface
    
    # Test tool characteristics
    tool_name = "kallisto"
    with patch.object(llm, '_get_tool_characteristics') as mock_get_chars:
        mock_get_chars.return_value = {
            "memory_weight": 0.5,
            "time_weight": 0.7
        }
        
        characteristics = llm._get_tool_characteristics(tool_name)
        assert isinstance(characteristics, dict)
        assert "memory_weight" in characteristics
        assert "time_weight" in characteristics
        assert all(isinstance(v, float) for v in characteristics.values())

@pytest.mark.asyncio
async def test_generate_workflow_plan(llm_interface, mock_openai_response):
    """Test workflow plan generation."""
    with patch.object(llm_interface, 'client') as mock_client, \
         patch('glob.glob', return_value=['test1.fastq.gz', 'test2.fastq.gz']):
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_openai_response)
        
        result = await llm_interface.generate_workflow_plan("Process RNA-seq data")
        assert result is not None
        assert isinstance(result, dict)
        assert "workflow_type" in result
        assert result["workflow_type"] == "rna_seq"
