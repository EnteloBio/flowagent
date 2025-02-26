import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from cognomic.core.llm import LLMInterface

@pytest.fixture
def llm_interface():
    return LLMInterface()

@pytest.mark.asyncio
async def test_generate_workflow_plan():
    """Test workflow plan generation."""
    llm = LLMInterface()
    
    mock_response = AsyncMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "workflow_type": "rna_seq_kallisto",
            "steps": [{"name": "test", "command": "echo test", "dependencies": []}]
        })))
    ]
    
    with patch('glob.glob', return_value=['test1.fastq.gz', 'test2.fastq.gz']):
        with patch.object(llm, '_detect_workflow_type') as mock_detect:
            mock_detect.return_value = ("rna_seq_kallisto", {
                "dir_structure": ["analysis_output"],
                "rules": {}
            })
            with patch.object(llm, 'client') as mock_client:
                mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
                
                result = await llm.generate_workflow_plan("test prompt")
                assert result is not None
                assert result["workflow_type"] == "rna_seq_kallisto"
                assert len(result["steps"]) == 1

@pytest.mark.asyncio
async def test_openai_call_mocked():
    """Test OpenAI API call with mocked response."""
    llm = LLMInterface()
    
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
async def test_error_handling():
    """Test error handling in LLM interface."""
    llm = LLMInterface()
    
    # Test with invalid prompt type
    with patch('glob.glob', return_value=['test1.fastq.gz']):
        with patch.object(llm, '_detect_workflow_type') as mock_detect:
            mock_detect.side_effect = TypeError("Prompt must be a string")
            with pytest.raises(TypeError, match="Prompt must be a string"):
                await llm.generate_workflow_plan(123)  # type: ignore
    
    # Test with empty prompt
    with patch('glob.glob', return_value=['test1.fastq.gz']):
        with patch.object(llm, '_detect_workflow_type') as mock_detect:
            mock_detect.side_effect = ValueError("Empty workflow prompt")
            with pytest.raises(ValueError, match="Empty workflow prompt"):
                await llm.generate_workflow_plan("")
    
    # Test with no fastq files
    with patch('glob.glob', return_value=[]):
        with patch.object(llm, '_detect_workflow_type') as mock_detect:
            mock_detect.side_effect = ValueError("No .fastq.gz files found in current directory")
            with pytest.raises(ValueError, match="No .fastq.gz files found in current directory"):
                await llm.generate_workflow_plan("test prompt")

@pytest.mark.asyncio
async def test_tool_characteristics():
    """Test tool characteristics extraction."""
    llm = LLMInterface()
    
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
