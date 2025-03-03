import pytest
import asyncio
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from flowagent.workflow import run_workflow, analyze_workflow
from flowagent.core.workflow_manager import WorkflowManager
from flowagent.core.workflow_dag import WorkflowDAG
from flowagent.core.llm import LLMInterface
import openai

@pytest.fixture
def workflow_manager():
    return WorkflowManager()

@pytest.fixture
def workflow_dag():
    dag = WorkflowDAG()
    dag.graph.clear()  # Clear any existing nodes
    return dag

@pytest.fixture
def test_fastq_files(tmp_path):
    """Create test fastq files."""
    # Create test fastq files
    (tmp_path / "sample1_R1.fastq.gz").touch()
    (tmp_path / "sample1_R2.fastq.gz").touch()
    return tmp_path

@pytest.mark.asyncio
async def test_workflow_dag_creation():
    """Test creation of workflow DAG with steps."""
    dag = WorkflowDAG()
    dag.graph.clear()  # Clear any existing nodes
    
    # Add test steps
    step1 = {
        "name": "step1",
        "command": "echo 'test1'",
        "status": "pending"
    }
    step2 = {
        "name": "step2",
        "command": "echo 'test2'",
        "status": "pending"
    }
    
    # Add steps to DAG
    dag.add_step(step1)
    dag.add_step(step2, dependencies=["step1"])
    
    # Verify DAG structure
    assert "step1" in dag.graph.nodes
    assert "step2" in dag.graph.nodes
    assert len(dag.graph.edges) == 1
    assert list(dag.graph.predecessors("step2")) == ["step1"]

@pytest.mark.asyncio
async def test_workflow_dag_visualization(tmp_path):
    """Test workflow DAG visualization."""
    dag = WorkflowDAG()
    dag.graph.clear()  # Clear any existing nodes
    
    # Add test steps with different statuses
    steps = [
        {
            "name": "start",
            "command": "echo start",
            "status": "completed"
        },
        {
            "name": "process",
            "command": "echo process",
            "status": "failed"
        },
        {
            "name": "end",
            "command": "echo end",
            "status": "pending"
        }
    ]
    
    # Add steps to DAG with dependencies
    dag.add_step(steps[0])
    dag.add_step(steps[1], dependencies=["start"])
    dag.add_step(steps[2], dependencies=["process"])
    
    # Test visualization
    output_file = tmp_path / "workflow.png"
    dag.visualize(output_file)
    assert output_file.exists()

@pytest.mark.asyncio
async def test_workflow_execution():
    """Test workflow execution."""
    # Mock OpenAI API response
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Here's the workflow plan:\n1. Start with quality control\n2. Perform alignment\n3. Generate counts"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }
    
    with patch("openai.AsyncClient") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        workflow = WorkflowDAG()
        
        # Add test steps
        steps = [
            {
                "name": "qc",
                "command": "echo qc",
                "status": "pending"
            },
            {
                "name": "align",
                "command": "echo align",
                "status": "pending"
            }
        ]
        
        workflow.add_step(steps[0])
        workflow.add_step(steps[1], dependencies=["qc"])
        
        # Execute workflow
        await workflow.execute_parallel()
        
        # Verify execution
        assert workflow.graph.nodes["qc"]["step"]["status"] == "completed"
        assert workflow.graph.nodes["align"]["step"]["status"] == "completed"

@pytest.mark.asyncio
async def test_workflow_error_handling():
    """Test workflow error handling."""
    # Mock OpenAI API error
    mock_error_response = {
        "error": {
            "message": "Invalid API key",
            "type": "invalid_request_error",
            "code": "invalid_api_key"
        }
    }
    mock_error = openai.APIStatusError(
        message="Invalid API key",
        response=MagicMock(status=401, text=json.dumps(mock_error_response)),
        body=mock_error_response
    )
    
    with patch("openai.AsyncClient") as mock_client:
        mock_client.return_value.chat.completions.create.side_effect = mock_error
        workflow = WorkflowDAG()
        
        # Add test step that will fail
        step = {
            "name": "fail",
            "command": "nonexistent_command",
            "status": "pending"
        }
        workflow.add_step(step)
        
        # Execute workflow and expect error result
        result = await workflow.execute_parallel()
        assert result["status"] == "failed"
        assert "nonexistent_command: not found" in result["error"]
        assert workflow.graph.nodes["fail"]["step"]["status"] == "failed"

@pytest.mark.asyncio
async def test_workflow_execution_with_fastq_files(tmp_path, test_fastq_files):
    """Test basic workflow execution."""
    # Create test files
    os.makedirs(test_fastq_files, exist_ok=True)
    test_file = test_fastq_files / "test.fastq.gz"
    test_file.touch()
    
    # Mock OpenAI API response
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Here's the workflow plan:\n1. Create output directories\n2. Run Kallisto\n3. Generate report"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }
    
    # Mock OpenAI API
    with patch("openai.AsyncClient") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_response
        
        workflow = WorkflowDAG()
        
        # Add test steps
        steps = [
            {
                "name": "create_dirs",
                "command": "mkdir -p output",
                "status": "pending"
            },
            {
                "name": "kallisto",
                "command": "echo kallisto",
                "status": "pending"
            },
            {
                "name": "report",
                "command": "echo report",
                "status": "pending"
            }
        ]
        
        # Add steps with dependencies
        workflow.add_step(steps[0])
        workflow.add_step(steps[1], dependencies=["create_dirs"])
        workflow.add_step(steps[2], dependencies=["kallisto"])
        
        # Execute workflow
        await workflow.execute_parallel()
        
        # Verify execution
        assert workflow.graph.nodes["create_dirs"]["step"]["status"] == "completed"
        assert workflow.graph.nodes["kallisto"]["step"]["status"] == "completed"
        assert workflow.graph.nodes["report"]["step"]["status"] == "completed"
