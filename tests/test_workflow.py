import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from flowagent.core.workflow_manager import WorkflowManager
from flowagent.core.workflow_dag import WorkflowDAG


@pytest.fixture
def workflow_manager():
    return WorkflowManager()


@pytest.fixture
def workflow_dag():
    dag = WorkflowDAG()
    dag.graph.clear()
    return dag


@pytest.fixture
def test_fastq_files(tmp_path):
    """Create test fastq files."""
    (tmp_path / "sample1_R1.fastq.gz").touch()
    (tmp_path / "sample1_R2.fastq.gz").touch()
    return tmp_path


@pytest.mark.asyncio
async def test_workflow_dag_creation():
    """Test creation of workflow DAG with steps."""
    dag = WorkflowDAG()
    dag.graph.clear()

    step1 = {"name": "step1", "command": "echo 'test1'", "status": "pending"}
    step2 = {"name": "step2", "command": "echo 'test2'", "status": "pending"}

    dag.add_step(step1)
    dag.add_step(step2, dependencies=["step1"])

    assert "step1" in dag.graph.nodes
    assert "step2" in dag.graph.nodes
    assert len(dag.graph.edges) == 1
    assert list(dag.graph.predecessors("step2")) == ["step1"]


@pytest.mark.asyncio
async def test_workflow_dag_visualization(tmp_path):
    """Test workflow DAG visualization."""
    dag = WorkflowDAG()
    dag.graph.clear()

    steps = [
        {"name": "start", "command": "echo start", "status": "completed"},
        {"name": "process", "command": "echo process", "status": "failed"},
        {"name": "end", "command": "echo end", "status": "pending"},
    ]

    dag.add_step(steps[0])
    dag.add_step(steps[1], dependencies=["start"])
    dag.add_step(steps[2], dependencies=["process"])

    output_file = tmp_path / "workflow.png"
    dag.visualize(output_file)
    assert output_file.exists()


@pytest.mark.asyncio
async def test_workflow_execution():
    """Test parallel execution of DAG steps using the local executor."""
    workflow = WorkflowDAG()

    steps = [
        {"name": "qc", "command": "echo qc", "status": "pending"},
        {"name": "align", "command": "echo align", "status": "pending"},
    ]

    workflow.add_step(steps[0])
    workflow.add_step(steps[1], dependencies=["qc"])

    await workflow.execute_parallel()

    assert workflow.graph.nodes["qc"]["step"]["status"] == "completed"
    assert workflow.graph.nodes["align"]["step"]["status"] == "completed"


@pytest.mark.asyncio
async def test_workflow_error_handling():
    """Test workflow error handling with a failing command."""
    workflow = WorkflowDAG()

    step = {"name": "fail", "command": "nonexistent_command", "status": "pending"}
    workflow.add_step(step)

    result = await workflow.execute_parallel()
    assert result["status"] == "failed"
    assert "nonexistent_command" in result["error"]
    assert "not found" in result["error"].lower()
    assert workflow.graph.nodes["fail"]["step"]["status"] == "failed"


@pytest.mark.asyncio
async def test_workflow_execution_with_fastq_files(tmp_path, test_fastq_files):
    """Test workflow execution with real echo commands."""
    os.makedirs(test_fastq_files, exist_ok=True)
    test_file = test_fastq_files / "test.fastq.gz"
    test_file.touch()

    workflow = WorkflowDAG()

    steps = [
        {"name": "create_dirs", "command": "mkdir -p output", "status": "pending"},
        {"name": "kallisto", "command": "echo kallisto", "status": "pending"},
        {"name": "report", "command": "echo report", "status": "pending"},
    ]

    workflow.add_step(steps[0])
    workflow.add_step(steps[1], dependencies=["create_dirs"])
    workflow.add_step(steps[2], dependencies=["kallisto"])

    await workflow.execute_parallel()

    assert workflow.graph.nodes["create_dirs"]["step"]["status"] == "completed"
    assert workflow.graph.nodes["kallisto"]["step"]["status"] == "completed"
    assert workflow.graph.nodes["report"]["step"]["status"] == "completed"
