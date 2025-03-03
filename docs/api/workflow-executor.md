# Workflow Executor API

The Workflow Executor manages the execution of workflows, integrating both standard steps and custom scripts.

## Class: WorkflowExecutor

```python
class WorkflowExecutor:
    """Executes workflows with integrated custom scripts."""
    
    def __init__(self, llm_interface: LLMInterface):
        """Initialize workflow executor.
        
        Args:
            llm_interface: LLM interface for workflow customization
        """
```

### Methods

#### execute_workflow
```python
async def execute_workflow(self,
                         input_data: Dict[str, str],
                         workflow_type: str,
                         custom_script_requests: Optional[List[str]] = None) -> Dict[str, Any]:
    """Execute workflow with custom scripts.
    
    Args:
        input_data: Dictionary of input file paths
        workflow_type: Type of workflow (e.g., "rna_seq", "chip_seq")
        custom_script_requests: Optional list of custom scripts to include
        
    Returns:
        Dictionary containing workflow results
    """
```

## Class: WorkflowStep

```python
@dataclass
class WorkflowStep:
    """Represents a step in the workflow."""
    name: str
    type: str  # 'standard' or 'custom'
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    script_info: Optional[ScriptInfo] = None
```

### Usage Example

```python
from flowagent.core.workflow_executor import WorkflowExecutor
from flowagent.core.llm import LLMInterface

# Initialize executor
executor = WorkflowExecutor(llm_interface)

# Execute workflow with custom scripts
results = await executor.execute_workflow(
    input_data={
        "fastq": "path/to/input.fastq",
        "annotation": "path/to/annotation.gtf"
    },
    workflow_type="rna_seq",
    custom_script_requests=["deseq2_normalize"]
)
```

## Workflow Types

### RNA-seq Workflow
Standard steps:
1. fastqc
2. alignment
3. feature_counts
4. differential_expression

Example with custom script:
```python
# RNA-seq workflow with normalization
workflow = await executor.execute_workflow(
    input_data={"fastq": "input.fastq"},
    workflow_type="rna_seq",
    custom_script_requests=["deseq2_normalize"]
)
```

### ChIP-seq Workflow
Standard steps:
1. fastqc
2. alignment
3. peak_calling
4. motif_analysis

Example with custom script:
```python
# ChIP-seq workflow with custom peak analysis
workflow = await executor.execute_workflow(
    input_data={"fastq": "input.fastq"},
    workflow_type="chip_seq",
    custom_script_requests=["custom_peak_analysis"]
)
```

## Error Handling

The Workflow Executor provides comprehensive error handling:

1. **Workflow Errors**
   - Invalid workflow type
   - Missing required inputs
   - Step execution failures

2. **Custom Script Errors**
   - Script not found
   - Requirements not met
   - Execution failures

Example error handling:
```python
try:
    results = await executor.execute_workflow(
        input_data,
        workflow_type,
        custom_scripts
    )
except ValueError as e:
    print(f"Invalid workflow configuration: {e}")
except RuntimeError as e:
    print(f"Workflow execution failed: {e}")
```

## Best Practices

1. **Input Validation**
   ```python
   # Validate inputs before execution
   if not all(os.path.exists(path) for path in input_data.values()):
       raise ValueError("Input files not found")
   ```

2. **Custom Script Integration**
   ```python
   # Request specific custom scripts
   workflow = await executor.execute_workflow(
       input_data=data,
       workflow_type="rna_seq",
       custom_script_requests=["script1", "script2"]
   )
   ```

3. **Result Handling**
   ```python
   # Process workflow results
   results = await executor.execute_workflow(...)
   for step, outputs in results.items():
       print(f"Step {step} outputs: {outputs}")
   ```
