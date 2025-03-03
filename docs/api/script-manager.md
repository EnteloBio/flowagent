# Script Manager API

The Script Manager is responsible for discovering, validating, and executing custom scripts.

## Class: ScriptManager

```python
class ScriptManager:
    """Manages custom workflow scripts."""
    
    def __init__(self, custom_scripts_dir: str):
        """Initialize script manager.
        
        Args:
            custom_scripts_dir: Path to custom scripts directory
        """
```

### Methods

#### get_script
```python
def get_script(self, name: str) -> Optional[ScriptInfo]:
    """Get script info by name.
    
    Args:
        name: Name of the script to retrieve
        
    Returns:
        ScriptInfo object if found, None otherwise
    """
```

#### get_scripts_for_workflow
```python
def get_scripts_for_workflow(self, workflow_type: str) -> List[ScriptInfo]:
    """Get all scripts compatible with a workflow type.
    
    Args:
        workflow_type: Type of workflow (e.g., "rna_seq", "chip_seq")
        
    Returns:
        List of compatible ScriptInfo objects
    """
```

#### validate_script_requirements
```python
def validate_script_requirements(self, script: ScriptInfo) -> bool:
    """Validate that all script requirements are met.
    
    Args:
        script: ScriptInfo object to validate
        
    Returns:
        True if all requirements are met, False otherwise
    """
```

#### execute_script
```python
async def execute_script(self, 
                        script: ScriptInfo, 
                        inputs: Dict[str, str]) -> Dict[str, Any]:
    """Execute a custom script.
    
    Args:
        script: ScriptInfo object to execute
        inputs: Dictionary mapping input names to file paths
        
    Returns:
        Dictionary containing output file paths
        
    Raises:
        ValueError: If required inputs are missing
        RuntimeError: If script execution fails
    """
```

## Class: ScriptInfo

```python
@dataclass
class ScriptInfo:
    """Information about a custom script."""
    name: str
    description: str
    script_file: str
    language: str
    input_requirements: List[Dict[str, str]]
    output_types: List[Dict[str, str]]
    workflow_types: List[str]
    execution_order: Dict[str, List[str]]
    requirements: Dict[str, List[str]]
    path: Path
```

### Usage Example

```python
from flowagent.core.script_manager import ScriptManager

# Initialize manager
manager = ScriptManager("path/to/custom_scripts")

# Get script for workflow
scripts = manager.get_scripts_for_workflow("rna_seq")

# Execute specific script
script = manager.get_script("deseq2_normalize")
if script and manager.validate_script_requirements(script):
    results = await manager.execute_script(
        script,
        {"counts_matrix": "path/to/counts.csv"}
    )
```

## Error Handling

The Script Manager provides detailed error information:

1. **Validation Errors**
   - Missing requirements
   - Incompatible versions
   - Missing dependencies

2. **Execution Errors**
   - Missing inputs
   - Script runtime errors
   - Output format errors

Example error handling:
```python
try:
    results = await manager.execute_script(script, inputs)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Execution failed: {e}")
```
