# Adding Custom Scripts

This guide walks you through the process of adding custom scripts to FlowAgent.

## Step 1: Choose Location

1. Identify the appropriate directory:
   - `rna_seq/` for RNA-seq specific scripts
   - `chip_seq/` for ChIP-seq specific scripts
   - `common/` for scripts usable across workflows

2. Create a new directory with a descriptive name:
   ```bash
   mkdir -p custom_scripts/rna_seq/my_analysis
   ```

## Step 2: Create Script

1. Create your script file (e.g., `my_analysis.R`, `my_analysis.py`)
2. Follow these requirements:

### Input Handling
```python
# Python example
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_name', required=True)
args = parser.parse_args()
```
```R
# R example
args <- commandArgs(trailingOnly = TRUE)
args_dict <- list()
for (i in seq(1, length(args), 2)) {
    args_dict[[sub("^--", "", args[i])]] <- args[i + 1]
}
```

### Output Format
All scripts must output JSON to stdout:
```python
# Python example
import json
print(json.dumps({"output_file": "path/to/output.csv"}))
```
```R
# R example
library(jsonlite)
cat(toJSON(list(output_file = "path/to/output.csv")))
```

## Step 3: Create Metadata

Create `metadata.json` in your script directory:

```json
{
    "name": "my_analysis",
    "description": "Description of what the script does",
    "script_file": "my_analysis.R",
    "language": "R",
    "input_requirements": [
        {
            "name": "input_name",
            "type": "file_type",
            "description": "Description of input"
        }
    ],
    "output_types": [
        {
            "name": "output_name",
            "type": "file_type",
            "description": "Description of output"
        }
    ],
    "workflow_types": ["rna_seq"],
    "execution_order": {
        "before": ["step_names"],
        "after": ["step_names"]
    },
    "requirements": {
        "r_packages": ["required_packages"],
        "python_packages": [],
        "system_dependencies": []
    }
}
```

## Step 4: Test Integration

1. Validate your script:
   ```python
   from flowagent.core.script_manager import ScriptManager
   
   # Initialize script manager
   manager = ScriptManager("path/to/custom_scripts")
   
   # Get your script
   script = manager.get_script("my_analysis")
   
   # Validate requirements
   assert manager.validate_script_requirements(script)
   ```

2. Test execution:
   ```python
   # Execute script
   results = await manager.execute_script(
       script,
       {"input_name": "path/to/input.txt"}
   )
   ```

## Best Practices

1. **Error Handling**
   - Exit with non-zero status on error
   - Write error messages to stderr
   ```python
   import sys
   if error:
       print("Error message", file=sys.stderr)
       sys.exit(1)
   ```

2. **Input Validation**
   - Check input file existence
   - Validate input format
   - Verify required parameters

3. **Output Management**
   - Use consistent output paths
   - Clean up temporary files
   - Document output formats

4. **Testing**
   - Include test data
   - Document test procedures
   - Add validation steps
