# Script Requirements

This page details the requirements for custom scripts in FlowAgent.

## Basic Requirements

1. **Script Format**
   - Must be executable
   - Must handle command-line arguments
   - Must output JSON to stdout
   - Must use stderr for errors

2. **Metadata**
   - Must have accompanying metadata.json
   - Must specify all inputs and outputs
   - Must declare workflow position

3. **Error Handling**
   - Must exit with non-zero status on error
   - Must provide meaningful error messages

## Metadata Schema

The `metadata.json` file must follow this schema:

```json
{
    "name": "string",              // Unique identifier
    "description": "string",       // Clear description
    "script_file": "string",       // Script filename
    "language": "string",          // Programming language
    "input_requirements": [        // List of inputs
        {
            "name": "string",      // Input parameter name
            "type": "string",      // File type/format
            "description": "string" // Input description
        }
    ],
    "output_types": [             // List of outputs
        {
            "name": "string",      // Output name
            "type": "string",      // File type/format
            "description": "string" // Output description
        }
    ],
    "workflow_types": [           // Compatible workflows
        "string"
    ],
    "execution_order": {          // Position in workflow
        "before": ["string"],     // Steps that come after
        "after": ["string"]       // Steps that come before
    },
    "requirements": {             // Dependencies
        "r_packages": ["string"], // Required R packages
        "python_packages": ["string"], // Required Python packages
        "system_dependencies": ["string"] // Required system tools
    }
}
```

## Language-Specific Requirements

### Python Scripts

1. **Argument Parsing**
   ```python
   import argparse
   
   parser = argparse.ArgumentParser()
   parser.add_argument('--input_name', required=True)
   args = parser.parse_args()
   ```

2. **JSON Output**
   ```python
   import json
   
   results = {"output_file": "path/to/output.csv"}
   print(json.dumps(results))
   ```

### R Scripts

1. **Argument Parsing**
   ```R
   args <- commandArgs(trailingOnly = TRUE)
   args_dict <- list()
   for (i in seq(1, length(args), 2)) {
       args_dict[[sub("^--", "", args[i])]] <- args[i + 1]
   }
   ```

2. **JSON Output**
   ```R
   library(jsonlite)
   
   results <- list(output_file = "path/to/output.csv")
   cat(toJSON(results))
   ```

### Bash Scripts

1. **Argument Parsing**
   ```bash
   while [[ $# -gt 0 ]]; do
       case $1 in
           --input_file)
               INPUT_FILE="$2"
               shift 2
               ;;
           *)
               shift
               ;;
       esac
   done
   ```

2. **JSON Output**
   ```bash
   echo "{\"output_file\": \"$OUTPUT_FILE\"}"
   ```

## Standard Workflow Steps

Scripts can reference these standard steps in their execution order:

### RNA-seq Workflow
- fastqc
- alignment
- feature_counts
- differential_expression

### ChIP-seq Workflow
- fastqc
- alignment
- peak_calling
- motif_analysis

## Best Practices

1. **Input Validation**
   - Check file existence
   - Validate file formats
   - Verify parameter values

2. **Output Management**
   - Use consistent naming
   - Clean up temporary files
   - Document file formats

3. **Error Messages**
   - Be specific and clear
   - Include troubleshooting hints
   - Log relevant details

4. **Performance**
   - Handle large files efficiently
   - Clean up resources
   - Report progress for long operations

5. **Documentation**
   - Include usage examples
   - Document assumptions
   - Explain algorithms used
