# Data Cleanup Example

This example demonstrates a universal data cleanup script that can be used across different workflow types.

## Script Overview

The script performs common data cleanup tasks such as:
- Removing temporary files
- Organizing output directories
- Compressing intermediate files
- Validating file integrity

## Implementation

```bash
#!/bin/bash

# data_cleanup.sh

# Parse JSON arguments
ARGS="$1"
WORKFLOW_DIR=$(echo "$ARGS" | jq -r '.workflow_dir')
COMPRESS_INTERMEDIATES=$(echo "$ARGS" | jq -r '.compress_intermediates')

# Function to compress intermediate files
compress_intermediates() {
    local dir="$1"
    find "$dir" -type f \( -name "*.sam" -o -name "*.bed" \) -exec gzip {} \;
}

# Function to validate file integrity
validate_files() {
    local dir="$1"
    local invalid_files=()
    
    # Check BAM files
    while IFS= read -r file; do
        if ! samtools quickcheck "$file"; then
            invalid_files+=("$file")
        fi
    done < <(find "$dir" -name "*.bam")
    
    # Check FASTQ files
    while IFS= read -r file; do
        if ! zcat "$file" | head -n 4 | grep -q "^@"; then
            invalid_files+=("$file")
        fi
    done < <(find "$dir" -name "*.fastq.gz")
    
    echo "${invalid_files[@]}"
}

# Clean temporary files
find "$WORKFLOW_DIR" -type f -name "*.tmp" -delete

# Organize output directories
mkdir -p "$WORKFLOW_DIR"/{fastqc,alignment,analysis,reports}

# Compress intermediate files if requested
if [ "$COMPRESS_INTERMEDIATES" = "true" ]; then
    compress_intermediates "$WORKFLOW_DIR"
fi

# Validate files
invalid_files=$(validate_files "$WORKFLOW_DIR")

# Generate cleanup report
cat > cleanup_report.json << EOF
{
    "cleaned_temp_files": $(find "$WORKFLOW_DIR" -type f -name "*.tmp" | wc -l),
    "compressed_files": $(find "$WORKFLOW_DIR" -type f -name "*.gz" | wc -l),
    "invalid_files": ["${invalid_files[@]}"]
}
EOF

# Output results
echo "{\"cleanup_report\": \"cleanup_report.json\"}"
```

## Metadata

```json
{
    "name": "data_cleanup",
    "description": "Universal data cleanup and organization script",
    "script_file": "data_cleanup.sh",
    "language": "bash",
    "input_requirements": [
        {
            "name": "workflow_dir",
            "type": "string",
            "description": "Directory containing workflow data"
        },
        {
            "name": "compress_intermediates",
            "type": "boolean",
            "description": "Whether to compress intermediate files"
        }
    ],
    "output_types": [
        {
            "name": "cleanup_report",
            "type": "file",
            "description": "JSON report of cleanup operations"
        }
    ],
    "workflow_types": ["rna_seq", "chip_seq", "common"],
    "execution_order": {
        "after": ["all"]
    },
    "requirements": {
        "system_dependencies": ["jq", "samtools"]
    }
}
```

## Usage

```python
from flowagent.core.workflow_executor import WorkflowExecutor

# Initialize workflow
executor = WorkflowExecutor(llm_interface)

# Execute cleanup
results = await executor.execute_workflow(
    input_data={
        "workflow_dir": "/path/to/workflow",
        "compress_intermediates": True
    },
    workflow_type="common",
    custom_script_requests=["data_cleanup"]
)

# Access cleanup report
with open(results["cleanup_report"]) as f:
    cleanup_report = json.load(f)
```

## Output Format

The cleanup report JSON contains:
- Number of temporary files cleaned
- Number of files compressed
- List of invalid files found

## Features

1. **File Organization**
   - Creates standard directories
   - Moves files to appropriate locations
   - Maintains consistent structure

2. **Space Optimization**
   - Removes temporary files
   - Compresses intermediate files
   - Archives old results

3. **Data Validation**
   - Checks file integrity
   - Validates file formats
   - Reports corrupted files

## Best Practices

1. **Safety**
   - Never delete original data
   - Validate before compression
   - Keep cleanup logs

2. **Efficiency**
   - Use parallel compression
   - Prioritize large files
   - Monitor disk usage

3. **Organization**
   - Follow naming conventions
   - Maintain directory structure
   - Document changes
