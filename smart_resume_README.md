# FlowAgent Smart Resume Feature

This feature allows FlowAgent to detect which steps of a workflow have been completed and only rerun the steps where output is missing.

## How It Works

1. **Output Detection**: The system checks each step's expected output files or directories
2. **Tool-Specific Validation**: Different tools have specific validators for their outputs
3. **Generic Fallback**: For unregistered tools, a generic validator checks common output patterns
4. **Step Filtering**: Steps that have already produced their expected outputs are skipped
5. **Smart Resumption**: The workflow continues from where it left off

## Using Smart Resume

### Command Line

```bash
# Run a workflow with smart resume
flowagent prompt "Analyze RNA-seq data using Kallisto" --checkpoint-dir workflow_state
```

### Helper Script

A helper script is provided for convenience:

```bash
# Make the script executable (if needed)
chmod +x scripts/resume-workflow.sh

# Run the script with your workflow prompt
scripts/resume-workflow.sh "Analyze RNA-seq data using Kallisto" workflow_state
```

## Supported Tools and Steps

The smart resume feature includes tool-specific validators for:

- **RNA-seq Analysis**:
  - **Kallisto**: Index and quantification steps
  - **FastQC**: Quality control reports
  - **MultiQC**: Aggregated QC reports

- **DNA-seq Analysis**:
  - **BWA**: Index and alignment
  - **Samtools**: Sorting, indexing, and other operations
  - **BCFtools**: Variant calling

- **General Bioinformatics**:
  - **Bowtie2**: Index building

- **Generic Support**:
  For tools without specific validators, the system will check:
  - Output files specified with common flags (-o, --output, --outdir, etc.)
  - Files with common bioinformatics extensions (.bam, .vcf, .tsv, etc.)
  - Directory creation (mkdir commands)

## Extending with Custom Validators

You can add support for additional tools by registering custom validators:

```python
from flowagent.core.smart_resume import register_custom_validator

# Define a custom validator for a tool
def my_custom_tool_validator(command: str, step: dict) -> bool:
    """Check if my_custom_tool outputs exist."""
    # Implementation specific to your tool's output format
    output_match = re.search(r'-o\s+([^\s]+)', command)
    if output_match:
        output_file = output_match.group(1)
        return os.path.isfile(output_file) and os.path.getsize(output_file) > 0
    return False

# Register the validator with a regex pattern that matches the tool's command
register_custom_validator(r'my_custom_tool\b', my_custom_tool_validator)
```

## Implementation Details

The implementation includes:

1. **Smart Resume Module**: `flowagent/core/smart_resume.py` contains the core functionality:
   - Tool validator registry system
   - Output path detection
   - File and directory existence checking

2. **Workflow Manager Integration**: Automatically detects and skips completed steps

3. **Dependency Handling**: The workflow DAG ensures dependencies are respected

## Troubleshooting

If a step should be rerun but is being skipped:

1. Delete the output directory/files for that specific step
2. Run the workflow again with the same checkpoint directory

If the entire workflow is being rerun:

1. Check that your checkpoint directory is correctly specified
2. Verify that the workflow's output patterns match what the detection is looking for
3. Check the log output for details on why steps were not detected as completed

## Debugging

To see which steps are being detected as completed and why, check the log output. You can increase verbosity with:

```bash
export FLOWAGENT_LOG_LEVEL=DEBUG
```

This will show details about which validators are being used for each step and why steps are detected as completed or not.
