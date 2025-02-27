# Workflow State Management

FlowAgent includes a robust checkpointing system that helps manage long-running RNA-seq analysis workflows. This system allows you to resume interrupted workflows and avoid repeating expensive computations.

## Using Checkpoints

### Basic Usage:
```bash
# Run workflow with checkpointing
flowagent prompt "Analyze RNA-seq data..." --checkpoint-dir workflow_state
```

### Resuming Interrupted Workflows:
```bash
# Resume from last successful checkpoint
flowagent prompt "Analyze RNA-seq data..." --checkpoint-dir workflow_state --resume
```

## How It Works

The checkpoint directory (e.g., `workflow_state`) stores:
- Progress tracking for each workflow step
- Intermediate computation results
- Error logs and debugging information
- Workflow configuration and parameters

This allows FlowAgent to:
- Resume workflows from the last successful step
- Avoid recomputing expensive operations
- Maintain workflow state across system restarts
- Track errors and provide detailed debugging information

## Best Practices

### Choose Descriptive Directory Names:
```bash
# Use meaningful names for different analyses
flowagent prompt "..." --checkpoint-dir rnaseq_liver_samples_20250225
```

### Backup Checkpoint Directories:
- Keep checkpoint directories for reproducibility
- Back up important checkpoints before rerunning analyses
- Use different checkpoint directories for different analyses

### Debugging Using Checkpoints:
- Examine checkpoint directory contents for troubleshooting
- Use `--resume` to retry failed steps without restarting
- Check error logs in checkpoint directory for detailed information
