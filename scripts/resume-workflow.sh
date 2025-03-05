#!/bin/bash
# Resume a FlowAgent workflow from the point where outputs are missing

# Display usage information
display_usage() {
    echo "Usage: $0 <prompt> [checkpoint_dir]"
    echo "  <prompt>         The workflow prompt to execute"
    echo "  [checkpoint_dir] Optional: The checkpoint directory (default: workflow_state)"
    echo ""
    echo "Example: $0 \"Analyze RNA-seq data using Kallisto\" workflow_state"
    exit 1
}

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    display_usage
fi

PROMPT="$1"
CHECKPOINT_DIR="${2:-workflow_state}"

# Run the workflow with smart resume
echo "Resuming workflow from checkpoint directory: $CHECKPOINT_DIR"
echo "Prompt: $PROMPT"
echo ""
flowagent prompt "$PROMPT" --checkpoint-dir "$CHECKPOINT_DIR"
