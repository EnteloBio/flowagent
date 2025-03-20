# SLURM Configuration for FlowAgent

This guide explains how to configure FlowAgent to work with SLURM-based High-Performance Computing (HPC) environments.

## Overview

FlowAgent supports executing workflows on SLURM clusters through the `CGATExecutor`. This integration leverages the CGATCore pipeline system to manage job submission, monitoring, and resource allocation.

## Configuration Options

### Environment Variables

FlowAgent uses the following environment variables for SLURM configuration:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `SLURM_QUEUE` | Default SLURM queue/partition | `short` |
| `SLURM_DEFAULT_MEMORY` | Default memory allocation per job | `4G` |
| `SLURM_DEFAULT_CPUS` | Default CPU cores per job | `1` |
| `SLURM_ACCOUNT` | SLURM account name | `""` |
| `SLURM_PARTITION` | SLURM partition name | `""` |
| `SLURM_QOS` | Quality of Service level | `""` |
| `SLURM_MAIL_USER` | Email for job notifications | `""` |
| `SLURM_MAIL_TYPE` | When to send email notifications | `ALL` |

You can set these in your `.env` file or directly in your environment.

### CGAT Configuration File

FlowAgent uses a `.cgat.yml` configuration file for more detailed SLURM settings. This file should be placed in your project's root directory.

Here's a sample configuration:

```yaml
# FlowAgent CGAT/SLURM Configuration

cluster:
  queue_manager: slurm
  queue: medium
  parallel_environment: smp

slurm:
  account: my_lab_account
  partition: compute
  mail_user: user@example.com
  mail_type: ALL
  qos: normal

# Tool-specific resource requirements
tools:
  kallisto_index:
    memory: 16G
    threads: 8
    queue: short
  star_align:
    memory: 32G
    threads: 16
    queue: long
  samtools_sort:
    memory: 8G
    threads: 4
    queue: medium
```

If this file doesn't exist, FlowAgent will create a default one based on your environment variables.

## LLM-Driven Workflow Generation

FlowAgent uses its LLM capabilities to automatically generate workflows based on user requirements. The LLM:

1. **Analyzes user input** to determine the appropriate analysis steps
2. **Designs an optimal workflow** with appropriate dependencies
3. **Allocates resources** based on tool requirements and data characteristics
4. **Submits jobs** to the SLURM cluster with appropriate parameters
5. **Monitors execution** and handles errors intelligently

For more details on how the LLM generates and manages workflows, see [LLM-Driven SLURM Integration](llm_slurm_integration.md).

## Workflow Structure

The LLM generates workflows with this structure:

```json
{
  "steps": [
    {
      "name": "step_name",
      "command": "command_to_execute",
      "resources": {
        "memory": "32G",
        "cpus": 8,
        "time_min": 120,
        "queue": "long"
      },
      "dependencies": []
    }
  ]
}
```

Each step includes:

- **name**: Unique identifier for the step
- **command**: The command to execute
- **resources**: Resource requirements (memory, CPUs, time limit, queue)
- **dependencies**: List of steps that must complete before this step runs

## Job Monitoring and Error Handling

FlowAgent provides robust job monitoring and error handling for SLURM jobs:

1. **Job Scripts**: All commands are written to script files in `logs/slurm_scripts/` for reproducibility
2. **Job Information**: Detailed job information is stored in `logs/job_info/` as JSON files
3. **Error Parsing**: Common SLURM errors are parsed and translated into user-friendly messages
4. **Resource Usage**: Job statistics are collected using `sacct` for performance analysis

## Common SLURM Errors and Solutions

| Error | Possible Solution |
|-------|-------------------|
| Job exceeded memory limit | Increase the memory allocation in your workflow step |
| Job exceeded time limit | Increase the time allocation or optimize your code |
| Invalid account/partition | Check your SLURM account and partition settings |
| Permission denied | Ensure you have the correct permissions for the SLURM account/partition |
| Requested node configuration not available | Reduce resource requirements or use a different partition |
| Unable to allocate resources | The cluster may be busy; try reducing resource requirements |
| Invalid QoS | Check your SLURM_QOS setting |

## Best Practices

1. **Resource Estimation**: Start with conservative resource estimates and adjust based on job statistics
2. **Dependencies**: Use the dependencies list to create efficient workflow DAGs
3. **Error Handling**: Check job output and error files for troubleshooting
4. **Module Loading**: Use the script template to load required environment modules
5. **Partition Selection**: Choose appropriate partitions based on job requirements

## Interacting with FlowAgent for SLURM Execution

Instead of writing code, you interact with FlowAgent using natural language:

```
User: "I need to analyze RNA-seq data from tumor and control samples using STAR and DESeq2."

FlowAgent: "I'll create a workflow for RNA-seq analysis using STAR and DESeq2. 
I'll configure it to run on your SLURM cluster with appropriate resource allocations.
Can you provide the location of your FASTQ files and reference genome?"
```

The LLM will handle all the technical details of configuring and submitting SLURM jobs.

## Troubleshooting

If you encounter issues with SLURM job submission:

1. Check if the `.cgat.yml` file is correctly configured
2. Verify that your SLURM account and partition are valid
3. Examine the job script files in `logs/slurm_scripts/`
4. Look at the job information files in `logs/job_info/`
5. Check SLURM output files (`slurm-*.out` and `slurm-*.err`)
6. Run `squeue -u $USER` to see if your jobs are in the queue
7. Run `sacct -u $USER` to see the status of completed jobs

## Advanced Configuration

For more advanced SLURM configurations, you can modify the `.cgat.yml` file or set additional environment variables. The LLM will automatically incorporate these settings when generating workflows.
