# HPC Configuration

FlowAgent supports High-Performance Computing (HPC) execution, with built-in support for SLURM, SGE, and TORQUE systems. The HPC settings can be configured through environment variables or in your `.env` file.

## Basic HPC Settings

```bash
# HPC Configuration
EXECUTOR_TYPE=hpc           # Use HPC executor instead of local
HPC_SYSTEM=slurm           # Options: slurm, sge, torque
HPC_QUEUE=all.q            # Your HPC queue name
HPC_DEFAULT_MEMORY=4G      # Default memory allocation
HPC_DEFAULT_CPUS=1         # Default CPU cores
HPC_DEFAULT_TIME=60        # Default time limit in minutes
```

## Resource Management

FlowAgent automatically manages HPC resources with sensible defaults that can be overridden:

### Memory Management
- Default: 4GB per job
- Override with `HPC_DEFAULT_MEMORY`
- Supports standard memory units (G, M, K)

### CPU Allocation
- Default: 1 CPU per job
- Override with `HPC_DEFAULT_CPUS`
- Automatically scales based on task requirements

### Queue Selection
- Default queue: "all.q"
- Override with `HPC_QUEUE`
- Queue-specific resource limits are respected

## Using HPC Execution

To run a workflow on your HPC system:

```bash
# Basic execution
flowagent "Your workflow description" --executor hpc

# Specify custom resource requirements
flowagent "Your workflow description" --executor hpc --memory 32G --threads 16
```

The system will automatically:
- Submit jobs to the appropriate queue
- Handle job dependencies
- Manage resource allocation
- Monitor job status
- Provide detailed logging
