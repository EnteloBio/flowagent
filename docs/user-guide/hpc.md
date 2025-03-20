# HPC Configuration

FlowAgent supports High-Performance Computing (HPC) execution, with built-in support for SLURM, SGE, and TORQUE systems. The HPC settings can be configured through environment variables or in your `.env` file.

## LLM-Driven HPC Execution

FlowAgent uses its LLM capabilities to automatically generate and manage workflows on HPC systems. Instead of writing scripts, you can simply describe your analysis requirements in natural language, and the LLM will:

1. Design an appropriate workflow
2. Allocate suitable computational resources
3. Submit jobs to your HPC system
4. Monitor execution and handle errors
5. Collect and present results

For detailed information on the LLM-driven SLURM integration, see:
- [SLURM Configuration](../hpc/slurm_configuration.md)
- [LLM-Driven SLURM Integration](../hpc/llm_slurm_integration.md)

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

To use FlowAgent with your HPC system, simply describe your analysis requirements:

```
User: "I need to analyze RNA-seq data from 10 samples using the SLURM cluster."

FlowAgent: "I'll set up an RNA-seq analysis workflow on your SLURM cluster. 
Can you provide more details about your samples and analysis goals?"
```

The LLM will handle all the technical details of configuring and submitting HPC jobs.

## SLURM-Specific Features

For SLURM clusters, FlowAgent provides additional features:

- Automatic `.cgat.yml` configuration file generation
- Tool-specific resource allocation
- Detailed job monitoring and statistics
- Intelligent error handling and recovery
- Email notifications for job status

See the [SLURM Configuration](../hpc/slurm_configuration.md) guide for more details.

## Supported HPC Systems

FlowAgent currently supports these HPC systems:

1. **SLURM** (Recommended) - Full support with advanced features
2. **SGE** (Sun Grid Engine) - Basic support
3. **TORQUE/PBS** - Basic support

Additional HPC systems may be supported in future releases.
