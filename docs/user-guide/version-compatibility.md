# Version Compatibility

FlowAgent automatically handles version compatibility for Kallisto indices:

## Version Checking

- Checks Kallisto version before index creation
- Validates index compatibility using `kallisto inspect`
- Stores version information in workflow metadata

## Error Prevention

- Detects version mismatches before execution
- Provides detailed error messages for incompatible indices
- Suggests resolution steps for version conflicts

## Metadata Management

- Tracks index versions across workflows
- Maintains compatibility information
- Enables reproducible analyses

## Updating the Environment

To update your conda environment with new dependencies:

```bash
conda env update -f conda/environment/environment.yml
```

## Managing Multiple Environments

For development or testing, you can create a separate environment:

```bash
conda env create -f conda/environment/environment.yml -n flowagent-dev
```
