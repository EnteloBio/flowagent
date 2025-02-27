# Configuration

## OpenAI Model Configuration

FlowAgent uses OpenAI's language models for workflow generation and analysis. Different operations have different model requirements:

### Workflow Generation (gpt-3.5-turbo or better)
- Basic workflow creation and execution can use gpt-3.5-turbo
- Set in your `.env` file:
  ```bash
  OPENAI_MODEL=gpt-3.5-turbo
  ```

### Report Generation (gpt-4-turbo-preview recommended)
- For comprehensive analysis and insights, use gpt-4-turbo-preview
- This model provides better reasoning and analysis capabilities
- Set in your `.env` file:
  ```bash
  OPENAI_MODEL=gpt-4-turbo-preview
  ```

## Example configurations:

1. For workflow execution:
   ```bash
   # Set model in .env
   OPENAI_MODEL=gpt-3.5-turbo

   # Run workflow
   flowagent "Analyze RNA-seq data in my fastq.gz files using Kallisto..."
   ```

2. For report generation:
   ```bash
   # Set model in .env
   OPENAI_MODEL=gpt-4-turbo-preview

   # Generate comprehensive analysis
   flowagent "analyze workflow results" --analysis-dir=results
   ```

You can also set the model temporarily using environment variables:
```bash
# For one-time report generation with gpt-4-turbo-preview
OPENAI_MODEL=gpt-4-turbo-preview flowagent "analyze workflow results" --analysis-dir=results
```
