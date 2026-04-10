# Configuration

## LLM Provider Configuration

FlowAgent supports multiple LLM providers. Set your provider and model in `.env`:

```bash
LLM_PROVIDER=openai           # openai | anthropic | google | ollama
LLM_MODEL=gpt-4.1             # Primary model
LLM_FALLBACK_MODEL=gpt-4.1-mini  # Fallback if primary unavailable
```

### Provider API keys

Set the key matching your chosen provider:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
GOOGLE_API_KEY=AIza...

# Ollama (local, no key needed)
LLM_BASE_URL=http://localhost:11434/v1
```

### Recommended models by provider

| Provider | Recommended | Notes |
|----------|------------|-------|
| OpenAI | `gpt-4.1`, `gpt-4.1-mini` | `gpt-4.1-nano` for lightweight tasks |
| Anthropic | `claude-sonnet-4-20250514` | `claude-opus-4-20250514` for complex analysis |
| Google | `gemini-2.5-flash` | `gemini-2.5-pro` for longer reasoning |
| Ollama | `llama4`, `qwen`, `deepseek` | Runs locally, no API key |

## Example configurations

1. Workflow execution (OpenAI):
   ```bash
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-4.1
   flowagent prompt "Analyze RNA-seq data in my fastq.gz files using Kallisto..."
   ```

2. Report generation (Anthropic):
   ```bash
   LLM_PROVIDER=anthropic
   LLM_MODEL=claude-sonnet-4-20250514
   flowagent prompt "analyze workflow results" --analysis-dir=results
   ```

3. Local model (Ollama):
   ```bash
   LLM_PROVIDER=ollama
   LLM_MODEL=llama4
   LLM_BASE_URL=http://localhost:11434/v1
   flowagent prompt "Analyze my ChIP-seq data..."
   ```

You can also override the model for a single run:
```bash
LLM_MODEL=gpt-4.1-mini flowagent prompt "quick QC check on my fastq files"
```
