# LLM Providers

FlowAgent works with four LLM providers behind a common interface
(`flowagent.core.providers`). Switch providers by setting two environment
variables — no code changes required.

## Supported providers

| Provider | Set `LLM_PROVIDER=` | Auth env var | Examples of `LLM_MODEL` |
|---|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4.1`, `gpt-4o`, `o3-mini` |
| Anthropic Claude | `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514`, `claude-opus-4` |
| Google Gemini | `google` | `GOOGLE_API_KEY` | `gemini-2.5-flash`, `gemini-2.5-pro` |
| Ollama (local) | `ollama` | none | `llama3.1:70b`, `qwen2.5:32b`, `mistral` |

The provider is auto-detected from `LLM_MODEL` if you omit
`LLM_PROVIDER` — `gpt-*` → openai, `claude-*` → anthropic, `gemini-*` →
google, anything else → ollama. Setting `LLM_PROVIDER` explicitly is
recommended.

## Quick configuration

### OpenAI

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4.1
export OPENAI_API_KEY=sk-...
# Optional:
export OPENAI_BASE_URL=https://api.openai.com/v1
```

### Anthropic

```bash
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-sonnet-4-20250514
export ANTHROPIC_API_KEY=sk-ant-...
```

### Google Gemini

```bash
export LLM_PROVIDER=google
export LLM_MODEL=gemini-2.5-flash
export GOOGLE_API_KEY=...
```

### Ollama (local, free, no API key)

```bash
# Start Ollama: https://ollama.com/download
ollama pull llama3.1:70b

export LLM_PROVIDER=ollama
export LLM_MODEL=llama3.1:70b
export LLM_BASE_URL=http://localhost:11434
```

## Persisting via `.env`

The repo root supports a `.env` file (auto-loaded). Copy
[`.env.example`](https://github.com/cribbslab/flowagent/blob/main/.env.example)
and edit:

```bash
cp .env.example .env
$EDITOR .env
```

## How FlowAgent uses LLM calls

A typical run makes 2–6 LLM calls:

| Call | Purpose | Tokens (typical) |
|---|---|---|
| `extract_input_files` | Identify FASTQ / BAM patterns from the prompt | small |
| `detect_workflow_type` | Classify as rna_seq_kallisto / chip_seq / etc. | small |
| `generate_workflow_plan` | The main planning call (returns full DAG) | medium |
| `suggest_resources` | Per-step CPU/memory recommendations (unknown tools only) | small |
| `_attempt_error_recovery` | Only on failure — diagnose + fix command | medium |
| `analyze_workflow` | Optional post-run analysis report | medium |

The planning call uses **structured output** (JSON-schema) where the
provider supports it (OpenAI, Anthropic, Google) for reliable parsing,
falling back to plain chat + repair logic for Ollama.

## Provider-specific notes

### OpenAI structured output

Uses `response_format={"type": "json_schema", "strict": true}` —
`WorkflowPlanSchema` is automatically converted by
`flowagent.core.schemas.to_json_schema()`. Failures fall back through
plain chat with prompt repair.

### Anthropic tool use

Uses Anthropic's tool-call API for both planning and the agent loop.
Retries on `rate_limit_error` with exponential back-off.

### Google Gemini

Uses Gemini's `responseSchema` for JSON output. Pricing is roughly an
order of magnitude cheaper than GPT-4.1, with measurable quality
trade-offs on adversarial prompts (see [Benchmarking](../benchmarking.md)).

### Ollama

Treats the local model as OpenAI-compatible. Works best with capable
instruction-tuned models (`llama3.1:70b`, `qwen2.5:32b`); smaller models
struggle with the structured output requirement and frequently need the
repair-fallback path. Set a longer timeout with
`LLM_TIMEOUT_SECONDS=600` for first-time loading.

## Fallback model

If the primary model fails (rate limit, model-not-found), FlowAgent
retries once with `LLM_FALLBACK_MODEL`:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1
LLM_FALLBACK_MODEL=gpt-4o-mini
```

Set the fallback to a cheaper / more available model in the same
provider family.

## Cost guidance

Approximate cost per FlowAgent run (one RNA-seq pipeline, no recovery):

| Model | Prompt tokens | Completion tokens | Estimated cost |
|---|---|---|---|
| `gpt-4.1` | ~5,000 | ~3,000 | ~$0.04 |
| `claude-sonnet-4` | ~5,000 | ~3,000 | ~$0.06 |
| `gemini-2.5-flash` | ~5,000 | ~3,000 | ~$0.003 |
| `ollama/llama3.1:70b` | n/a | n/a | $0 (local) |

Each error-recovery attempt adds ~$0.01–0.02 with GPT-4.1.

## Choosing a model for benchmarking

For paper-grade reproducibility we recommend pinning **exact model
strings with version dates** in `benchmarks/config/models.yaml`. Model
behaviour changes silently between major releases — always record the
date your benchmark ran. See [Benchmarking](../benchmarking.md).
