# Web Interface

`flowagent serve` launches a [Chainlit](https://chainlit.io/)-based chat
UI on top of the same `LLMInterface` and `WorkflowManager` used by the
CLI. It's the friendliest way for non-CLI users (wet-lab biologists,
collaborators) to drive the system.

## Launching

```bash
flowagent serve                       # binds 0.0.0.0:8000
flowagent serve --port 8001           # custom port
flowagent serve --host 127.0.0.1      # localhost only
```

Then open `http://localhost:8000` in a browser.

## What you get

- **Chat panel** — type a workflow prompt the same way you would on the
  CLI. Streamed token output from the LLM.
- **Plan visualisation** — once the LLM produces a workflow plan, the
  step DAG is rendered before execution begins.
- **Live step status** — each step shows a status badge (running,
  completed, failed, recovered).
- **Recovery transparency** — when the [error recovery loop](error-recovery.md)
  fixes a step, the original and patched commands are both displayed
  with the LLM's diagnosis.
- **File browser** — inspect input and output files in the working
  directory without leaving the UI.

## Authentication

The web interface uses JWT-signed sessions. Set a strong secret key:

```bash
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
```

Add it to your `.env` for persistence. The web UI will refuse to start
with the default placeholder secret in production-like environments.

## Configuration

The server reads the same env vars as the CLI:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1
OPENAI_API_KEY=sk-...

EXECUTOR_TYPE=local              # default executor for runs initiated in the UI
SECRET_KEY=...

# Web-specific
CHAINLIT_AUTH_SECRET=...         # optional
```

See [LLM Providers](llm-providers.md) for provider-switching details.

## Use cases

| Scenario | Why the UI |
|---|---|
| **Demo / teaching** | Visual DAG and step status are easier to follow than tail-ing logs |
| **Wet-lab user** | No CLI required; chat is more discoverable than `flowagent prompt --pipeline-format ...` |
| **Reviewing recovery** | The diagnosis + fixed-command panels make the recovery loop legible |
| **Collaborative debugging** | Share the screen / URL when a pipeline fails on a remote machine |

## Limitations

- **One concurrent run per session.** The UI doesn't multiplex multiple
  workflows. Use the CLI for parallel runs.
- **No HPC submission UX yet.** You can pick `--executor hpc` via the
  settings panel but there's no native job-monitor view; use the CLI for
  cluster work.
- **Production deployment is single-node.** Chainlit's session store is
  in-memory by default. For multi-user deployments, follow the Chainlit
  production-deployment guide.

## Alternative: the CLI is sufficient

If you're comfortable with the CLI, you don't need the web UI. The CLI
exposes everything the web UI does (and more — `--executor kubernetes`
isn't surfaced in the web settings yet).
