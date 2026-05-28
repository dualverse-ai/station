# Provider API Backup Fallback

Station supports a simple provider-level API fallback system for Gemini, Claude,
OpenAI, and Grok. The fallback state is shared in memory across all agents that
use the same provider. Agents do not own provider endpoints, and no per-agent
routing state files are written.

## Configuration

The base endpoint is still the normal provider environment:

```bash
OPENAI_API_KEY=base-key
OPENAI_BASE_URL=https://cost-low.example.test/v1
OPENAI_HTTP_PROXY=http://proxy.example.test:8080
OPENAI_HTTPS_PROXY=http://proxy.example.test:8080
```

Backups use matching `BACKUP_` environment variables with semicolon-delimited
entries:

```bash
BACKUP_OPENAI_API_KEY=backup-key;official-key
BACKUP_OPENAI_BASE_URL=https://backup.example.test/v1;
BACKUP_OPENAI_HTTP_PROXY=;
BACKUP_OPENAI_HTTPS_PROXY=;
```

Rules:

- `BACKUP_*_API_KEY` defines the number of backup endpoints and cannot contain
  blank entries.
- Backup Base URL and proxy lists must be empty or have the same number of
  semicolon-delimited entries as the backup API key list.
- A blank backup Base URL means use the provider SDK default, which is normally
  the official endpoint.
- A blank backup proxy uses the Station proxy, matching the base endpoint
  behavior. If the Station proxy is also blank, no proxy is used.
- Invalid backup configuration raises `ValueError` at startup. Station should
  not silently ignore malformed backup settings.

The dashboard API settings modal exposes backups as rows, so users do not need
to type semicolons manually. Leaving an existing backup key blank keeps the
current key; new backup rows require a key. The backend rejects semicolons in
row fields before converting the rows to `BACKUP_*` environment values.

## Provider Env Names

OpenAI:

- `BACKUP_OPENAI_API_KEY`
- `BACKUP_OPENAI_BASE_URL`
- `BACKUP_OPENAI_HTTP_PROXY`
- `BACKUP_OPENAI_HTTPS_PROXY`

Claude:

- `BACKUP_ANTHROPIC_API_KEY`
- `BACKUP_ANTHROPIC_BASE_URL`
- `BACKUP_ANTHROPIC_HTTP_PROXY`
- `BACKUP_ANTHROPIC_HTTPS_PROXY`

Gemini:

- `BACKUP_GOOGLE_API_KEY`
- `BACKUP_GOOGLE_GEMINI_BASE_URL`
- `BACKUP_GOOGLE_GEMINI_HTTP_PROXY`
- `BACKUP_GOOGLE_GEMINI_HTTPS_PROXY`

Grok:

- `BACKUP_XAI_API_KEY`
- `BACKUP_XAI_BASE_URL`
- `BACKUP_XAI_HTTP_PROXY`
- `BACKUP_XAI_HTTPS_PROXY`

## Runtime Behavior

Each provider has one shared default endpoint. The default starts at the base
endpoint from the normal provider env vars.

When a provider call fails for any reason, including context overflow, the
connector retries the same endpoint once before rotating to the next configured
endpoint. The endpoint list loops: base, backup 1, backup 2, then base again.
There is no retry sleep while moving inside one endpoint loop. When the loop
wraps back to the first endpoint, Station prints that the provider fallback loop
finished and waits using the normal LLM retry delay before starting the next
loop.

Station also records the recent result history for the current provider default.
If the trailing 10 non-expired calls for that default have a failure rate above
70 percent, Station promotes the provider default to the next endpoint. Samples
older than one hour are dropped before the rate is evaluated.

When the default is not the base endpoint, connectors try a base recovery probe
at most once every 30 minutes. The probe uses the current connector's configured
model and asks for a minimal response. If the probe succeeds, the shared default
returns to base.

Endpoint switches are printed to stdout with provider, endpoint index/name, and
Base URL. API keys are not printed.

## Parallel Mode

Parallel sync mode creates separate connector instances for multiple agents, but
they all use the same process-level provider fallback state in
`station/runtime_api_config.py`. A failure from one agent can therefore move the
provider default for all agents. This is intentional: a single provider timeout
is treated as provider health information, not as an agent-specific setting.

No `llm_routing_state.yaml` files are used, and there is no
`custom_api_for_routing.yaml` endpoint router.
