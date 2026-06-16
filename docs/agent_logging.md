# Agent Trace Logging

Both the intelligence agent and the task agent can write JSONL trace logs — one JSON object per line, appended on each call. Logging is off by default.

## Configuration

Each agent has its own `trace_log` key in `config/main.json`:

```json
{
    "server": {
        "data_dir": "/path/to/data"
    },
    "intelligence": {
        "agent": {
            "trace_log": null
        }
    },
    "task_agent": {
        "trace_log": null
    }
}
```

The value controls where the log is written:

| Value | Behavior |
|---|---|
| `null` | Logging disabled (default) |
| `"agent_traces_intelligence.jsonl"` | Written to `server.data_dir/agent_traces_intelligence.jsonl` |
| `"/absolute/path/to/file.jsonl"` | Written to that exact path |

A relative filename is always resolved against `server.data_dir`. An absolute path is used as-is.

## What is logged

### Intelligence agent (`intelligence.agent.trace_log`)

One entry is written per API call (both anomaly dispatches and direct queries):

- `call_type` — `"dispatch"` (anomaly-triggered) or `"query"` (user-triggered)
- `timestamp` — Unix timestamp of the call
- `model` — model name used
- `system_prompt`, `prompt`, `response` — full text of the system prompt, user prompt, and model response
- `anomaly_type`, `severity` — (dispatch only) the triggering anomaly details
- `query` — (query only) the original user query text
- `input_tokens`, `output_tokens` — token counts from the API response; `null` if the call failed
- `context_window` — model's max context size; only populated when using a CBORG/LiteLLM endpoint, `null` otherwise
- `context_fill_pct` — percentage of the context window used by the input; `null` if the call failed or `context_window` is unavailable
- `error` — exception message if the API call failed, otherwise `null`

### Task agent (`task_agent.trace_log`)

One entry is written per `run()` call (i.e. per user goal):

- `call_type` — always `"run"`
- `timestamp` — Unix timestamp when the run started
- `model` — model name used
- `goal` — the goal string passed to the agent
- `messages` — full conversation history (all iterations)
- `total_iterations` — number of tool-use steps taken
- `stop_reason` — why the run ended (`"done"`, `"max_iterations"`, `"cancelled"`, etc.)
- `response` — the agent's final text response
- `input_tokens`, `output_tokens` — token counts summed across all LLM calls in the run; sourced from the OpenAI-compatible API fields (`prompt_tokens`/`completion_tokens`) used by CBORG and other OpenAI-compatible endpoints
