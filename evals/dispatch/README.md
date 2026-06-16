# Dispatch Evals

Assertion-based evals for `AgentInterface.dispatch()` — the anomaly-triggered intelligence agent. Each test case is a tuple of (anomaly_type × severity × event_context), run through the real LLM, and scored against binary pass/fail criteria.

## How tuples work

Each tuple in `tuples.jsonl` defines one test case as a combination of three dimensions:

- **anomaly_type** — which detector rule fired (`intensity_drop`, `intensity_drift`, `focus_decline`)
- **severity** — `warn` or `critical`
- **event_context** — how informative the recent event history is (`empty`, `scan_lifecycle_only`, `relevant`, `misleading`)

`build_inputs.py` converts each tuple into a concrete synthetic input: a realistic `anomaly` dict (with real float values derived from deployed thresholds) and a `recent_events` list. Those inputs are then run through the actual LLM via `run_eval.py`. Assertions are scored against the responses in `run_assertions.py`. See `tuples.jsonl` for the full set of test cases.

## Prerequisites

```bash
# ANTHROPIC_API_KEY must be set (same key works for cborg)
export ANTHROPIC_API_KEY=<your-key>

# intelligence.agent.enabled must be true in your config
# (.venv/pystxmcontrol_cfg/main.json)
```

## Running the pipeline

Run each step from the repo root. All scripts use `.venv/bin/python`.

### Step 1 — Build synthetic inputs

Converts tuples → concrete `(anomaly, recent_events)` pairs. Re-run if you change `tuples.jsonl` or the config thresholds.

```bash
.venv/bin/python evals/dispatch/build_inputs.py
# → writes evals/dispatch/inputs.jsonl
```

### Step 2 — Run the eval

Sends each input through the real LLM and records responses. Each run gets a unique timestamp-based `run_id`. Results are **appended** to `traces.jsonl` so you can accumulate runs across sessions.

Also writes a token summary and posts to `/cost/estimate` for a cost breakdown.

```bash
.venv/bin/python evals/dispatch/run_eval.py
# → appends to evals/dispatch/traces.jsonl
# → appends to evals/dispatch/runs_meta.jsonl  (tokens + cost per run)
# → appends to <data_dir>/agent_traces_intelligence.jsonl  (production system log)
```

### Step 3 — Score assertions

Scores the latest run against 7 binary criteria. Results are appended to `results.jsonl`. Prints a pass/fail table to the console.

```bash
.venv/bin/python evals/dispatch/run_assertions.py
# → appends to evals/dispatch/results.jsonl
```

### Step 4 — Generate the report

Builds `report.html` from all accumulated results. Open it in a browser.

```bash
.venv/bin/python evals/dispatch/report.py
# → writes evals/dispatch/report.html
open evals/dispatch/report.html
```

## Output files

| File | Contents |
|------|----------|
| `tuples.jsonl` | Test case definitions (needs scientist review) |
| `inputs.jsonl` | Synthetic inputs built from tuples |
| `traces.jsonl` | LLM responses — all runs accumulated |
| `runs_meta.jsonl` | Token counts + cost estimate per run |
| `results.jsonl` | Assertion pass/fail per trace — all runs accumulated |
| `report.html` | Visual report — heatmap + pass rates + token/cost table |

## Assertions

| Assertion | What it checks |
|-----------|---------------|
| `no_error` | Response is not an API error |
| `on_topic` | Addresses the correct anomaly domain |
| `has_action` | Includes at least one suggested action |
| `critical_urgent` | Critical anomalies use urgency language |
| `uses_shutter_context` | Mentions shutter when a shutter event is in context |
| `uses_zone_plate_context` | Mentions zone plate when ZonePlateZ moved |
| `uses_energy_context` | Mentions energy/focal length when Energy moved during focus decline |
