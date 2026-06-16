# Intelligence Agent Evals

Assertion-based evals for `AgentInterface` (`IntelligenceModule`) — the anomaly-triggered monitoring agent. Each test case is a tuple of (anomaly_type × severity × event_context), run through the real LLM via `dispatch()`, and scored against binary pass/fail criteria.

## How tuples work

Each tuple in `tuples.jsonl` defines one test case as a combination of three dimensions:

- **anomaly_type** — which detector rule fired (`intensity_drop`, `intensity_drift`, `focus_decline`)
- **severity** — `warn` or `critical`
- **event_context** — how informative the recent event history is (`empty`, `scan_lifecycle_only`, `relevant`, `misleading`)

`build_inputs.py` converts each tuple into a concrete synthetic input: a realistic `anomaly` dict (with real float values derived from deployed thresholds) and a `recent_events` list. Those inputs are then run through the actual LLM via `run_eval.py`. Assertions are scored against the responses in `run_assertions.py`. See `tuples.jsonl` for the full set of test cases.

## Prerequisites

```bash
# Set the API key for whichever provider is configured in intelligence.agent.provider:
export ANTHROPIC_API_KEY=<your-key>   # provider: "anthropic" (or cborg with base_url)
export OPENAI_API_KEY=<your-key>      # provider: "openai" (or any compatible endpoint)

# intelligence.agent.enabled must be true in your config.
# The scripts resolve the config via sys.prefix — activating the right
# environment is enough. The config lives at:
#   .venv/pystxmcontrol_cfg/main.json          (venv)
#   ~/conda/envs/<your-env>/pystxmcontrol_cfg/main.json  (conda)
# Note: this is the installed config in your environment, not the
# main.json in the repo source tree.
```

## Running the pipeline

Run each step from the repo root. Activate your environment first, then use `python`:

```bash
source .venv/bin/activate        # venv
# or
conda activate <your-env>        # conda
```

### Step 1 — Build synthetic inputs

Converts tuples → concrete `(anomaly, recent_events)` pairs. Re-run if you change `tuples.jsonl` or the config thresholds.

```bash
.venv/bin/python evals/intelligence_agent/build_inputs.py
# → writes evals/intelligence_agent/inputs.jsonl
```

### Step 2 — Run the eval

Sends each input through the real LLM and records responses. Each run gets a unique timestamp-based `run_id`. Results are **appended** to `traces.jsonl` so you can accumulate runs across sessions.

Also writes a token summary and posts to `/cost/estimate` for a cost breakdown.

```bash
.venv/bin/python evals/intelligence_agent/run_eval.py
# → appends to evals/intelligence_agent/traces.jsonl
# → appends to evals/intelligence_agent/runs_meta.jsonl  (tokens + cost per run)
# → appends to <data_dir>/agent_traces_intelligence.jsonl  (production system log)
```

### Step 3 — Score assertions

Scores runs against 7 binary criteria and appends results to `results.jsonl`. By default scores only the latest run, but since `traces.jsonl` stores the full raw responses you can re-score historical runs after updating assertion definitions — no need to re-call the LLM.

```bash
# Score the latest run (default) → appends to results.jsonl
.venv/bin/python evals/intelligence_agent/run_assertions.py

# Score one specific run by ID → appends to results.jsonl
.venv/bin/python evals/intelligence_agent/run_assertions.py --run-id 20260610_143022

# Re-score every accumulated run with updated assertions → named output file required
.venv/bin/python evals/intelligence_agent/run_assertions.py --run-id all --output results_keyword_v2.jsonl
```

`--output` is accepted for any invocation and is required when `--run-id all` is used. The intent is that `results.jsonl` stays as the canonical scoreboard for your current assertion definitions, while named files (e.g. `results_keyword_v2.jsonl`, `results_with_judge.jsonl`) capture experimental snapshots for comparison. This makes it straightforward to compare the effect of different assertion implementations against the same set of historical traces without re-running the LLM.

### Step 4 — Generate the report

Builds `report.html` from all accumulated results. Open it in a browser.

```bash
.venv/bin/python evals/intelligence_agent/report.py
# → writes evals/intelligence_agent/report.html
open evals/intelligence_agent/report.html
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
