# Task Agent Evals

Assertion-based evals for `TaskAgent` — the multi-turn autonomous agent that controls the STXM to execute scientist-specified goals. Each test case is a tuple of (task_type × complication × element), run through the real agent against a scripted mock server, and scored against binary pass/fail criteria on the tool calls the agent produces.

Unlike the intelligence agent evals (one-shot LLM call → text response), the task agent is multi-turn: each task runs through one or more conversation stages on the same agent instance, and assertions check **which tools were called and with what parameters**, not text content.

## How tuples work

Each tuple in `tuples.jsonl` defines one test scenario along four dimensions:

- **task_type** — what the scientist is asking for (`elemental_map`, `image_scan`, `line_spectrum`)
- **complication** — what makes the scenario hard or failure-prone (`happy_path`, `no_recs`, `scan_error`, `stale_params`, `given_particle`)
- **element** — the element being studied (`Fe`, or `null` for non-elemental scans)
- **plan_detail** — how much physics context is in the *initial intent itself* (`specified` vs `underspecified`). Every stage's approval message after that is a plain "Yes, please proceed." regardless of `plan_detail` — the detail differentiation lives entirely in the intent, not in what the user says afterward.
  - `specified` — the intent already states the exact scan type, edge, energies, and point count a scientist who knows what they want would give (e.g. "...use Fe L-edge energies (pre-edge ~695 eV, on-edge ~709 eV), not Fe K-edge at ~7 keV"). Even a model that would otherwise ask for clarification has everything it needs to call tools.
  - `underspecified` — the intent is a bare ask ("find the Fe particles in a 10 um x 10 um area") with no scan/energy detail. The agent must supply the correct approach itself. Both of the following count as meeting expectations: proposing the correct two-energy Fe L-edge setup directly, or asking the user a clarifying question first — `run_assertions.py`'s `energy_in_range` check on the plan stage is N/A (not a fail) when no energies are stated yet, and only evaluated once concrete numbers appear.

`build_inputs.py` expands each tuple into a concrete task definition: an intent string, mock server overrides, particle coordinates, and per-stage expected assertions. Those tasks are run through the real `TaskAgent` (with a scripted mock replacing the hardware server) in `run_eval.py`. Tool calls are scored deterministically in `run_assertions.py`.

## Prerequisites

```bash
# Set the API key for the provider configured in task_agent.provider in your config:
export ANTHROPIC_API_KEY=<your-key>   # provider: "anthropic" (or cborg with base_url)
export OPENAI_API_KEY=<your-key>      # provider: "openai" (or any compatible endpoint)

# The scripts resolve the config via sys.prefix — activating your environment is enough.
# The config lives at:
#   .venv/pystxmcontrol_cfg/main.json                       (venv)
#   ~/conda/envs/<your-env>/pystxmcontrol_cfg/main.json     (conda)
# This is the installed config in your environment, not the main.json in the repo root.
```

## Running the pipeline

Run each step from the repo root with your environment activated:

```bash
source .venv/bin/activate        # venv
# or
conda activate <your-env>        # conda
```

### Step 1 — Build task inputs

Expands `tuples.jsonl` → concrete task definitions in `tasks.jsonl`. Re-run whenever you add or change tuples.

```bash
python evals/task_agent/build_inputs.py
# → writes evals/task_agent/tasks.jsonl
```

### Step 2 — Run the eval

Runs the real `TaskAgent` through every task. The hardware server is replaced with a scripted in-memory mock — no stxmserver or physical instrument needed. Each run gets a unique timestamp-based `run_id`. Results are **appended** to `traces.jsonl` so you can accumulate runs across sessions.

The mock also tracks which LLM iteration each tool call came from (`llm_iter`), enabling detection of parallel-call bugs (e.g. `update_scan` and `start_multiregion_scan` fired in the same LLM completion before the agent sees the update result).

```bash
python evals/task_agent/run_eval.py
# → appends to evals/task_agent/traces.jsonl
# → appends to evals/task_agent/runs_meta.jsonl  (tokens + cost per run)
```

### Step 3 — Score assertions

Scores each stage's tool calls against the expected assertions. All checks are deterministic — no LLM judge. Results are appended to `results.jsonl`. By default scores only the latest run; since `traces.jsonl` stores the full raw tool calls, you can re-score historical runs after updating assertion logic without re-running the agent.

```bash
# Score the latest run (default) → appends to results.jsonl
python evals/task_agent/run_assertions.py

# Score one specific run by ID
python evals/task_agent/run_assertions.py --run-id 20260629_144547

# Re-score all accumulated runs with updated assertions → named output required
python evals/task_agent/run_assertions.py --run-id all --output results_v2.jsonl
```

`--output` is accepted for any invocation and is required with `--run-id all`. Use named output files to capture experimental snapshots for comparison without overwriting the canonical `results.jsonl`.

### Step 4 — Generate the report

Builds `report.html` from all accumulated results. Open it in a browser.

```bash
python evals/task_agent/report.py
# → writes evals/task_agent/report.html
open evals/task_agent/report.html
```

## Output files

| File | Contents | Committed to git? |
|------|----------|--------------------|
| `tuples.jsonl` | Test case definitions (human-authored) | **Yes** |
| `tasks.jsonl` | Full task definitions built from tuples | No — regenerate with `build_inputs.py` |
| `traces.jsonl` | Agent tool calls — all runs accumulated | No |
| `runs_meta.jsonl` | Token counts + cost estimate per run | No |
| `results.jsonl` | Assertion pass/fail per stage — all runs | No |
| `report.html` | Visual report — heatmap + pass rates | No |

## Stages: plan vs. action

Every task now runs through a `plan_proposal` stage (`kind: "plan"`) before its `tasks.jsonl`-defined
stages (`kind: "action"`):

- **`plan_proposal`** — turn 0: the agent's response to the raw intent alone, before any tool call
  or human confirmation. Scored against `task["plan_expected"]` (built by `build_inputs.py`'s
  `_plan_expected()`). This is normally free text, not a tool call — so `run_eval.py` injects an
  extra eval-only tool, `propose_plan` (see `plan_probe.py`), the agent can call to declare
  concrete parameters without executing anything. That tool call, not prose, is what gets scored
  (see "Plan-stage scoring" below). Nothing in `pystxmcontrol/controller/task_agent/` is modified
  to make this work — it's purely an eval-harness technique.
- **action stages** — the existing `tasks.jsonl` stages: the agent's response to a human
  confirmation message, scored directly against its tool calls.
- **`tool_unit`** — a fresh, single-turn, isolated tool-call test (`build_inputs.py`'s
  `_tool_unit_cases()`): "call `update_scan` with reasonable parameters for X," no plan/confirm
  dance, no prior conversation history. Isolates "does the model know how to call this tool
  correctly" from "does it choose the right moment to, inside a multi-turn plan." One case per
  `task_type`, not per tuple, since the physics don't vary by complication.

## Assertions

Each stage is scored on five binary assertions:

| Assertion | Abbrev | What it checks |
|-----------|--------|----------------|
| `well_formed` | WF | No API error; all tool call arguments parse correctly |
| `plan_formatted` | PF | `plan_proposal` only: the agent used `propose_plan` (or a real `update_scan` call) when it stated concrete parameters, instead of leaving them only in prose. N/A if a real tool call already gives structured data, or the agent asked a clarifying question with no concrete numbers yet |
| `right_approach` | RA | At least one tool in `must_call` was called |
| `params_match` | PMS | `update_scan` parameters (or, for `plan_proposal`, the agent's `propose_plan` call) match the expected strategy (see sub-checks below) |
| `dispatchable` | DISP | Every tool call is executable: known name, valid argument schema, and `update_scan` accepted by a real offline `ToolSet` |

`params_match` is broken into sub-checks defined per stage in `tasks.jsonl`:

| Sub-check | What it checks |
|-----------|----------------|
| `energy_list_len: N` | `update_scan` uses an `energy_list` with exactly N entries |
| `energy_in_range: [lo, hi]` | All implied energies fall within `[lo, hi]` eV. N/A if none are found — never a false fail on a vague-but-not-wrong plan |
| `if_update_scan_center_near` | `update_scan` x/y_center is within `tol` µm of the target particle (N/A if agent used `load_intelligence_particles` for positioning, or stated no coordinate) |
| `max_calls: {tool: N}` | Tool called at most N times — catches blind-retry loops on errors |
| `scan_type_param: str` | Any `update_scan` call sets `scan_type` to the expected value |
| `no_parallel_start` | `update_scan` and `start_multiregion_scan` are in different LLM iterations — agent must see the update result before firing the scan |
| `energy_spectrum_min_points: N` | `energy_points` or `len(energy_list)` is at least N — distinguishes a spectrum from a two-point elemental map |
| `update_scan_nonempty` | `update_scan` is called with at least one non-empty argument — catches `update_scan({})` confusion from stale server state |

Only `energy_in_range` applies to `plan_proposal` stages — the rest require a real `update_scan`
call sequence that a plan stage doesn't have.

## Plan-stage scoring

`plan_proposal` is prose, not a tool call, so it can't be scored against `param_checks` the way a
real `update_scan()` call is — unless the agent gives us something structured to parse.

Two approaches were tried:

1. **A system-prompt instruction** asking the agent to end any plan that states concrete
   parameters with a fenced ` ```json ` block. Measured compliance: **0/7 (0%)** — the model
   (`gemini-pro`) never produced the block, even while stating exact numbers in prose. Verified
   this was a real compliance gap, not a parsing bug, by inspecting raw `final_text`.
2. **`propose_plan`, an eval-injected tool** (`plan_probe.py`): `run_eval.py` monkeypatches
   `agent._llm.chat.completions.create` to add one extra tool — mirroring `update_scan`'s exact
   field names — the agent can call to declare parameters without executing anything. Measured
   compliance: **10/10 (100%)**. Models are far more reliable at native tool-calling than at
   following a "please format your text like this" instruction, and this eval already had strong
   evidence for that (`dispatchable` was ~100% throughout). Nothing in
   `pystxmcontrol/controller/task_agent/{agent,tools}.py` changes to make this work — it's purely
   an eval-harness technique, reusable against any other agent under eval the same way.

`run_assertions.py`'s `_plan_signal()` resolves both the synthetic `update_scan`-shaped dict AND
the format-compliance verdict together, in one pass (previously two separately-maintained
functions with parallel branching — easy to update one and forget the other). Priority order for
the dict:

1. a real speculative `update_scan()` call the model made this turn, if any
2. a real `propose_plan()` call, merged onto any partial real `update_scan()` fields (the real
   call wins on conflicts, since it reflects what's actually staged)
3. a stray fenced `` ```json `` block in free text (`_plan_block()`) — kept as a defensive
   fallback in case a model emits this format unprompted
4. legacy regex energy-scraping (`_energies_from_text()`) — only as a last resort when none of
   the above exists, e.g. a model/run predating `propose_plan`

Format *compliance itself* (`plan_formatted`) is resolved alongside it: if the agent states
concrete numbers in prose but used neither `propose_plan` nor a `` ```json `` block, that's a
`False` — an agent/LLM instruction-following failure, not an eval-heuristic gap. It's `None`
(N/A) whenever there's nothing to require structured output for: a real tool call already
supplied structured data, or the agent legitimately asked a clarifying question with no concrete
numbers yet.

The regex fallback still exists for backward compatibility but is intentionally coarse — it can't
tell a *proposed* scan energy from an incidentally-mentioned current reading, safety threshold, or
step size ("currently at 600 eV", "moving Energy by >100 eV", "a 1 eV resolution"). A short list of
local-context exclusion words filters the common cases on both sides of the match, but this is a
best-effort heuristic, not full comprehension, which is exactly why it's no longer the primary
source of truth. `energy_in_range` is N/A (not a fail) whenever no energies are found at all, so a
vague-but-not-wrong plan is never penalized.

## Adding new test cases

1. Add a line to `tuples.jsonl` with the new `task`, `task_type`, `complication`, `element`, and `notes`.
2. Add the corresponding stage logic to `build_inputs.py` (`_stages()`, `_intent()`,
   `_mock_overrides()`, `_plan_expected()` as needed).
3. Run `build_inputs.py` to regenerate `tasks.jsonl`.
4. Run the full pipeline (steps 2–4 above) to see results.
