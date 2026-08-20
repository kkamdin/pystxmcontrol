# Task Agent Evals

Assertion-based evals for `TaskAgent` — the multi-turn autonomous agent that controls the STXM to execute scientist-specified goals. Each test case is a tuple of (task_type × complication × element), run through the real agent against a scripted mock server, and scored against binary pass/fail criteria on the tool calls the agent produces.

Unlike the intelligence agent evals (one-shot LLM call → text response), the task agent is multi-turn: each task runs through one or more conversation stages on the same agent instance, and assertions check **which tools were called and with what parameters**, not text content.

## How tuples work

Each tuple in `tuples.jsonl` defines one test scenario along four dimensions:

- **task_type** — what the scientist is asking for (`elemental_map`, `image_scan`, `line_spectrum` — the last of these is intentionally not being expanded further right now; see "Scope" below)
- **complication** — what makes the scenario hard or failure-prone:
  - `happy_path`, `no_recs`, `stale_params`, `given_particle` (line_spectrum only) — original set
  - `scan_error` / **`tool_error`** — a software-side tool call fails cleanly (server error, busy controller). `scan_error` is the original image_scan case; `tool_error` is the same idea extended to elemental_map failure points (survey scan fails, or the high-res follow-up fails after particles are found)
  - **`hardware_error`** — a hardware-side fault (motor occupied, shutter interlock, DAQ fault, stage limit switch). Same expected response as tool_error: stop, report the specific error, ask — never retry blindly or fabricate a result
  - **`out_of_range_element`** — the requested element's accessible edge (or an explicitly-stated energy) falls outside this beamline's 250–2000 eV soft X-ray range. Agent must recognize the physical limitation, not attempt an impossible acquisition
  - **`invalid_request`** — the request itself doesn't map to a real capability (colloquial/invalid `scan_type`, a motion axis that doesn't exist on this instrument, or something entirely outside the toolset). Agent must recognize this and ask, not hallucinate a matching tool call
  - **`mid_task_change`** — the user redirects partway through, after an earlier stage already established real context (a completed survey, a found particle). Tests whether the agent actually adapts to the new instruction instead of persisting with the abandoned plan — every other complication tests a single, unchanging goal from start to finish; this is the only one where the goal itself changes mid-conversation
- **element** — `Fe`, `C`, `O`, `Ni`, `Cu`, `Ce` (all with edges inside the beamline's 250–2000 eV range — see `ELEMENT_EDGES` in `build_inputs.py`), `S` / `P` (edges genuinely outside that range, used only for `out_of_range_element` cases), or `null` for non-elemental scans. The element-coverage group (`specified` and `underspecified`, 5 elements each) deliberately spans all three edge types this beamline can reach: K-edge (`C`, `O` — light/organic), L-edge (`Fe`, `Ni`, `Cu` — 3d transition metals), M-edge (`Ce` — lanthanide, e.g. CeO2 redox chemistry). Earlier revisions covered 9 same-edge-type-heavy elements (`N`, `Ca`, `Mn`, `Co`, `Si`); trimmed down since a 6th/7th/8th/9th element of an edge type already covered adds little signal beyond what one clean representative per edge type already tests — does the agent know the right edge *type* and energy for an unfamiliar element
- **plan_detail** — how much physics/task context is in the *initial intent itself*. Every stage's approval message after that is a plain "Yes, please proceed." (or a close variant) regardless of `plan_detail` — the detail differentiation lives entirely in the intent, not in what the user says afterward.
  - `specified` — the intent already states the exact scan type, edge, energies, and point count a scientist who knows what they want would give (e.g. "...use Fe L-edge energies (pre-edge ~695 eV, on-edge ~709 eV), not Fe K-edge at ~7 keV"). Even a model that would otherwise ask for clarification has everything it needs to call tools.
  - `underspecified` — the intent is a bare ask ("find the Fe particles in a 10 um x 10 um area") with no scan/energy detail. The agent must supply the correct approach itself. Both of the following count as meeting expectations: proposing the correct two-energy Fe L-edge setup directly, or asking the user a clarifying question first — `run_assertions.py`'s `energy_in_range` check on the plan stage is N/A (not a fail) when no energies are stated yet, and only evaluated once concrete numbers appear.
  - `minimal` — even less detail than underspecified: no area, no scan type, sometimes not even a fully-formed task ("show me my sample"). Stage expectations are deliberately loose here (any real engagement counts) since there's no way to derive one "correct" scan from zero detail.

`build_inputs.py` expands each tuple into a concrete task definition via `SCENARIOS`, a registry keyed by task name — each entry is one self-contained dict (intent, particle, mock server overrides, plan-stage expectations, per-stage tool-call expectations). Those tasks are run through the real `TaskAgent` (with a scripted mock replacing the hardware server) in `run_eval.py`. Tool calls are scored deterministically in `run_assertions.py`.

### Scope: why line_spectrum coverage isn't being expanded right now

The new element/complication coverage added in this round focuses on `elemental_map` (two-energy scans to find element-sensitive particles / sample inhomogeneity) rather than `line_spectrum` (oxidation-state-ratio spectroscopy on a found particle). `line_spectrum` keeps its original 5 tuples unchanged; broader `line_spectrum` coverage is deferred to a later round.

### Scenario registry, not five parallel dispatch functions

Earlier, `build_inputs.py` had five separate functions (`_intent`, `_particle`, `_mock_overrides`, `_plan_expected`, `_stages`), each its own `if/elif` chain keyed by `(task_type, complication)`. At 10 tuples that was already fragile — nothing enforced the five functions agreed on which combinations existed, or that a change to one wasn't forgotten in another. At 40+ tuples it would have meant touching five places per new test case with no cross-check. `SCENARIOS` replaces that: each test case is one self-contained registry entry (a few use shared constructors like `_find_element_scenario()` / `_out_of_range_scenario()` to avoid retyping the same shape by hand, but each still owns its full definition). Adding a test case means adding one new entry; nothing else.

## Prerequisites

Verified from a genuinely fresh `conda create -n <env> python=3.11` + install — two things below
are easy to miss and both fail in confusing ways if skipped, so do them in order.

### 1. Install non-editable, or copy the config by hand

```bash
pip install .          # NOT -e . -- see why below
```

`pip install -e .` (an editable install, the natural first instinct for a repo you're about to
work in) silently skips the `[tool.setuptools.data-files]` step that copies `config/*.json` to
`<env-prefix>/pystxmcontrol_cfg/` — this is a known setuptools/pip limitation with editable
installs, not specific to this package. Every eval script resolves its config via `sys.prefix`
(see below), so on an editable install they all fail with a plain `FileNotFoundError` pointing at
a `pystxmcontrol_cfg/main.json` that was simply never created — nothing about that error says
"you used the wrong install flag."

If you need an editable install anyway (active development on `pystxmcontrol` itself), copy the
config over by hand once after installing:

```bash
pip install -e .
mkdir -p "$(python -c 'import sys; print(sys.prefix)')/pystxmcontrol_cfg"
cp config/{daq,main,motor,scan}.json config/log.txt config/xeryon_default.txt \
   "$(python -c 'import sys; print(sys.prefix)')/pystxmcontrol_cfg/"
```

### 2. Point the installed config at your LLM gateway, then set the key

The scripts resolve the config via `sys.prefix` — the **installed** config in your active
environment, not `config/main.json` in the repo source tree:

```
.venv/pystxmcontrol_cfg/main.json                     (venv)
<conda-root>/envs/<your-env>/pystxmcontrol_cfg/main.json   (conda)
```

The repo's tracked `config/main.json` ships with `task_agent.provider.base_url: null` for every
provider — i.e. calls go straight to the real `api.openai.com`/`api.anthropic.com`. If your team
routes through a gateway (e.g. LBL's CBORG, an OpenAI-compatible proxy), edit **your installed**
copy of `main.json` and set that provider's `base_url` accordingly before running anything —
otherwise you'll get a same-shaped-but-wrong failure: a `401 Incorrect API key` error that quotes
`platform.openai.com`, which reads exactly like "your key is wrong" when the actual problem is
"this request never reached the gateway you meant it to." Through a gateway that fronts multiple
model families behind one key (CBORG does this), every provider entry's `api_key_env` typically
points at the *same* env var regardless of the underlying model family — e.g. both the
`anthropic` and `openai` provider blocks set `"api_key_env": "OPENAI_API_KEY"`, because that's the
one gateway key being used, not a real OpenAI key.

```bash
# Set whichever env var(s) your installed main.json's provider.api_key_env entries actually
# name -- through a gateway this is often just one key covering every provider block:
export OPENAI_API_KEY=<your-gateway-or-provider-key>
export ANTHROPIC_API_KEY=<your-key>
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

`run_eval.py` uses whichever single model is configured at `task_agent.model` in your installed
`main.json`. To compare several models without hand-editing that config between runs, use
`run_eval_multi.py` instead — see the next section.

#### Running against multiple models

`run_eval_multi.py` drives the same machinery as `run_eval.py` across a list of models in one
pass, without ever touching `main.json` on disk — each model gets its own **in-memory** copy of
the config with `task_agent.model` overridden, so nothing about your local environment changes
and nothing needs to be reverted afterward.

```bash
# All models in the MODELS dict at the top of the file
python evals/task_agent/run_eval_multi.py

# Just a few, comma-separated (CBORG-style provider/model or a bare model id, matching
# whatever task_agent.provider.base_url in your config expects)
python evals/task_agent/run_eval_multi.py --models gemini-pro,anthropic/claude-sonnet

# Skip models that already have traces recorded (useful for resuming an interrupted batch)
python evals/task_agent/run_eval_multi.py --skip-tested

# Targeted retry: re-run only specific tasks for specific models, e.g. after purging rows
# contaminated by a rate limit or network blip on just those tasks. The run_id gets a "_retry"
# suffix so it's identifiable as a partial run rather than a fresh full one.
python evals/task_agent/run_eval_multi.py --models google/grok-4.3 --tasks hw_daq_fault,fe_spectrum
```

One broken or rate-limited model never aborts the batch: failures are collected per-model and
reported at the end, and `traces.jsonl` is scanned afterward for two distinct failure signatures
so a model that fails silently doesn't look like a clean run before you've even scored it —
`agent.run()` swallows its own LLM-call exceptions internally and returns them as ordinary
`final_text` (`"LLM call failed: ..."`) rather than raising, and separately, a real bug triggered
by an unusual response shape (e.g. inline `<think>` blocks) can escape as `trace["error"]` instead.
Status per model (`OK` / `PARTIAL` / `ALL_FAILED`), token counts, and an estimated cost (fetched
from your provider's `/model_group/info` pricing endpoint, when available) are written to
`batch_run_summary.json` (gitignored — a per-batch scratch summary, not accumulated history).

**Repeat runs.** Because every run gets a fresh timestamp-based `run_id` and `traces.jsonl` is
append-only, running the same batch again doesn't overwrite anything — it accumulates a second
data point per model. Running it 3–4 times and comparing per-run pass rates is a cheap way to see
how repeatable a model's behavior actually is before trusting a single run's numbers, especially
for models with any nondeterminism in tool-calling.

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

> **`--output` is append-mode, always — including for `results.jsonl` itself.** After changing
> assertion logic in `run_assertions.py`, re-scoring everything into the canonical file means
> truncating it first, or you'll end up with duplicate rows for every run:
> ```bash
> > evals/task_agent/results.jsonl   # truncate -- run_assertions.py only ever appends
> python evals/task_agent/run_assertions.py --run-id all --output results.jsonl
> ```
> This is safe precisely because nothing here re-calls the LLM: `traces.jsonl` already has every
> raw tool call from every run, so rescoring is pure local recomputation.

### Step 4 — Generate the report

Builds `report.html` from all accumulated results — a heatmap-style view, one row per stage x
model. Open it in a browser.

```bash
python evals/task_agent/report.py
# → writes evals/task_agent/report.html
open evals/task_agent/report.html
```

### Step 5 — Generate the interactive assertion breakdown (optional)

A second, complementary report: one row per **case** (all 44, not per-stage), one column per
**assertion**, and a model picker that swaps every cell client-side — answers "for model X, which
assertion fails on which case?" more directly than `report.html`'s per-stage heatmap does. Self-
contained, single HTML file, no server needed; works offline once generated.

```bash
python evals/task_agent/generate_assertion_report.py
# → writes evals/task_agent/assertion_report.html
open evals/task_agent/assertion_report.html

# Compare a named experimental results file instead of the canonical one:
python evals/task_agent/generate_assertion_report.py --results results_v2.jsonl --output report_v2.html
```

Cell values are **passes/applicable**, pooled across every stage of that case and every
accumulated run for the selected model — not just the latest run, so re-run Step 2 a few times
(see "Repeat runs" above) before trusting a case's numbers if a model's tool-calling has any
nondeterminism. Group and overall percentages are an **unweighted mean across cases**: each
case's own passes/applicable are collapsed into one rate first, then every case counts equally
regardless of how many stages/runs fed it, so a 12-check case can't outvote a 2-check one.

## Output files

| File | Contents | Committed to git? |
|------|----------|--------------------|
| `tuples.jsonl` | Test case definitions (human-authored) | **Yes** |
| `tasks.jsonl` | Full task definitions built from tuples | No — regenerate with `build_inputs.py` |
| `traces.jsonl` | Agent tool calls — all runs accumulated | No |
| `runs_meta.jsonl` | Token counts + cost estimate per run | No |
| `batch_run_summary.json` | Per-batch status/cost summary from `run_eval_multi.py` | No |
| `results.jsonl` | Assertion pass/fail per stage — all runs | No |
| `report.html` | Visual report — per-stage heatmap + pass rates | No |
| `assertion_report.html` | Interactive per-case × per-assertion report, model picker | No |

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

Each stage is scored on six binary assertions:

| Assertion | Abbrev | What it checks |
|-----------|--------|----------------|
| `well_formed` | WF | No API error; all tool call arguments parse correctly |
| `plan_formatted` | PF | `plan_proposal` only: `True` when the agent declared parameters via `propose_plan`, a real `update_scan` call, or a structured JSON block. N/A (never a fail) otherwise — not calling one of those is equally consistent with a legitimate clarifying question, or a refusal that merely echoes the instrument's current/general state rather than proposing anything. See "Plan-stage scoring" below |
| `right_approach` | RA | At least one tool in `must_call` was called |
| `avoids_wrong` | AW | No tool call is in the stage's `forbidden_tools` list — e.g. must NOT call `start_scan` when the request is physically impossible or outside the toolset's capabilities |
| `params_match` | PMS | `update_scan` parameters (or, for `plan_proposal`, the agent's `propose_plan` call) match the expected strategy (see sub-checks below) |
| `dispatchable` | DISP | Every tool call is executable: known name, valid argument schema, and `update_scan` accepted by a real offline `ToolSet` |

`params_match` is broken into sub-checks defined per stage in `tasks.jsonl`:

| Sub-check | What it checks |
|-----------|----------------|
| `energy_list_len: N` | `update_scan` uses an `energy_list` with exactly N entries |
| `energy_in_range: [lo, hi]` | All implied energies fall within `[lo, hi]` eV. N/A if none are found — never a false fail on a vague-but-not-wrong plan. Used both per-element (a narrow, physically-correct window) and beamline-wide (`[250, 2000]`, for `out_of_range_element` cases) |
| `if_update_scan_center_near` | `update_scan` x/y_center is within `tol` µm of the target particle (N/A if agent used `load_intelligence_particles` for positioning, or stated no coordinate) |
| `max_calls: {tool: N}` | Tool called at most N times — catches blind-retry loops on errors |
| `scan_type_param: str` | Any `update_scan` call sets `scan_type` to the expected value |
| `scan_type_valid: true` | Any `update_scan` `scan_type` is one of the instrument's real configured types, not a colloquial/invalid guess (e.g. "z-stack") — the mock always accepts `update_scan` regardless of `scan_type`, so this checks the *agent* avoided an invalid type rather than relying on (mocked-away) server rejection |
| `no_parallel_start` | `update_scan` and `start_multiregion_scan` are in different LLM iterations — agent must see the update result before firing the scan |
| `energy_spectrum_min_points: N` | `energy_points` or `len(energy_list)` is at least N — distinguishes a spectrum from a two-point elemental map |
| `update_scan_nonempty` | `update_scan` is called with at least one non-empty argument — catches `update_scan({})` confusion from stale server state |

Only `energy_in_range` applies to `plan_proposal` stages — the rest require a real `update_scan`
call sequence that a plan stage doesn't have.

## Design principle: assertions are floor checks, not quality judgments

A passing test here means "no known-bad behavior was detected" — the same thing a green test
suite means in ordinary software: it does not mean the response was the *best possible* one,
just that nothing on our checklist of known failure modes fired. Keep that distinction explicit
when writing `expected` blocks, especially for ambiguous/underspecified scenarios.

**Example: `image_scan_vague`** ("scan my sample", no area/energy/scan-type given at all).
There is no single correct tool call here — asking a clarifying question (no tool calls at all)
and proposing sensible defaults via `propose_plan` (a real tool call) are *equally legitimate*
responses; forcing a `must_call` would have penalized whichever path a given model didn't take.
The stage has **no `must_call`** at all — only `forbidden_tools: ["start_scan",
"start_multiregion_scan"]`. `right_approach` is simply N/A here; `avoids_wrong` is the only
signal that matters: did the agent run the experiment blindly without any established
parameters? That's the one thing worth automatically, deterministically catching.

**Why not also grade whether the clarifying question was any *good*?** That's a fundamentally
different kind of check — one that requires interpretation, not just tool-call inspection. A
keyword heuristic ("does the response contain a `?`") doesn't actually solve that: it's still
just a floor check wearing a disguise, and it can be gamed by a non-answer that happens to end
in a question mark. The eval-audit philosophy this project follows is explicit about this split:
- **Floor/regression checks** (did it avoid the specific known-bad action?) — deterministic,
  binary, cheap, run on every trace. This is what `forbidden_tools`/`avoids_wrong` is for.
- **Quality/usefulness judgment** (was this actually a *good*, helpful response?) — reserve this
  for either a human reading the actual transcript, or a properly *validated* LLM judge
  (calibrated against human labels — TPR/TNR, not vibes). Don't fold it into the tool-call
  assertions; it's a different job with a different failure mode (a bad judge produces
  confident-looking numbers that mean nothing).

At this suite's current scale (44 cases × 14 models), manual spot-review of surprising
results is cheap and more trustworthy than a brittle stand-in heuristic — that's the actual
practice this project follows when a pass/fail rate looks off: read the raw trace
(`tool_calls` + `final_text` in `traces.jsonl`) before concluding anything, real problem or
assertion problem. Only invest in a validated LLM judge if/when manual review genuinely stops
scaling (hundreds of models/cases, not tens).

**Example: `find_o_particle_vague` / `find_ni_particle_vague`** (underspecified plan_detail --
"find the oxygen/nickel particles" with no energy detail). The *point* of the underspecified
variant is testing whether the agent has the right domain knowledge -- does it know the correct
edge/pre-edge energies for this element -- not whether it self-executes the survey after already
being told "Yes, please proceed." A model that calls `propose_plan` with the right energies and
then asks one more time before configuring anything has demonstrated exactly the understanding
this scenario is meant to test; it just hasn't pulled the trigger yet. So `_find_element_scenario`'s
`vague` branch adds `propose_plan` to `must_call` alongside the real execution path
(`load_intelligence_particles`/`get_intelligence_recommendations`) -- either is acceptable
evidence of correct understanding.

Critically, this widening does **not** mean any `propose_plan` call passes regardless of content:
`evaluate()`'s action-stage `upd` derivation falls back to a real `propose_plan` call's arguments
when no real `update_scan` call exists, so `params_match`'s `energy_in_range`/`energy_list_len`
sub-checks still run against whatever was actually proposed. A confident-but-wrong proposal (the
wrong element's edge, or a scan targeting some other element entirely) still fails -- crediting
"took the right kind of action" is not the same as skipping verification of *what* it proposed.
This mirrors the "specified" scenario's own contract: if there is an execution, the execution
must use the right parameters; here, if there is a proposal, the proposal must use the right
parameters.

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

Format *compliance itself* (`plan_formatted`) is resolved alongside it, and is `True` only when a
real `propose_plan`/`update_scan` call or a `` ```json `` block was found (branches 1–3 above).
Falling through to the regex (branch 4) always yields `None` (N/A) now, **never** `False` — an
earlier version of this check flagged `False` whenever the regex found a bare number in the reply
with no structured call behind it, on the theory that stating concrete numbers in unstructured
prose is itself an instruction-following failure. In practice this over-fired: a correct refusal
routinely cites the instrument's *current* state to justify itself ("Configured motors are only:
**Energy** (600.0 eV)..." after `get_config()`) or its *general* capability range ("photon energies
typically between 100-2000 eV"), and the regex can't reliably tell that apart from a genuine sloppy
commitment to new scan numbers — see the git history around this file for the concrete traces that
exposed it (found via `anthropic/claude-opus`'s `unsupported_rotation` case, where all 3 runs were
correct, clean refusals but scored `plan_formatted: False`). `right_approach`, `avoids_wrong`, and
`dispatchable` already independently cover whether the agent's actual behavior was correct, so
losing this one weak signal costs little.

The trade-off: a model that genuinely *does* commit to concrete numbers in prose without ever
calling `propose_plan` (e.g. "accept my defaults of 2465 eV / 2475 eV?") now scores `plan_formatted:
None` instead of `False` — that specific miss is no longer caught by this check. It would still be
visible by reading `final_text` directly, or by a future, narrower signal (e.g. flag `False` only
when the prose explicitly offers the numbers as a ready-to-execute choice, not merely as a
current-state or range citation) if it turns out to matter enough to rebuild.

`energy_in_range` (a `params_match` sub-check, not `plan_formatted`) still uses the regex-scraped
energies when no real call exists, and remains N/A (not a fail) whenever no energies are found at
all — that part is unchanged, so a vague-but-not-wrong plan is never penalized on `params_match`
either.

## Adding new test cases

1. Add a line to `tuples.jsonl` with the new `task`, `task_type`, `complication`, `element`, and `notes`.
2. Add the corresponding stage logic to `build_inputs.py` (`_stages()`, `_intent()`,
   `_mock_overrides()`, `_plan_expected()` as needed).
3. Run `build_inputs.py` to regenerate `tasks.jsonl`.
4. Run the full pipeline (steps 2–4 above) to see results.
