"""
Step 5 (optional): build an interactive per-model, per-assertion HTML report from results.jsonl.

Unlike report.py's heatmap (one row per stage x model), this report answers a narrower
question well: for a given model, which of the 6 assertions fails on which test case? Rows are
the 44 cases (grouped the same way as the README); columns are the 6 assertions; a dropdown
switches which model's data fills the table, entirely client-side (all models' data is embedded
inline, so the page has no server and works offline). Each cell reads passes/applicable, pooled
across every stage of that case and every accumulated run for the selected model. Group and
overall percentages are an unweighted mean across cases, not a pool of every individual check --
a case with 12 checks doesn't outvote one with 2.

Usage:
    python evals/task_agent/generate_assertion_report.py
    # -> writes evals/task_agent/assertion_report.html

    python evals/task_agent/generate_assertion_report.py --results results_v2.jsonl --output report_v2.html
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from build_inputs import TUPLES_PATH, _tool_unit_cases
from run_assertions import ORDER as ASSERTIONS

HERE = Path(__file__).parent
DEFAULT_RESULTS = HERE / "results.jsonl"
DEFAULT_OUTPUT = HERE / "assertion_report.html"

# _tool_unit_cases() (build_inputs.py) is the source of truth for which tool_unit tasks exist,
# but it only carries {task, kind, user, expected} -- no task_type/complication/element/
# plan_detail, since those don't apply to an isolated single-call test. Fill in matching display
# metadata by hand; the "unrecognized task" check in build_task_meta() below fails loudly if a
# new tool_unit case is added here without a matching entry.
TOOL_UNIT_DISPLAY_META = {
    "tool_unit_elemental_map": {"task_type": "tool_unit", "complication": "isolated_call", "element": None, "plan_detail": "specified"},
    "tool_unit_line_spectrum": {"task_type": "tool_unit", "complication": "isolated_call", "element": None, "plan_detail": "specified"},
    "tool_unit_image_scan":    {"task_type": "tool_unit", "complication": "isolated_call", "element": None, "plan_detail": "specified"},
}

# Presentation grouping for the report -- purely cosmetic, has no effect on scoring. Update this
# alongside tuples.jsonl when adding a new case; build_task_meta() asserts every known task is
# covered by exactly one group so a forgotten entry fails the build instead of silently vanishing
# from the report.
GROUPS = [
    ("Fe baseline", "Original 10 scenarios — elemental maps, line spectra, and stale-state handling on iron.",
     ["find_fe_particle", "no_particles_found", "scan_error", "fe_spectrum", "fe_spectrum_no_recs",
      "stale_scan_params", "spectrum_given_particle", "find_fe_particle_vague", "fe_spectrum_vague",
      "spectrum_given_particle_vague"]),
    ("Element coverage — specified", "Two-energy elemental maps spanning K-, L-, and M-edge types, full physics detail given.",
     ["find_c_particle", "find_o_particle", "find_ni_particle", "find_cu_particle", "find_ce_particle"]),
    ("Element coverage — underspecified", "Same elements, bare ask — agent must derive the correct edge itself.",
     ["find_c_particle_vague", "find_o_particle_vague", "find_ni_particle_vague", "find_cu_particle_vague", "find_ce_particle_vague"]),
    ("Image scan basics", "Plain image scan, no element.",
     ["image_scan_basic", "image_scan_vague"]),
    ("Tool error handling", "Software-side tool failure mid-survey or mid-follow-up.",
     ["elemental_map_survey_error", "elemental_map_followup_error"]),
    ("Hardware error handling", "Motor contention, interlock, DAQ fault, stage limit switch.",
     ["hw_motor_occupied", "hw_daq_fault", "hw_shutter_interlock", "hw_stage_limit", "hw_energy_motor_fault_followup"]),
    ("Minimal plan detail", '"Show me my sample" tier — near-zero detail.',
     ["show_me_sample", "find_iron_bare", "find_copper_bare"]),
    ("Beyond the beamline's range", "Element edge or explicit energy outside 250–2000 eV.",
     ["sulfur_out_of_range", "fe_k_edge_confusion", "phosphorus_boundary", "explicit_energy_out_of_range"]),
    ("Invalid requests", "Colloquial scan type, unsupported motion axis, non-STXM ask.",
     ["invalid_scan_type", "unsupported_rotation", "non_stxm_request"]),
    ("Mid-conversation change", "User redirects partway through, after real context is already established.",
     ["element_switch_mid_task", "task_switch_mid_task"]),
    ("Tool-use unit tests", "Single explicit instruction, one fresh agent, no plan/confirm turn — isolates raw tool-call accuracy from conversational reasoning.",
     ["tool_unit_elemental_map", "tool_unit_line_spectrum", "tool_unit_image_scan"]),
]

TYPE_LABEL = {"elemental_map": "Elemental", "image_scan": "Image", "line_spectrum": "Line spec", "tool_unit": "Tool unit"}
COMP_LABEL = {
    "happy_path": "happy path", "no_recs": "no recs", "scan_error": "scan error", "stale_params": "stale params",
    "given_particle": "given particle", "tool_error": "tool error", "hardware_error": "hardware error",
    "out_of_range_element": "out of range", "invalid_request": "invalid request", "mid_task_change": "mid-task change",
    "isolated_call": "isolated call",
}
ELEM_EDGE = {"Fe": "L", "Ni": "L", "Cu": "L", "C": "K", "O": "K", "Ce": "M", "S": "–", "P": "–"}
ASSERTION_LABEL = {
    "well_formed": "Well-formed", "plan_formatted": "Plan fmt.", "right_approach": "Right tool",
    "avoids_wrong": "Avoids wrong", "params_match": "Params match", "dispatchable": "Dispatchable",
}


def build_task_meta() -> dict:
    """{task_name: {task_type, complication, element, plan_detail}} for every case, scenario or
    tool-unit alike -- and validates that GROUPS covers exactly this set, so a case added to
    tuples.jsonl (or a group edited) without updating the other can't silently drop off the report."""
    tuples = [json.loads(l) for l in TUPLES_PATH.read_text().splitlines() if l.strip()]
    meta = {t["task"]: {"task_type": t["task_type"], "complication": t["complication"],
                         "element": t["element"], "plan_detail": t["plan_detail"]} for t in tuples}
    tool_unit_names = {c["task"] for c in _tool_unit_cases()}
    if tool_unit_names != set(TOOL_UNIT_DISPLAY_META):
        raise SystemExit(f"TOOL_UNIT_DISPLAY_META is out of sync with build_inputs._tool_unit_cases(): "
                          f"{tool_unit_names ^ set(TOOL_UNIT_DISPLAY_META)}")
    meta.update(TOOL_UNIT_DISPLAY_META)

    grouped = [t for _, _, tasks in GROUPS for t in tasks]
    if len(grouped) != len(set(grouped)):
        dupes = {t for t in grouped if grouped.count(t) > 1}
        raise SystemExit(f"Task(s) listed in more than one GROUPS entry: {dupes}")
    if set(grouped) != set(meta):
        raise SystemExit(f"GROUPS in generate_assertion_report.py is out of sync with the task set "
                          f"(tuples.jsonl + tool-unit cases): {set(grouped) ^ set(meta)}")
    return meta


def build_matrix(results_path: Path, task_meta: dict) -> dict:
    """{model: {task: {assertion: [pass, applicable]}}}, pooled across every stage and run."""
    results = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    models = sorted(set(r["model"] for r in results))
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for r in results:
        m, task = r["model"], r["task"]
        for a in ASSERTIONS:
            v = r["assertions"].get(a)
            if v is None:
                continue
            agg[m][task][a][1] += 1
            if v:
                agg[m][task][a][0] += 1
    return {
        "models": models,
        "data": {m: {task: {a: agg[m][task][a] for a in ASSERTIONS} for task in task_meta} for m in models},
    }


def render_rows(task_meta: dict) -> tuple[str, int]:
    rows, n = [], 0
    for gname, gdesc, tasks in GROUPS:
        task_list = "|".join(tasks)
        rows.append(f'''
      <tr class="group-row">
        <td colspan="{6 + len(ASSERTIONS)}">
          <div class="group-head">
            <span class="group-name">{gname}</span>
            <span class="group-desc">{gdesc}</span>
            <span class="group-stats" data-group="{task_list}"><span class="chip chip-na">–</span><span class="chip-n">· {len(tasks)} cases</span></span>
          </div>
        </td>
      </tr>''')
        for task in tasks:
            n += 1
            meta = task_meta[task]
            elem = meta["element"]
            edge = ELEM_EDGE.get(elem, "")
            elem_html = "–" if not elem else (f'{elem} <span style="opacity:.55">{edge}</span>' if edge and edge != "–" else elem)
            cells = "".join(
                f'<td class="result acell" data-task="{task}" data-a="{a}"><span class="chip chip-na">–</span></td>'
                for a in ASSERTIONS
            )
            rows.append(f'''
      <tr>
        <td class="num">{n:02d}</td>
        <td class="task"><code>{task}</code></td>
        <td class="dim">{TYPE_LABEL[meta["task_type"]]}</td>
        <td class="dim">{COMP_LABEL[meta["complication"]]}</td>
        <td class="elem">{elem_html}</td>
        <td class="dim">{meta["plan_detail"]}</td>
        {cells}
      </tr>''')
    return "".join(rows), n


MODEL_LABEL_OVERRIDES = {
    # Cosmetic display names for the model picker; any model not listed here falls back to its
    # raw id, so this needs no maintenance when a new model is added to run_eval_multi.MODELS.
    "anthropic/claude-opus": "Claude Opus",
    "anthropic/claude-sonnet": "Claude Sonnet",
    "gemini-3.1-pro-high": "Gemini 3.1 Pro (high)",
    "gemini-pro": "Gemini Pro",
    "amazon/gpt-5.5-medium": "GPT-5.5-medium",
    "openai/gpt-5.5-medium": "GPT-5.5-medium (OpenAI)",
    "google/grok-4.3": "Grok 4.3",
    "google/glm-5": "GLM-5",
    "xai/grok-4.20-reasoning": "Grok 4.20 (reasoning)",
    "google/gemma-4": "Gemma 4",
    "google/qwen-3": "Qwen 3",
    "devstral-2": "Devstral 2",
    "google/deepseek-r1": "DeepSeek R1",
    "amazon/gpt-oss-20b": "GPT-OSS-20B",
    "amazon/llama-4-scout": "Llama 4 Scout",
    "nemotron-nano-3": "Nemotron Nano 3",
}


def render_html(matrix: dict, task_meta: dict) -> str:
    rows_html, n_tasks = render_rows(task_meta)
    header_html = "".join(f'<th class="result-col" title="{ASSERTION_LABEL[a]}">{ASSERTION_LABEL[a]}</th>' for a in ASSERTIONS)

    # Best-performing model first -- overall unweighted mean across cases, same metric the page
    # itself computes client-side (kept in sync manually; see render()'s taskRate/meanRate in JS).
    def overall_rate(model):
        rates = []
        for task in task_meta:
            p = tot = 0
            for a in ASSERTIONS:
                pp, tt = matrix["data"][model][task][a]
                p += pp
                tot += tt
            if tot:
                rates.append(p / tot)
        return sum(rates) / len(rates) if rates else 0.0

    model_order = sorted(matrix["models"], key=overall_rate, reverse=True)
    options_html = "".join(
        f'<option value="{m}">{MODEL_LABEL_OVERRIDES.get(m, m)}</option>' for m in model_order
    )

    data_js = json.dumps(matrix["data"], separators=(",", ":"))
    groups_js = json.dumps([{"name": g[0], "tasks": g[2]} for g in GROUPS], separators=(",", ":"))
    assertions_js = json.dumps(ASSERTIONS)
    n_models = len(matrix["models"])

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TaskAgent Eval Suite — Assertion Breakdown</title>
<style>
  :root {{
    --bg: #F3F5F1; --surface: #FFFFFF; --surface-2: #E9EDE7;
    --text: #14201C; --text-dim: #52645C; --line: #D7DED5;
    --accent: #B5701F; --accent-soft: #B5701F1A;
    --pass: #2F7A50; --pass-soft: #2F7A501A;
    --fail: #B23B32; --fail-soft: #B23B321A;
    --warn: #96792A; --warn-soft: #96792A1F;
    --shadow: rgba(20, 32, 28, 0.08);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #0B1210; --surface: #101A17; --surface-2: #17221E;
      --text: #E7EFEA; --text-dim: #8FA69C; --line: #253530;
      --accent: #E8A23D; --accent-soft: #E8A23D26;
      --pass: #6FBF8B; --pass-soft: #6FBF8B22;
      --fail: #E2645A; --fail-soft: #E2645A22;
      --warn: #D9C15C; --warn-soft: #D9C15C22;
      --shadow: rgba(0, 0, 0, 0.45);
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #0B1210; --surface: #101A17; --surface-2: #17221E;
    --text: #E7EFEA; --text-dim: #8FA69C; --line: #253530;
    --accent: #E8A23D; --accent-soft: #E8A23D26;
    --pass: #6FBF8B; --pass-soft: #6FBF8B22;
    --fail: #E2645A; --fail-soft: #E2645A22;
    --warn: #D9C15C; --warn-soft: #D9C15C22;
    --shadow: rgba(0, 0, 0, 0.45);
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 56px 32px 96px; }}
  .eyebrow {{ font-family: "SF Mono", "Cascadia Code", Consolas, Menlo, monospace; font-size: 12px;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); font-weight: 600; }}
  h1 {{ font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif; font-size: 40px;
    line-height: 1.15; font-weight: 600; margin: 10px 0 14px; max-width: 24ch; text-wrap: balance;
    letter-spacing: -0.01em; }}
  .subtitle {{ font-size: 16px; line-height: 1.6; color: var(--text-dim); max-width: 68ch; margin: 0 0 8px; }}
  .run-meta {{ font-family: "SF Mono", "Cascadia Code", Consolas, Menlo, monospace; font-size: 12.5px;
    color: var(--text-dim); margin-top: 18px; display: flex; flex-wrap: wrap; gap: 6px 18px; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--line);
    border: 1px solid var(--line); border-radius: 10px; overflow: hidden; margin: 40px 0 32px;
    box-shadow: 0 1px 2px var(--shadow); }}
  .stat {{ background: var(--surface); padding: 22px 22px 18px; }}
  .stat-num {{ font-family: "SF Mono", "Cascadia Code", Consolas, Menlo, monospace;
    font-variant-numeric: tabular-nums; font-size: 30px; font-weight: 600; letter-spacing: -0.02em; }}
  .stat-num.accent {{ color: var(--accent); }}
  .stat-label {{ font-size: 12.5px; color: var(--text-dim); margin-top: 4px; }}
  .picker {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin: 0 0 28px;
    padding: 18px 22px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }}
  .picker label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent); font-weight: 700; }}
  .picker select {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14.5px; font-weight: 600; color: var(--text); background: var(--surface-2);
    border: 1px solid var(--line); border-radius: 7px; padding: 8px 34px 8px 12px; appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%2352645C'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 12px center; cursor: pointer; }}
  .picker select:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
  .picker .picker-note {{ font-size: 12.5px; color: var(--text-dim); margin-left: auto; }}
  .legend {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 0 0 48px;
    padding: 22px 24px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }}
  .legend-item h3 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent);
    margin: 0 0 6px; font-weight: 700; }}
  .legend-item p {{ margin: 0; font-size: 13px; line-height: 1.55; color: var(--text-dim); }}
  .legend-item code {{ color: var(--text); }}
  code {{ font-family: "SF Mono", "Cascadia Code", Consolas, Menlo, monospace; font-size: 0.92em; }}
  h2.section-title {{ font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    font-size: 22px; font-weight: 600; margin: 0 0 18px; display: flex; align-items: baseline; gap: 12px; }}
  h2.section-title .current-model {{ font-family: "SF Mono", "Cascadia Code", Consolas, Menlo, monospace;
    font-size: 13px; font-weight: 600; color: var(--accent); background: var(--accent-soft);
    padding: 3px 10px; border-radius: 100px; }}
  .table-scroll {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 10px;
    background: var(--surface); box-shadow: 0 1px 2px var(--shadow); }}
  table {{ border-collapse: collapse; width: 100%; min-width: 980px; }}
  thead th {{ position: sticky; top: 0; background: var(--surface-2); text-align: left; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-dim); font-weight: 600;
    padding: 12px 14px; border-bottom: 1px solid var(--line); z-index: 1; }}
  thead th.result-col {{ text-align: right; white-space: nowrap; }}
  tbody tr:not(.group-row) {{ border-bottom: 1px solid var(--line); }}
  tbody tr:not(.group-row):hover {{ background: var(--surface-2); }}
  tbody tr:last-child {{ border-bottom: none; }}
  td {{ padding: 9px 14px; font-size: 13.5px; vertical-align: middle; white-space: nowrap; }}
  td.num {{ font-family: "SF Mono", "Cascadia Code", Consolas, Menlo, monospace; color: var(--text-dim);
    font-variant-numeric: tabular-nums; width: 1%; }}
  td.task code {{ font-weight: 500; }}
  td.dim {{ color: var(--text-dim); font-size: 13px; }}
  td.elem {{ font-family: "SF Mono", "Cascadia Code", Consolas, Menlo, monospace; font-weight: 700; letter-spacing: 0.01em; }}
  td.result {{ text-align: right; }}
  .chip {{ display: inline-flex; align-items: center; gap: 6px;
    font-family: "SF Mono", "Cascadia Code", Consolas, Menlo, monospace; font-variant-numeric: tabular-nums;
    font-size: 12.5px; font-weight: 600; padding: 3px 9px; border-radius: 100px; white-space: nowrap; }}
  .chip-pass {{ background: var(--pass-soft); color: var(--pass); }}
  .chip-warn {{ background: var(--warn-soft); color: var(--warn); }}
  .chip-fail {{ background: var(--fail-soft); color: var(--fail); }}
  .chip-na {{ background: var(--surface-2); color: var(--text-dim); }}
  .chip-n {{ font-weight: 400; opacity: 0.75; }}
  .group-row td {{ padding: 0; white-space: normal; }}
  .group-head {{ display: flex; align-items: baseline; gap: 14px; padding: 16px 14px 10px; border-top: 1px solid var(--line); }}
  tbody tr.group-row:first-child .group-head {{ border-top: none; }}
  .group-name {{ font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif; font-size: 16px;
    font-weight: 600; flex-shrink: 0; }}
  .group-desc {{ font-size: 12.5px; color: var(--text-dim); flex: 1; }}
  .group-stats {{ flex-shrink: 0; display: flex; align-items: center; gap: 6px; }}
  footer {{ margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--line); font-size: 12.5px;
    color: var(--text-dim); line-height: 1.6; }}
  @media (max-width: 720px) {{
    .wrap {{ padding: 36px 18px 64px; }} h1 {{ font-size: 30px; }}
    .stats {{ grid-template-columns: repeat(2, 1fr); }} .legend {{ grid-template-columns: repeat(2, 1fr); }}
    .group-head {{ flex-wrap: wrap; }} .picker .picker-note {{ margin-left: 0; }}
  }}
</style>
</head>
<body>
<div class="wrap">

  <div class="eyebrow">ALS 7.0.1.2 · STXM TaskAgent Evals</div>
  <h1>Where Each Model's Assertions Actually Fail</h1>
  <p class="subtitle">
    {n_tasks} test cases × {len(ASSERTIONS)} deterministic assertion checks, scored against the real agent
    running a scripted mock instrument. Pick a model below to see exactly which assertion breaks
    down on which case — cells read <b>passes / applicable checks</b> across all stages and
    accumulated runs for that case.
  </p>
  <div class="run-meta">
    <span>{n_models} models scored</span>
    <span>{n_tasks} cases</span>
    <span>generated by generate_assertion_report.py</span>
  </div>

  <div class="stats">
    <div class="stat"><div class="stat-num">{n_tasks}</div><div class="stat-label">Test cases</div></div>
    <div class="stat"><div class="stat-num">{len(ASSERTIONS)}</div><div class="stat-label">Assertion checks</div></div>
    <div class="stat"><div class="stat-num">{n_models}</div><div class="stat-label">Models compared</div></div>
    <div class="stat"><div class="stat-num accent" id="stat-overall">—</div>
      <div class="stat-label" id="stat-overall-label">Overall pass rate, selected model</div></div>
  </div>

  <div class="picker">
    <label for="model-select">Model</label>
    <select id="model-select">{options_html}</select>
    <span class="picker-note">Switches every cell below · sorted by overall pass rate, best first</span>
  </div>

  <div class="legend">
    <div class="legend-item"><h3>Task type</h3>
      <p>What the scientist is asking for — <code>elemental map</code> (two-energy contrast scan),
        <code>image scan</code> (plain scan), <code>line spectrum</code> (oxidation-state ratio on a
        found particle), or <code>tool unit</code> (one explicit instruction, one tool call, no
        conversation).</p></div>
    <div class="legend-item"><h3>Complication</h3>
      <p>What makes it hard: nothing (<code>happy path</code>), a software or hardware fault, stale
        server state, a request the beamline physically can't satisfy, or a mid-task redirect after
        real context is already established.</p></div>
    <div class="legend-item"><h3>Element</h3>
      <p>Fe, C, O, Ni, Cu, Ce span all three edge types this beamline reaches — K (C, O), L (Fe, Ni,
        Cu), M (Ce). S and P sit outside the 250–2000&nbsp;eV range — used only to test whether the
        agent notices.</p></div>
    <div class="legend-item"><h3>Assertion columns</h3>
      <p><code>well_formed</code> parsed cleanly · <code>right_approach</code> called the expected
        tool · <code>params_match</code> got the specific parameters right · <code>dispatchable</code>
        the call actually executes against the real ToolSet.</p></div>
  </div>

  <h2 class="section-title">All {n_tasks} Cases <span class="current-model" id="current-model-badge">—</span></h2>

  <div class="table-scroll">
    <table>
      <thead><tr>
        <th>#</th><th>Task</th><th>Type</th><th>Complication</th><th>Element</th><th>Detail</th>
        {header_html}
      </tr></thead>
      <tbody>
      {rows_html}
      </tbody>
    </table>
  </div>

  <footer>
    Each cell reads <b>passes / applicable checks</b> for that assertion, pooled across every stage
    of that task and every accumulated run for the selected model. A dash (–) means the check never
    applied to that task (e.g. <code>plan_formatted</code> only applies to plan-proposal stages;
    <code>avoids_wrong</code> only applies when a task defines <code>forbidden_tools</code>). N/A
    checks are excluded from the denominator, not counted as failures. The group and overall
    percentages are an <b>unweighted mean across cases</b>, not a pool of every individual check —
    each case's own passes/applicable are combined into one rate first, then every case counts
    equally regardless of how many stages or runs fed it, so a case with 12 checks doesn't outvote
    one with 2. See README.md's "Assertions" section for what each check and sub-check means.
  </footer>

</div>

<script>
  const DATA = {data_js};
  const GROUPS = {groups_js};
  const ASSERTIONS = {assertions_js};

  function classify(p, tot) {{
    if (tot === 0) return {{cls: "chip-na", text: "–"}};
    if (p === tot) return {{cls: "chip-pass", text: p + "/" + tot}};
    if (p === 0) return {{cls: "chip-fail", text: p + "/" + tot}};
    return {{cls: "chip-warn", text: p + "/" + tot}};
  }}

  // A case's own rate pools its checks (a case can have several stages/runs feeding one
  // assertion), but every CASE counts equally toward a group/overall rate regardless of how
  // many checks it happens to carry -- a 12-check case and a 2-check case each cast one vote.
  function taskRate(modelData, task) {{
    let p = 0, tot = 0;
    ASSERTIONS.forEach(a => {{
      const cell = (modelData[task] && modelData[task][a]) || [0, 0];
      p += cell[0]; tot += cell[1];
    }});
    return tot === 0 ? null : p / tot;
  }}

  function meanRate(rates) {{
    const valid = rates.filter(r => r !== null);
    if (!valid.length) return null;
    return valid.reduce((a, b) => a + b, 0) / valid.length;
  }}

  function render(model) {{
    const modelData = DATA[model] || {{}};

    document.querySelectorAll("td.acell").forEach(td => {{
      const task = td.dataset.task, a = td.dataset.a;
      const cell = (modelData[task] && modelData[task][a]) || [0, 0];
      const [p, tot] = cell;
      const {{cls, text}} = classify(p, tot);
      td.innerHTML = '<span class="chip ' + cls + '">' + text + '</span>';
    }});

    const allTasks = Object.keys(modelData);

    document.querySelectorAll(".group-stats").forEach(span => {{
      const tasks = span.dataset.group.split("|");
      const rate = meanRate(tasks.map(t => taskRate(modelData, t)));
      const {{cls, text}} = rate === null ? {{cls: "chip-na", text: "–"}}
        : {{cls: rate === 1 ? "chip-pass" : rate === 0 ? "chip-fail" : "chip-warn",
            text: Math.round(100 * rate) + "%"}};
      span.innerHTML = '<span class="chip ' + cls + '">' + text + '</span><span class="chip-n">· '
        + tasks.length + ' case' + (tasks.length > 1 ? "s" : "") + '</span>';
    }});

    const overallRate = meanRate(allTasks.map(t => taskRate(modelData, t)));
    const pct = overallRate === null ? 0 : Math.round(1000 * overallRate) / 10;
    document.getElementById("stat-overall").textContent = pct + "%";
    document.getElementById("stat-overall-label").textContent =
      "Overall pass rate — unweighted mean across " + allTasks.length + " cases";
    document.getElementById("current-model-badge").textContent = model;
  }}

  const select = document.getElementById("model-select");
  select.addEventListener("change", () => render(select.value));
  render(select.value);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Path to results.jsonl")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write the HTML report")
    args = ap.parse_args()

    if not args.results.exists():
        raise SystemExit(f"{args.results} not found -- run run_assertions.py first")

    task_meta = build_task_meta()
    matrix = build_matrix(args.results, task_meta)
    if not matrix["models"]:
        raise SystemExit(f"{args.results} has no rows -- nothing to report")

    args.output.write_text(render_html(matrix, task_meta))
    print(f"Wrote {args.output}  ({len(matrix['models'])} models x {len(task_meta)} cases)")


if __name__ == "__main__":
    main()
