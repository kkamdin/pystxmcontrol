"""
Generate report.html from results.jsonl.

Shows a binary criteria matrix with rows = task / stage and columns = assertion.
When params_match fails, the cell tooltip shows which sub-checks failed.

Usage:
    python evals/task_agent/report.py
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results.jsonl"
RUNS_META_PATH = Path(__file__).parent / "runs_meta.jsonl"
REPORT_PATH = Path(__file__).parent / "report.html"

ASSERTION_ORDER = ["well_formed", "plan_formatted", "right_approach", "params_match", "dispatchable"]
ASSERTION_ABBR = {
    "well_formed": "WF",
    "plan_formatted": "PF",
    "right_approach": "RA",
    "params_match": "PMS",
    "dispatchable": "DISP",
}
ASSERTION_DESC = {
    "well_formed": "No API error; tool args parse correctly",
    "plan_formatted": "Plan stage: used propose_plan (or a JSON block) when concrete params were stated",
    "right_approach": "Called a tool listed in must_call",
    "params_match": "update_scan params match the expected strategy",
    "dispatchable": "Every tool call is executable by the real ToolSet",
}


def _color(rate: float | None) -> str:
    if rate is None:
        return "#e5e7eb"
    if rate == 1.0:
        return "#22c55e"
    if rate == 0.0:
        return "#ef4444"
    if rate < 0.5:
        g = int(rate * 2 * 251)
        return f"rgb(239,{g},68)"
    r = int((1 - rate) * 2 * 239)
    return f"rgb({r},197,68)"


def _text_color(rate: float | None) -> str:
    if rate is None:
        return "#6b7280"
    return "#fff" if rate in (0.0, 1.0) else "#1f2937"


def _bar(rate: float, width_px: int = 180) -> str:
    filled = int(rate * width_px)
    pct = f"{rate * 100:.0f}%"
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="width:{width_px}px;height:14px;background:#e5e7eb;border-radius:3px;overflow:hidden">'
        f'<div style="width:{filled}px;height:100%;background:#22c55e"></div></div>'
        f'<span style="font-size:12px;color:#374151">{pct}</span>'
        f'</div>'
    )


def _th(text: str, title: str = "", rotate: bool = False) -> str:
    if rotate:
        style = ('style="writing-mode:vertical-rl;transform:rotate(180deg);'
                 'white-space:nowrap;padding:8px 4px;font-size:11px;font-weight:600;'
                 'color:#374151;text-align:left"')
    else:
        style = ('style="padding:8px 12px;font-size:12px;font-weight:600;'
                 'color:#6b7280;text-align:left;white-space:nowrap"')
    title_attr = f' title="{title}"' if title else ""
    return f"<th {style}{title_attr}>{text}</th>"


def _td_label(text: str, sub: str = "") -> str:
    sub_html = f'<br><span style="font-size:10px;color:#9ca3af">{sub}</span>' if sub else ""
    return (f'<td style="padding:5px 12px;font-size:12px;color:#374151;'
            f'white-space:nowrap;border-right:1px solid #e5e7eb">{text}{sub_html}</td>')


def _td_cell(counts: tuple[int, int] | None, tooltip: str = "") -> str:
    rate = counts[0] / counts[1] if counts is not None else None
    bg = _color(rate)
    fg = _text_color(rate)
    if counts is None:
        label = "—"
    else:
        n_pass, n_app = counts
        label = ("✓" if n_pass == n_app else "✗") if n_app == 1 else f"{n_pass}/{n_app}"
    title_attr = f' title="{tooltip}"' if tooltip else ""
    return (f'<td style="text-align:center;padding:4px;background:{bg};'
            f'color:{fg};font-size:13px;font-weight:600;'
            f'min-width:44px;border:1px solid #fff"{title_attr}>{label}</td>')


def main() -> None:
    rows = [json.loads(l) for l in RESULTS_PATH.read_text().splitlines() if l.strip()]
    if not rows:
        print("No results — run run_assertions.py first.")
        return

    # Group rows by run_id.
    by_run: dict[str, list] = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    run_ids = sorted(by_run.keys())

    # Determine the canonical (task, stage) order from the first run.
    first_run = by_run[run_ids[0]]
    stage_order = [(r["task"], r["stage"]) for r in first_run]
    # Merge any (task, stage) pairs that appeared in later runs but not the first.
    seen = set(stage_order)
    for rid in run_ids[1:]:
        for r in by_run[rid]:
            key = (r["task"], r["stage"])
            if key not in seen:
                stage_order.append(key)
                seen.add(key)

    # Build cell_data: {(task, stage, assertion): (n_pass, n_applicable) | None}
    cell_data: dict[tuple, tuple[int, int] | None] = {}
    # Also collect params_detail failures for tooltips.
    detail_failures: dict[tuple, set[str]] = defaultdict(set)

    for (task, stage) in stage_order:
        stage_rows = [r for r in rows if r["task"] == task and r["stage"] == stage]
        for aname in ASSERTION_ORDER:
            vals = [r["assertions"].get(aname) for r in stage_rows]
            applicable = [v for v in vals if v is not None]
            cell_data[(task, stage, aname)] = (
                (sum(applicable), len(applicable)) if applicable else None
            )
        # Collect all params_detail sub-checks that ever failed for this stage.
        for r in stage_rows:
            pd = r["assertions"].get("params_detail") or {}
            for k, v in pd.items():
                if v is False:
                    detail_failures[(task, stage)].add(k)

    # Per-assertion overall pass rate.
    assertion_rates: dict[str, float | None] = {}
    for aname in ASSERTION_ORDER:
        cells = [v for (_, _, an), v in cell_data.items() if an == aname and v is not None]
        if not cells:
            assertion_rates[aname] = None
        else:
            tp = sum(p for p, _ in cells)
            ta = sum(a for _, a in cells)
            assertion_rates[aname] = tp / ta if ta else None

    # Per-task pass rate (across all stages and assertions).
    task_names = list(dict.fromkeys(task for task, _ in stage_order))
    task_rates: dict[str, float] = {}
    for task in task_names:
        stage_rows = [r for r in rows if r["task"] == task]
        vals = [v for r in stage_rows
                for aname in ASSERTION_ORDER
                for v in [r["assertions"].get(aname)] if v is not None]
        task_rates[task] = sum(vals) / len(vals) if vals else 0.0

    # Load runs_meta.
    runs_meta: dict[str, dict] = {}
    if RUNS_META_PATH.exists():
        for line in RUNS_META_PATH.read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                runs_meta[m["run_id"]] = m

    # Build per-run summary rows.
    run_rows_html = ""
    for rid in run_ids:
        run_rows = by_run[rid]
        ts = datetime.fromtimestamp(run_rows[0].get("timestamp", 0)).strftime("%Y-%m-%d %H:%M")
        model = run_rows[0].get("model", "—")
        all_vals = [v for r in run_rows
                    for aname in ASSERTION_ORDER
                    for v in [r["assertions"].get(aname)] if v is not None]
        overall = sum(all_vals) / len(all_vals) if all_vals else 0.0
        meta = runs_meta.get(rid, {})
        tok_in = meta.get("total_input_tokens")
        tok_out = meta.get("total_output_tokens")
        cost_data = meta.get("cost_estimate") or {}

        if tok_in is not None:
            n = len(run_rows)
            tok_cell = (f"{tok_in + tok_out:,} total"
                        f"<br><small style='color:#6b7280'>"
                        f"{tok_in // max(n,1):,} in / {tok_out // max(n,1):,} out avg/stage</small>")
        else:
            tok_cell = "—"

        total_cost = cost_data.get("total_usd")
        cost_cell = f"${total_cost:.4f}" if total_cost is not None else "—"

        run_rows_html += (
            f'<tr>'
            f'<td style="font-family:monospace;font-size:12px;padding:6px 12px">{rid}</td>'
            f'<td style="font-size:12px;padding:6px 12px;color:#6b7280">{ts}</td>'
            f'<td style="font-size:12px;padding:6px 12px">{model}</td>'
            f'<td style="font-size:12px;padding:6px 12px;text-align:center">{len(run_rows)}</td>'
            f'<td style="font-size:12px;padding:6px 12px">{tok_cell}</td>'
            f'<td style="font-size:12px;padding:6px 12px">{cost_cell}</td>'
            f'<td style="padding:6px 12px">{_bar(overall)}</td>'
            f'</tr>'
        )

    # Build heatmap rows — one row per (task, stage).
    prev_task = None
    heatmap_rows_html = ""
    for (task, stage) in stage_order:
        task_label = task if task != prev_task else ""
        prev_task = task
        cells_html = ""
        for aname in ASSERTION_ORDER:
            counts = cell_data.get((task, stage, aname))
            tooltip = ""
            if aname == "params_match" and counts is not None and counts[0] < counts[1]:
                failed = sorted(detail_failures.get((task, stage), set()))
                if failed:
                    tooltip = "Failed sub-checks: " + ", ".join(failed)
            cells_html += _td_cell(counts, tooltip)
        heatmap_rows_html += (
            f'<tr>'
            f'{_td_label(task_label, stage)}'
            f'{cells_html}'
            f'</tr>\n'
        )

    # Build assertion summary rows.
    assertion_rows_html = ""
    for aname in ASSERTION_ORDER:
        rate = assertion_rates[aname]
        assertion_rows_html += (
            f'<tr>'
            f'<td style="font-size:12px;font-weight:600;padding:6px 12px;'
            f'font-family:monospace;white-space:nowrap">{aname}</td>'
            f'<td style="font-size:12px;padding:6px 12px;color:#6b7280">{ASSERTION_DESC.get(aname, "")}</td>'
            f'<td style="padding:6px 12px">{_bar(rate if rate is not None else 0)}</td>'
            f'</tr>'
        )

    # Build task summary rows.
    task_rows_html = ""
    for task in task_names:
        stage_count = sum(1 for t, _ in stage_order if t == task)
        task_rows_html += (
            f'<tr>'
            f'<td style="font-size:12px;font-weight:600;padding:6px 12px;'
            f'font-family:monospace">{task}</td>'
            f'<td style="font-size:12px;padding:6px 12px;text-align:center;color:#6b7280">{stage_count}</td>'
            f'<td style="padding:6px 12px">{_bar(task_rates[task])}</td>'
            f'</tr>'
        )

    col_headers = "".join(
        _th(ASSERTION_ABBR[an], ASSERTION_DESC.get(an, an), rotate=True)
        for an in ASSERTION_ORDER
    )
    n_runs = len(run_ids)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Task Agent Eval Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f9fafb; color: #111827; padding: 32px; }}
    h1 {{ font-size: 22px; font-weight: 700; color: #111827; }}
    h2 {{ font-size: 15px; font-weight: 600; color: #374151; margin: 28px 0 12px; }}
    .subtitle {{ font-size: 13px; color: #6b7280; margin-top: 4px; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
             padding: 20px; margin-bottom: 24px; overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    tr:nth-child(even) {{ background: #f9fafb; }}
    th {{ background: #f3f4f6; }}
    .legend {{ display: flex; gap: 16px; font-size: 12px; color: #6b7280; margin-top: 12px; }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .dot {{ width: 12px; height: 12px; border-radius: 2px; }}
    td[title] {{ cursor: help; }}
  </style>
</head>
<body>
  <h1>Task Agent Evals</h1>
  <p class="subtitle">
    Assertion-based binary criteria matrix &nbsp;·&nbsp; ALS 7.0.1.2 STXM
    &nbsp;·&nbsp; {n_runs} run{"s" if n_runs != 1 else ""}
    &nbsp;·&nbsp; Generated {generated}
  </p>

  <h2>Runs</h2>
  <div class="card">
    <table>
      <thead><tr>
        {_th("Run ID")} {_th("Date")} {_th("Model")} {_th("Stages")}
        {_th("Tokens")} {_th("Cost est.")} {_th("Overall pass rate")}
      </tr></thead>
      <tbody>{run_rows_html}</tbody>
    </table>
  </div>

  <h2>Binary Criteria Matrix</h2>
  <p style="font-size:12px;color:#6b7280;margin-bottom:10px">
    Hover a PMS cell to see which param sub-checks failed.
  </p>
  <div class="card">
    <table>
      <thead>
        <tr>
          {_th("Task / Stage")}
          {col_headers}
        </tr>
      </thead>
      <tbody>{heatmap_rows_html}</tbody>
    </table>
    <div class="legend">
      <div class="legend-item"><div class="dot" style="background:#22c55e"></div>
        Pass{"  (all runs)" if n_runs > 1 else ""}
      </div>
      <div class="legend-item"><div class="dot" style="background:#ef4444"></div>
        Fail{"  (all runs)" if n_runs > 1 else ""}
      </div>
      {"<div class='legend-item'><div class='dot' style='background:#f59e0b'></div> Mixed across runs</div>" if n_runs > 1 else ""}
      <div class="legend-item"><div class="dot" style="background:#e5e7eb"></div>N/A</div>
    </div>
  </div>

  <h2>Assertion Pass Rates</h2>
  <div class="card">
    <table>
      <thead><tr>
        {_th("Assertion")} {_th("Description")} {_th("Pass rate")}
      </tr></thead>
      <tbody>{assertion_rows_html}</tbody>
    </table>
  </div>

  <h2>Task Pass Rates</h2>
  <div class="card">
    <table>
      <thead><tr>
        {_th("Task")} {_th("Stages")} {_th("Pass rate")}
      </tr></thead>
      <tbody>{task_rows_html}</tbody>
    </table>
  </div>
</body>
</html>"""

    REPORT_PATH.write_text(html)
    print(f"Report → {REPORT_PATH}")


if __name__ == "__main__":
    main()
