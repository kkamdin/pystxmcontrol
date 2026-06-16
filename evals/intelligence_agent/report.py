"""
Generate report.html from results.jsonl.

Shows a binary criteria matrix (heatmap) across all runs, with pass rates
per assertion. Re-run whenever you want a fresh view.

Usage:
    .venv/bin/python evals/intelligence_agent/report.py
"""

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results.jsonl"
RUNS_META_PATH = Path(__file__).parent / "runs_meta.jsonl"
REPORT_PATH = Path(__file__).parent / "report.html"

ASSERTION_DESCRIPTIONS = {
    "no_error":               "Response is not an API error",
    "on_topic":               "Addresses the anomaly domain",
    "has_action":             "Includes a suggested action",
    "critical_urgent":        "Critical severity → urgency language",
    "uses_shutter_context":   "Uses shutter event context",
    "uses_zone_plate_context":"Uses ZonePlateZ motor context",
    "uses_energy_context":    "Uses Energy motor context (focus)",
}


def _color(pass_rate: float | None) -> str:
    """Return a CSS background color for a pass rate (0–1), or gray for N/A."""
    if pass_rate is None:
        return "#e5e7eb"  # gray-200 — N/A
    if pass_rate == 1.0:
        return "#22c55e"  # green-500
    if pass_rate == 0.0:
        return "#ef4444"  # red-500
    # Interpolate red → amber → green
    if pass_rate < 0.5:
        g = int(pass_rate * 2 * 251)
        return f"rgb(239,{g},68)"   # red → amber
    else:
        r = int((1 - pass_rate) * 2 * 239)
        return f"rgb({r},197,68)"   # amber → green


def _text_color(pass_rate: float | None) -> str:
    if pass_rate is None:
        return "#6b7280"
    return "#fff" if pass_rate in (0.0, 1.0) else "#1f2937"


def _bar(rate: float, width_px: int = 200) -> str:
    filled = int(rate * width_px)
    pct = f"{rate * 100:.0f}%"
    return (
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<div style="width:{width_px}px;height:16px;background:#e5e7eb;border-radius:4px;overflow:hidden">'
        f'<div style="width:{filled}px;height:100%;background:#22c55e"></div></div>'
        f'<span style="font-size:13px;color:#374151">{pct}</span>'
        f'</div>'
    )


def main() -> None:
    rows = []
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("No results found — run run_assertions.py first.")
        return

    # Group by run_id
    runs: dict[str, list] = defaultdict(list)
    for r in rows:
        runs[r["run_id"]].append(r)
    run_ids = sorted(runs.keys())

    # Assertion names (preserve order from first result)
    assertion_names = list(rows[0]["assertions"].keys())

    # Per-tuple aggregated results: {tuple_id: {assertion: [True/False/None, ...]}}
    tuple_order = []
    seen_ids = set()
    for r in rows:
        if r["id"] not in seen_ids:
            tuple_order.append(r)
            seen_ids.add(r["id"])
    tuple_order.sort(key=lambda r: r["id"])

    # Build cell data: {(tuple_id, assertion): (n_pass, n_applicable) or None}
    cell_data: dict[tuple, tuple[int, int] | None] = {}
    for tup in tuple_order:
        tid = tup["id"]
        for aname in assertion_names:
            vals = [r["assertions"].get(aname) for r in rows if r["id"] == tid]
            applicable = [v for v in vals if v is not None]
            if not applicable:
                cell_data[(tid, aname)] = None
            else:
                cell_data[(tid, aname)] = (sum(applicable), len(applicable))

    # Per-assertion overall pass rate (weighted by applicable count per cell)
    assertion_rates: dict[str, float | None] = {}
    for aname in assertion_names:
        cells = [v for (tid, an), v in cell_data.items() if an == aname and v is not None]
        if not cells:
            assertion_rates[aname] = None
        else:
            total_pass = sum(n_pass for n_pass, _ in cells)
            total_app = sum(n_app for _, n_app in cells)
            assertion_rates[aname] = total_pass / total_app if total_app else None

    # Load runs_meta keyed by run_id for token/cost lookup.
    runs_meta: dict[str, dict] = {}
    if RUNS_META_PATH.exists():
        with open(RUNS_META_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    m = json.loads(line)
                    runs_meta[m["run_id"]] = m

    # Per-run summary
    run_summaries = []
    for rid in run_ids:
        run_rows = runs[rid]
        ts = datetime.fromtimestamp(run_rows[0]["timestamp"]).strftime("%Y-%m-%d %H:%M")
        model = run_rows[0]["model"]
        all_vals = [v for r in run_rows for v in r["assertions"].values() if v is not None]
        overall = sum(all_vals) / len(all_vals) if all_vals else 0
        meta = runs_meta.get(rid, {})
        run_summaries.append({"run_id": rid, "ts": ts, "model": model,
                               "n": len(run_rows), "overall": overall, "meta": meta})

    n_runs = len(run_ids)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- HTML ----
    def th(text: str, title: str = "", rotate: bool = False) -> str:
        style = (
            'style="writing-mode:vertical-rl;transform:rotate(180deg);'
            'white-space:nowrap;padding:8px 4px;font-size:12px;font-weight:600;'
            'color:#374151;text-align:left"'
            if rotate else
            'style="padding:8px 12px;font-size:12px;font-weight:600;'
            'color:#6b7280;text-align:left;white-space:nowrap"'
        )
        title_attr = f' title="{title}"' if title else ""
        return f"<th {style}{title_attr}>{text}</th>"

    def td_label(text: str) -> str:
        return (f'<td style="padding:6px 12px;font-size:12px;color:#374151;'
                f'white-space:nowrap;border-right:1px solid #e5e7eb">{text}</td>')

    def td_cell(counts: tuple[int, int] | None) -> str:
        rate = counts[0] / counts[1] if counts is not None else None
        bg = _color(rate)
        fg = _text_color(rate)
        if counts is None:
            label = "—"
        else:
            n_pass, n_app = counts
            label = ("✓" if n_pass == n_app else "✗") if n_app == 1 else f"{n_pass}/{n_app}"
        return (f'<td style="text-align:center;padding:4px;background:{bg};'
                f'color:{fg};font-size:13px;font-weight:600;'
                f'min-width:44px;border:1px solid #fff">{label}</td>')

    # Run summary rows
    run_rows_html = ""
    for rs in run_summaries:
        bar = _bar(rs["overall"], 120)
        meta = rs["meta"]
        tok_in = meta.get("total_input_tokens")
        tok_out = meta.get("total_output_tokens")
        ctx_win = meta.get("context_window")
        cost_data = meta.get("cost_estimate") or {}
        im = meta.get("inputs_meta", {})
        acfg = im.get("anomaly_config", {})
        inputs_built_at = im.get("built_at", "—")
        threshold_summary = (
            "  ".join(f"{k}={v}" for k, v in acfg.items())
            if acfg else "—"
        )

        if tok_in is not None:
            total_tok = tok_in + tok_out
            avg_in = tok_in // max(rs["n"], 1)
            avg_out = tok_out // max(rs["n"], 1)
            tok_cell = (f"{total_tok:,} total"
                        f"<br><small style='color:#6b7280'>"
                        f"{avg_in:,} in / {avg_out:,} out avg/req</small>")
            if ctx_win:
                fill = round(tok_in / ctx_win / max(rs["n"], 1) * 100, 1)
                tok_cell += f"<br><small style='color:#6b7280'>{fill}% ctx fill (avg)</small>"
        else:
            tok_cell = "—"

        total_cost = cost_data.get("total_cost_usd")
        if total_cost is not None:
            in_c = cost_data.get("input_cost_usd", 0)
            out_c = cost_data.get("output_cost_usd", 0)
            in_rate = cost_data.get("input_cost_per_token", 0)
            out_rate = cost_data.get("output_cost_per_token", 0)
            cost_cell = (
                f"${total_cost:.4f}"
                f"<br><small style='color:#6b7280'>"
                f"in ${in_c:.4f} / out ${out_c:.4f}</small>"
                f"<br><small style='color:#6b7280'>"
                f"@ ${in_rate*1e6:.2f}/${out_rate*1e6:.2f} per MTok</small>"
            )
        else:
            cost_cell = "—"

        params_cell = (
            f'<span style="font-family:monospace;font-size:11px;color:#374151">'
            f'{threshold_summary}</span>'
            f'<br><small style="color:#6b7280">inputs built {inputs_built_at}</small>'
            if acfg else "—"
        )

        run_rows_html += (
            f'<tr>'
            f'<td style="font-family:monospace;font-size:12px;padding:6px 12px">{rs["run_id"]}</td>'
            f'<td style="font-size:12px;padding:6px 12px;color:#6b7280">{rs["ts"]}</td>'
            f'<td style="font-size:12px;padding:6px 12px">{rs["model"]}</td>'
            f'<td style="font-size:12px;padding:6px 12px;text-align:center">{rs["n"]}</td>'
            f'<td style="font-size:12px;padding:6px 12px">{params_cell}</td>'
            f'<td style="font-size:12px;padding:6px 12px">{tok_cell}</td>'
            f'<td style="font-size:12px;padding:6px 12px">{cost_cell}</td>'
            f'<td style="padding:6px 12px">{bar}</td>'
            f'</tr>'
        )

    # Heatmap rows
    heatmap_rows_html = ""
    for tup in tuple_order:
        tid = tup["id"]
        t = tup["tuple"]
        label = f"{t['anomaly_type']} / {t['severity']} / {t['event_context']}"
        cells = "".join(td_cell(cell_data[(tid, an)]) for an in assertion_names)
        heatmap_rows_html += f"<tr>{td_label(label)}{cells}</tr>\n"

    # Assertion pass rate rows
    assertion_rows_html = ""
    for aname in assertion_names:
        rate = assertion_rates[aname]
        desc = ASSERTION_DESCRIPTIONS.get(aname, aname)
        bar = _bar(rate if rate is not None else 0)
        pct = f"{rate*100:.0f}%" if rate is not None else "N/A"
        assertion_rows_html += (
            f'<tr>'
            f'<td style="font-size:12px;font-weight:600;padding:6px 12px;'
            f'font-family:monospace;white-space:nowrap">{aname}</td>'
            f'<td style="font-size:12px;padding:6px 12px;color:#6b7280">{desc}</td>'
            f'<td style="padding:6px 12px">{bar}</td>'
            f'</tr>'
        )

    # Column headers (rotated)
    col_headers = "".join(
        th(an, ASSERTION_DESCRIPTIONS.get(an, an), rotate=True)
        for an in assertion_names
    )


    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Intelligence Agent Eval Report</title>
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
    .legend-dot {{ width: 12px; height: 12px; border-radius: 2px; }}
  </style>
</head>
<body>
  <h1>Intelligence Agent Evals</h1>
  <p class="subtitle">
    Assertion-based binary criteria matrix &nbsp;·&nbsp;
    ALS 7.0.1.2 STXM &nbsp;·&nbsp;
    {n_runs} run{"s" if n_runs != 1 else ""} &nbsp;·&nbsp;
    Generated {generated}
  </p>

  <h2>Runs</h2>
  <div class="card">
    <table>
      <thead><tr>
        {th("Run ID")} {th("Date")} {th("Model")} {th("Traces")} {th("Parameters")} {th("Tokens")} {th("Cost est.")} {th("Overall pass rate")}
      </tr></thead>
      <tbody>{run_rows_html}</tbody>
    </table>
  </div>

  <h2>Binary Criteria Matrix</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          {th("Test case")}
          {col_headers}
        </tr>
      </thead>
      <tbody>
        {heatmap_rows_html}
      </tbody>
    </table>
    <div class="legend">
      <div class="legend-item">
        <div class="legend-dot" style="background:#22c55e"></div> Pass{"  (all runs)" if n_runs > 1 else ""}
      </div>
      <div class="legend-item">
        <div class="legend-dot" style="background:#ef4444"></div> Fail{"  (all runs)" if n_runs > 1 else ""}
      </div>
      {"<div class='legend-item'><div class='legend-dot' style='background:#f59e0b'></div> Mixed across runs</div>" if n_runs > 1 else ""}
      <div class="legend-item">
        <div class="legend-dot" style="background:#e5e7eb"></div> N/A (assertion not applicable)
      </div>
    </div>
  </div>

  <h2>Assertion Pass Rates</h2>
  <div class="card">
    <table>
      <thead><tr>
        {th("Assertion")} {th("Description")} {th("Pass rate")}
      </tr></thead>
      <tbody>{assertion_rows_html}</tbody>
    </table>
  </div>
</body>
</html>"""

    REPORT_PATH.write_text(html)
    print(f"Report → {REPORT_PATH}")


if __name__ == "__main__":
    main()
