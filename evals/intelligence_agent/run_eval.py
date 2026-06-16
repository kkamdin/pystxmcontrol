"""
Step 6: Run each synthetic input through AgentInterface and collect traces.

Each run gets a unique run_id (timestamp slug). Traces are appended to
traces.jsonl so all runs accumulate in one file.

Also calls _log_trace() so each run appears in the production system log
(agent_traces_intelligence.jsonl) alongside real session traces.

Token counts and estimated cost are written to runs_meta.jsonl after each run.
Use the /cost/estimate endpoint — see report.py for display.

Requires ANTHROPIC_API_KEY to be set and intelligence.agent.enabled = true in config.

Usage:
    .venv/bin/python evals/intelligence_agent/run_eval.py
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pystxmcontrol.controller.intelligence import AgentInterface, _SYSTEM_PROMPT

CONFIG_PATH = REPO_ROOT / ".venv/pystxmcontrol_cfg/main.json"
INPUTS_PATH = Path(__file__).parent / "inputs.jsonl"
INPUTS_META_PATH = Path(__file__).parent / "inputs_meta.json"
TRACES_PATH = Path(__file__).parent / "traces.jsonl"
RUNS_META_PATH = Path(__file__).parent / "runs_meta.jsonl"


def _compute_cost(agent: AgentInterface, total_input: int, total_output: int) -> dict | None:
    """Compute run cost from per-token rates fetched from /model_group/info."""
    if agent.input_cost_per_token is None or agent.output_cost_per_token is None:
        return None
    input_cost = total_input * agent.input_cost_per_token
    output_cost = total_output * agent.output_cost_per_token
    return {
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
        "input_cost_per_token": agent.input_cost_per_token,
        "output_cost_per_token": agent.output_cost_per_token,
    }


def main() -> None:
    run_id = time.strftime("%Y%m%d_%H%M%S")

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    inputs_meta = {}
    if INPUTS_META_PATH.exists():
        inputs_meta = json.loads(INPUTS_META_PATH.read_text())
    else:
        print("Warning: inputs_meta.json not found — run build_inputs.py first.")

    agent = AgentInterface(config)

    inputs = []
    with open(INPUTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                inputs.append(json.loads(line))

    print(f"Run: {run_id}  ({len(inputs)} inputs)")

    total_input_tokens = 0
    total_output_tokens = 0

    with open(TRACES_PATH, "a") as out:
        for inp in inputs:
            t = inp["tuple"]
            prompt = agent._format_prompt(inp["anomaly"], inp["recent_events"])
            error = None
            usage = {"input_tokens": 0, "output_tokens": 0}
            try:
                # TODO: calls _call_api() directly, bypassing dispatch() and its
                # prompt-formatting logic. If dispatch() is updated but _format_prompt()
                # is not, evals won't catch it. Consider an integration path through
                # dispatch() for full coverage.
                response, usage = agent._call_api(prompt)
            except Exception as exc:
                response = f"[Agent unavailable: {exc}]"
                error = str(exc)

            total_input_tokens += usage["input_tokens"]
            total_output_tokens += usage["output_tokens"]

            # Write to the production system log (no tuple context).
            agent._log_trace({
                "call_type": "dispatch",
                "timestamp": time.time(),
                "model": agent.model,
                "system_prompt": _SYSTEM_PROMPT,
                "prompt": prompt,
                "response": response,
                "anomaly_type": inp["anomaly"].get("type"),
                "severity": inp["anomaly"].get("severity"),
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "context_window": agent.context_window,
                "context_fill_pct": (
                    round(usage["input_tokens"] / agent.context_window * 100, 2)
                    if agent.context_window else None
                ),
                "error": error,
            })

            # Append to eval-specific traces (full context for assertions).
            trace = {
                "run_id": run_id,
                "id": inp["id"],
                "tuple": t,
                "anomaly": inp["anomaly"],
                "recent_events": inp["recent_events"],
                "system_prompt": _SYSTEM_PROMPT,
                "prompt": prompt,
                "response": response,
                "model": agent.model,
                "timestamp": time.time(),
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "error": error,
            }
            out.write(json.dumps(trace) + "\n")

            status = "OK" if not error else "ERR"
            print(f"  [{inp['id']:2d}] {status} {t['anomaly_type']:15s} {t['severity']:8s} "
                  f"{t['event_context']:20s} → {len(response)} chars  "
                  f"in={usage['input_tokens']} out={usage['output_tokens']}")

    cost_data = _compute_cost(agent, total_input_tokens, total_output_tokens)

    meta = {
        "run_id": run_id,
        "timestamp": time.time(),
        "model": agent.model,
        "provider": agent.provider,
        "n_inputs": len(inputs),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "context_window": agent.context_window,
        "cost_estimate": cost_data,
        "inputs_meta": inputs_meta,
    }
    with open(RUNS_META_PATH, "a") as f:
        f.write(json.dumps(meta) + "\n")

    # Print token summary.
    print(f"\nTokens    → in={total_input_tokens:,}  out={total_output_tokens:,}  "
          f"total={total_input_tokens + total_output_tokens:,}")
    if cost_data:
        total = cost_data["total_cost_usd"]
        in_c = cost_data["input_cost_usd"]
        out_c = cost_data["output_cost_usd"]
        rates = (f"@ ${cost_data['input_cost_per_token']*1e6:.2f}/"
                 f"${cost_data['output_cost_per_token']*1e6:.2f} per MTok in/out")
        print(f"Cost      → ${total:.4f} total  (in ${in_c:.4f}  out ${out_c:.4f})  {rates}")

    print(f"\nTraces    → {TRACES_PATH}")
    print(f"Runs meta → {RUNS_META_PATH}")
    print(f"System log → {config['intelligence']['agent']['trace_log']}")
    print(f"\nNext: .venv/bin/python evals/intelligence_agent/run_assertions.py")


if __name__ == "__main__":
    main()
