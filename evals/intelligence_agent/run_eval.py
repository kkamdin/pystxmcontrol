"""
Step 6: Run each synthetic input through AgentInterface and collect traces.

Routes each input through AgentInterface.dispatch() — the same call path
production uses — so any change to prompt formatting or dispatch logic is
caught automatically.

Each run gets a unique run_id (timestamp slug). Traces are appended to
traces.jsonl so all runs accumulate in one file.

dispatch() also writes each run to the production system log
(agent_traces_intelligence.jsonl) alongside real session traces.

Token counts and estimated cost are written to runs_meta.jsonl after each run.

Requires the appropriate API key env var to be set (ANTHROPIC_API_KEY or OPENAI_API_KEY,
depending on intelligence.agent.provider in config) and intelligence.agent.enabled = true.

Usage:
    .venv/bin/python evals/intelligence_agent/run_eval.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from pystxmcontrol.controller.intelligence import AgentInterface, _SYSTEM_PROMPT

CONFIG_PATH = Path(sys.prefix) / "pystxmcontrol_cfg/main.json"
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
    agent.cooldown_seconds = 0  # disable cooldown so all inputs run without waiting

    # Capture the formatted prompt and token usage from dispatch's internal _log_trace call.
    _trace_capture: dict = {}
    _orig_log_trace = agent._log_trace
    def _capturing_log_trace(entry: dict) -> None:
        _trace_capture.update(entry)
        _orig_log_trace(entry)
    agent._log_trace = _capturing_log_trace

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
            _trace_capture.clear()
            t = inp["tuple"]

            try:
                suggestion = asyncio.run(agent.dispatch(inp["anomaly"], inp["recent_events"]))
                response = suggestion["suggestion"] if suggestion else "[Agent unavailable: in cooldown]"
            except Exception as exc:
                response = f"[Agent unavailable: {exc}]"

            # dispatch() already wrote to the production system log via _log_trace.
            # Pull usage and error from what it logged; use 0 when the API call failed (None → int).
            usage = {
                "input_tokens": _trace_capture.get("input_tokens") or 0,
                "output_tokens": _trace_capture.get("output_tokens") or 0,
            }
            error = _trace_capture.get("error")

            total_input_tokens += usage["input_tokens"]
            total_output_tokens += usage["output_tokens"]

            # Append to eval-specific traces (full context for assertions).
            trace = {
                "run_id": run_id,
                "id": inp["id"],
                "tuple": t,
                "anomaly": inp["anomaly"],
                "recent_events": inp["recent_events"],
                "system_prompt": _SYSTEM_PROMPT,
                "prompt": _trace_capture.get("prompt", ""),
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
