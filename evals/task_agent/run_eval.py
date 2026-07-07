"""
Step 1+2: Drive the REAL TaskAgent through each task and record what it does.

This is the production agent — its real `_SYSTEM_PROMPT`, message handling, and tool-use
loop — so the trace reflects exactly what the agent sees and decides. Only the *server* is
mocked: `ToolSet.dispatch` is replaced with a scripted in-memory responder, so no stxmserver
and no hardware are needed. We record the input (message history incl. system prompt) and the
agent's tool calls per stage; run_assertions.py checks them.

Flow per task:
  agent.run(intent)          -> session startup + the agent proposes a plan (text)
  agent.run(stage[0].user)   -> execute the plan for that stage
  agent.run(stage[1].user)   -> next step after the scan result
  ...

The mock server also tracks which LLM iteration each tool call came from (llm_iter), so
run_assertions.py can catch parallel tool-call bugs (e.g. update_scan and start_multiregion_scan
fired in the same LLM response before the agent saw the update result).

Usage:
    python evals/task_agent/run_eval.py
"""

import json
import sys
import time
from pathlib import Path

from pystxmcontrol.controller.task_agent.agent import TaskAgent

import plan_probe

CFGDIR = Path(sys.prefix) / "pystxmcontrol_cfg"
TASKS_PATH = Path(__file__).parent / "tasks.jsonl"
TRACES_PATH = Path(__file__).parent / "traces.jsonl"


def _serialize_messages(messages: list) -> list[dict]:
    """Render TaskAgent's message list as plain dicts (OpenAI message objects -> model_dump()).

    Eval-only utility -- doesn't touch TaskAgent's own internals, just reads its public-ish
    `_messages` list the same way the rest of this harness already reaches into `_llm` and
    `_toolset`. Kept here instead of on TaskAgent so pystxmcontrol/ stays untouched; the only
    consumer besides this harness is TaskAgent's own optional trace_log feature, which builds
    the identical list inline when it needs it.
    """
    return [m if isinstance(m, dict) else m.model_dump() for m in messages]
RUNS_META_PATH = Path(__file__).parent / "runs_meta.jsonl"


class _MockServer:
    """Scripted, in-memory stand-in for the controller.

    Records tool calls per turn, each annotated with `llm_iter` — the index of the LLM
    completion that requested the call. Calls from the same LLM response share the same
    llm_iter; sequential calls from different responses have different llm_iters. This lets
    run_assertions.py detect parallel-call bugs where two tools are fired in the same
    completion before the agent has seen either result.
    """

    def __init__(self, particle, overrides=None):
        self.particle = particle
        self.overrides = overrides or {}   # tool name -> canned result (adverse-condition tests)
        self.turn_calls = []   # {"name", "arguments", "llm_iter"} for the current agent.run()
        self._llm_iter = 0     # incremented after each LLM response, before its tools run

    def new_llm_iter(self):
        """Signal that a new LLM response has arrived. Call this after each completion."""
        self._llm_iter += 1

    def _report(self):
        return json.dumps({"recommendations": [{
            "type": "task_recommendation", "subtype": "two_energy_particles",
            "edge_energy_eV": 709.5, "preedge_energy_eV": 705.0, "particle_count": 1,
            "particles": [{"center_um": {"x": self.particle["x"], "y": self.particle["y"]},
                           "size_um": {"x": 0.8, "y": 0.8}, "area_px": 51}],
            "reason": f"Found 1 Fe particle at x={self.particle['x']}, y={self.particle['y']} um.",
        }]}, indent=2)

    def dispatch(self, name, args):
        self.turn_calls.append({"name": name, "arguments": args, "llm_iter": self._llm_iter})
        if name in self.overrides:        # adverse-condition injection (error / no particles / ...)
            return self.overrides[name]
        p = self.particle
        return {
            "get_safety_instructions": "SAFETY: confirm scans; ask before Energy moves >100 eV; ask if >10 energies.",
            "get_config": json.dumps({"motors": ["Energy", "SampleX", "SampleY", "ZonePlateZ"],
                "scan_types": ["Image", "Focus", "Line Spectrum", "Single Motor", "Double Motor"],
                "positions": {"Energy": 600.0, "SampleX": -50.0, "SampleY": -50.0}}),
            "update_scan": "Scan updated: " + json.dumps(args),
            # Eval-only probe tool (see plan_probe.py) -- not a real ToolSet method.
            "propose_plan": plan_probe.PROPOSE_PLAN_RESULT_PREFIX + json.dumps(args),
            "check_scan_limits": json.dumps({"ok": True, "message": "fits within travel"}),
            "start_scan": "Scan started.",
            "wait_for_scan": "Scan complete - instrument is now idle. Call get_last_scan_stats() to analyse.",
            "get_scan_status": "Instrument is idle.",
            "get_intelligence_recommendations": self._report(),
            "load_intelligence_particles": "Loaded 1 particle region from the two-energy report.",
            "start_multiregion_scan": "Started multi-region scan over 1 region.",
            "get_last_scan_stats": json.dumps({"mean": 988.0, "contrast": 0.07,
                "dark_region_centroid_um": {"x": p["x"], "y": p["y"]}}),
            "get_image_center_of_mass": json.dumps({"center_of_mass_um": {"x": p["x"], "y": p["y"]}}),
            # Intentionally returns Line Spectrum params when Image is requested — simulates the
            # stale-params confusion seen in production traces.
            "get_last_scan_params": json.dumps({"scan_type": "Line Spectrum", "x_range": 4.0,
                "x_points": 40, "energy_start": 700.0, "energy_stop": 720.0, "energy_points": 41}),
        }.get(name, "OK")


def _make_agent(main_config):
    class _Client:
        motorInfo = json.loads((CFGDIR / "motor.json").read_text())
        scanConfig = json.loads((CFGDIR / "scan.json").read_text())
        currentMotorPositions = {}
        main_config = None
    _Client.main_config = main_config
    return TaskAgent(main_config, _Client(), image_model=None)


class _TokenMeter:
    """Accumulates token usage across every LLM call the agent makes."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def install(self, agent):
        create = agent._llm.chat.completions.create

        def wrapped(*a, **k):
            resp = create(*a, **k)
            u = getattr(resp, "usage", None)
            if u:
                self.input_tokens += getattr(u, "prompt_tokens", 0) or 0
                self.output_tokens += getattr(u, "completion_tokens", 0) or 0
            return resp

        agent._llm.chat.completions.create = wrapped


def _install_iter_tracker(agent, server):
    """Wrap LLM completions to mark new response groups in the mock server.

    After each LLM completion returns — and before the agent dispatches its tool calls —
    server.new_llm_iter() is called. All tool calls dispatched from the same LLM response
    share the same llm_iter value, allowing run_assertions.py to detect when two tools were
    requested in parallel (same completion) rather than sequentially.
    """
    create = agent._llm.chat.completions.create

    def wrapped(*a, **k):
        resp = create(*a, **k)
        server.new_llm_iter()
        return resp

    agent._llm.chat.completions.create = wrapped


def _fetch_model_rates(task_cfg):
    """Per-token $ rates + context window from the CBORG/LiteLLM /model_group/info endpoint."""
    import os
    prov = task_cfg.get("provider", {})
    base_url = prov.get("base_url")
    key = os.environ.get(prov.get("api_key_env", "OPENAI_API_KEY"), "")
    if not (base_url and key):
        return {}
    try:
        import httpx
        r = httpx.get(f"{base_url.rstrip('/')}/model_group/info",
                      params={"model_group": task_cfg.get("model")},
                      headers={"Authorization": f"Bearer {key}"}, timeout=5.0)
        r.raise_for_status()
        data = r.json().get("data") or []
        d = data[0] if data else {}
        return {"input_cost_per_token": d.get("input_cost_per_token"),
                "output_cost_per_token": d.get("output_cost_per_token"),
                "context_window": d.get("max_input_tokens")}
    except Exception as exc:
        print(f"  (could not fetch model rates: {exc})")
        return {}


def _cost(rates, in_tok, out_tok):
    ic, oc = rates.get("input_cost_per_token"), rates.get("output_cost_per_token")
    if ic is None or oc is None:
        return None
    return {"input_usd": round(in_tok * ic, 6), "output_usd": round(out_tok * oc, 6),
            "total_usd": round(in_tok * ic + out_tok * oc, 6),
            "input_cost_per_token": ic, "output_cost_per_token": oc}


def _run_turn(agent, server, meter, user_msg):
    """Run one agent turn and capture everything needed to score it."""
    input_messages = _serialize_messages(agent._messages)
    server.turn_calls = []
    server._llm_iter = 0
    in0, out0 = meter.input_tokens, meter.output_tokens
    error, final = None, None
    try:
        final = agent.run(user_msg)
    except Exception as e:
        error = str(e)
    calls = list(server.turn_calls)
    st_in, st_out = meter.input_tokens - in0, meter.output_tokens - out0
    return {
        "input_messages": input_messages, "tool_calls": calls, "final_text": final,
        "input_tokens": st_in, "output_tokens": st_out, "error": error,
    }


def main():
    run_id = time.strftime("%Y%m%d_%H%M%S")
    main_config = json.loads((CFGDIR / "main.json").read_text())
    task_cfg = main_config.get("task_agent", {})
    tasks = [json.loads(l) for l in TASKS_PATH.read_text().splitlines() if l.strip()]
    rates = _fetch_model_rates(task_cfg)
    print(f"Run: {run_id}  ({len(tasks)} task(s), model={task_cfg.get('model')})")

    total_in = total_out = 0
    with open(TRACES_PATH, "a") as out:
        for task in tasks:
            agent = _make_agent(main_config)
            server = _MockServer(task.get("particle", {"x": 0.0, "y": 0.0}), task.get("mock_overrides"))
            agent._toolset.dispatch = server.dispatch
            meter = _TokenMeter()
            meter.install(agent)
            _install_iter_tracker(agent, server)
            plan_probe.install(agent)

            if task.get("kind") == "tool_unit":
                # Isolated single-turn tool-call test (see build_inputs.py's
                # _tool_unit_cases()): one fresh agent, one instruction, no plan/confirm
                # dance. Scored with the same param_checks machinery as plan/action stages.
                all_stages = [{
                    "stage": "tool_unit_call", "kind": "tool_unit", "user": task["user"],
                    "expected": task.get("expected", {}),
                }]
            else:
                # Turn 0 (plan_proposal) is scored like any other stage: the model must produce
                # a sound plan from the raw intent alone, before any tool call or human
                # confirmation. See build_inputs.py's _plan_expected() for what "sound" means.
                all_stages = [{
                    "stage": "plan_proposal", "kind": "plan", "user": task["intent"],
                    "expected": task.get("plan_expected", {}),
                }] + [{**s, "kind": "action"} for s in task["stages"]]

            for stage in all_stages:
                turn = _run_turn(agent, server, meter, stage["user"])
                out.write(json.dumps({
                    "run_id": run_id, "task": task["task"], "stage": stage["stage"],
                    "kind": stage["kind"], "user": stage["user"], "expected": stage["expected"],
                    **turn,
                    "model": agent.model, "timestamp": time.time(),
                }) + "\n")
                names = ", ".join(c["name"] for c in turn["tool_calls"]) or "(none)"
                print(f"  [{task['task']}/{stage['stage']}] in={turn['input_tokens']} out={turn['output_tokens']} -> {names}"
                      + (f"  ERR={turn['error']}" if turn["error"] else ""))

            total_in += meter.input_tokens
            total_out += meter.output_tokens

    cost = _cost(rates, total_in, total_out)
    meta = {"run_id": run_id, "timestamp": time.time(), "model": task_cfg.get("model"),
            "n_tasks": len(tasks), "total_input_tokens": total_in, "total_output_tokens": total_out,
            "context_window": rates.get("context_window"), "cost_estimate": cost}
    with open(RUNS_META_PATH, "a") as f:
        f.write(json.dumps(meta) + "\n")

    print(f"\nTokens    -> in={total_in:,}  out={total_out:,}  total={total_in + total_out:,}")
    if cost:
        print(f"Cost      -> ${cost['total_usd']:.4f}  (in ${cost['input_usd']:.4f}  out ${cost['output_usd']:.4f})")
    else:
        print("Cost      -> unavailable (no per-token rates from /model_group/info)")
    print(f"Traces    -> {TRACES_PATH}\nRuns meta -> {RUNS_META_PATH}\n"
          f"Next: python evals/task_agent/run_assertions.py")


if __name__ == "__main__":
    main()
