"""
Drive run_eval.py's machinery across multiple models in one pass.

Each model gets its own in-memory copy of main_config with task_agent.model overridden --
main.json on disk is never touched. One broken or rate-limited model doesn't abort the batch:
failures are collected and reported at the end, and traces.jsonl is scanned afterward for the
"LLM call failed:" sentinel (see run_assertions.py's well_formed check) so a model that fails
silently inside agent.run() -- which swallows its own LLM-call exceptions and returns them as
ordinary final_text rather than raising -- still gets flagged instead of looking like a clean
100% run.

Usage:
    python evals/task_agent/run_eval_multi.py                                   # all MODELS below
    python evals/task_agent/run_eval_multi.py --models gemini-pro,anthropic/claude-sonnet
    python evals/task_agent/run_eval_multi.py --skip-tested                     # skip models with existing traces

    # Targeted retry: re-run only specific tasks for specific models (e.g. after purging
    # contaminated rows for a rate-limit/network failure on just those tasks). The run_id gets
    # a "_retry" suffix so downstream run-grouping can tell it apart from a full fresh run.
    python evals/task_agent/run_eval_multi.py --models google/grok-4.3 --tasks hw_daq_fault,fe_spectrum

After running, rescore (results.jsonl is APPEND-mode -- truncate it first):
    > evals/task_agent/results.jsonl
    python evals/task_agent/run_assertions.py --run-id all --output results.jsonl
"""
import argparse
import copy
import json
import time
from pathlib import Path

from run_eval import (
    CFGDIR, TASKS_PATH, TRACES_PATH, RUNS_META_PATH,
    _make_agent, _TokenMeter, _install_iter_tracker, _run_turn, _MockServer,
    _fetch_model_rates, _cost,
)
import plan_probe

MODELS = {
    # -- general-used (mainstream default tier) --
    "gemini-pro": "general-used",
    "amazon/gpt-5.5-medium": "general-used",
    "google/grok-4.3": "general-used",
    "anthropic/claude-sonnet": "general-used",
    "openai/gpt-5.5-medium": "general-used",
    # -- highly-reasoning --
    "anthropic/claude-opus": "highly-reasoning",
    "gemini-3.1-pro-high": "highly-reasoning",
    "xai/grok-4.20-reasoning": "highly-reasoning",
    "google/deepseek-r1": "highly-reasoning",
    "google/glm-5": "highly-reasoning",
    # -- open-source / cheap --
    "nemotron-nano-3": "open-source-cheap",
    "amazon/gpt-oss-20b": "open-source-cheap",
    "google/gemma-4": "open-source-cheap",
    "amazon/llama-4-scout": "open-source-cheap",
    "devstral-2": "open-source-cheap",
    "google/qwen-3": "open-source-cheap",
}


def _already_tested(model: str) -> bool:
    if not TRACES_PATH.exists():
        return False
    with open(TRACES_PATH) as f:
        return any(json.loads(l).get("model") == model for l in f if l.strip())


def _build_stages(task: dict) -> list:
    if task.get("kind") == "tool_unit":
        return [{"stage": "tool_unit_call", "kind": "tool_unit", "user": task["user"],
                 "expected": task.get("expected", {})}]
    return [{"stage": "plan_proposal", "kind": "plan", "user": task["intent"],
             "expected": task.get("plan_expected", {})}] + \
           [{**s, "kind": "action"} for s in task["stages"]]


def run_one_model(model: str, base_config: dict, tasks: list, retry: bool = False) -> dict:
    """Run the given task list against one model. Never raises -- catches everything per-task
    so one bad task/model doesn't abort the batch. `tasks` may be the full suite or a filtered
    subset (see --tasks) for a targeted retry; `retry=True` tags the run_id accordingly so
    downstream run-grouping can tell a partial retry apart from a full fresh run."""
    main_config = copy.deepcopy(base_config)
    main_config["task_agent"]["model"] = model
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + model.replace("/", "_") + ("_retry" if retry else "")

    n_ok = n_err = 0
    total_in = total_out = 0
    setup_errors = []
    with open(TRACES_PATH, "a") as out:
        for task in tasks:
            try:
                agent = _make_agent(main_config)
                server = _MockServer(task.get("particle", {"x": 0.0, "y": 0.0}), task.get("mock_overrides"))
                agent._toolset.dispatch = server.dispatch
                meter = _TokenMeter()
                meter.install(agent)
                _install_iter_tracker(agent, server)
                plan_probe.install(agent)

                for stage in _build_stages(task):
                    turn = _run_turn(agent, server, meter, stage["user"])
                    out.write(json.dumps({
                        "run_id": run_id, "task": task["task"], "stage": stage["stage"],
                        "kind": stage["kind"], "user": stage["user"], "expected": stage["expected"],
                        **turn, "model": agent.model, "timestamp": time.time(),
                    }) + "\n")
                n_ok += 1
                total_in += meter.input_tokens
                total_out += meter.output_tokens
            except Exception as e:
                n_err += 1
                setup_errors.append(f"{task.get('task', '?')}: {e}")
                print(f"    ! setup error on {model} / {task.get('task', '?')}: {e}")

    # Scan what was JUST written for this run_id for two distinct failure signatures:
    # 1. the "LLM call failed:" sentinel -- agent.run() swallows its own LLM-call exceptions
    #    (rate limits, incompatibility errors) and returns them as ordinary final_text, so a
    #    fully-broken model still writes "successful" rows above.
    # 2. trace["error"] populated -- an exception that escaped agent.run() entirely and was
    #    caught by _run_turn's own try/except instead (e.g. a real TaskAgent bug triggered by
    #    an unusual response shape from a specific model, as seen with deepseek-r1's inline
    #    <think> blocks). run_assertions.py's well_formed check already covers both shapes;
    #    this scan exists purely so a broken model doesn't look like a clean run in the
    #    batch summary before scoring even happens.
    n_contaminated = 0
    contaminated_stages = []
    with open(TRACES_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row.get("run_id") != run_id:
                continue
            if (row.get("final_text") or "").startswith("LLM call failed:") or row.get("error"):
                n_contaminated += 1
                contaminated_stages.append(f"{row['task']}/{row['stage']}")

    rates = _fetch_model_rates({"model": model, "provider": base_config["task_agent"]["provider"]})
    cost = _cost(rates, total_in, total_out)

    if n_err == 0 and n_contaminated == 0:
        status = "OK"
    elif n_ok == 0 or n_contaminated >= n_ok * 3:  # roughly "every stage failed"
        status = "ALL_FAILED"
    else:
        status = "PARTIAL"

    return {
        "model": model, "status": status, "run_id": run_id,
        "n_ok": n_ok, "n_err": n_err, "setup_errors": setup_errors,
        "n_contaminated": n_contaminated, "contaminated_stages": contaminated_stages,
        "input_tokens": total_in, "output_tokens": total_out,
        "cost_usd": cost["total_usd"] if cost else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="Comma-separated model IDs (default: all in MODELS dict)")
    ap.add_argument("--skip-tested", action="store_true", help="Skip models that already have traces recorded")
    ap.add_argument("--tasks", default=None,
                     help="Comma-separated task names to run instead of the full suite -- for a "
                          "targeted retry of specific tasks after purging contaminated rows")
    args = ap.parse_args()

    base_config = json.loads((CFGDIR / "main.json").read_text())
    all_tasks = [json.loads(l) for l in TASKS_PATH.read_text().splitlines() if l.strip()]

    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in all_tasks if t["task"] in wanted]
        missing = wanted - {t["task"] for t in tasks}
        if missing:
            raise SystemExit(f"Unknown task name(s): {sorted(missing)}")
    else:
        tasks = all_tasks

    model_list = args.models.split(",") if args.models else list(MODELS.keys())
    if args.skip_tested:
        model_list = [m for m in model_list if not _already_tested(m)]

    print(f"Running {len(tasks)} tasks against {len(model_list)} model(s):")
    for m in model_list:
        print(f"  - {m} ({MODELS.get(m, '?')})")
    print()

    summary = []
    for i, model in enumerate(model_list, 1):
        print(f"[{i}/{len(model_list)}] {model} ...")
        t0 = time.time()
        result = run_one_model(model, base_config, tasks, retry=bool(args.tasks))
        result["elapsed_s"] = round(time.time() - t0, 1)
        summary.append(result)
        cost_str = f"${result['cost_usd']:.3f}" if result["cost_usd"] is not None else "cost n/a"
        print(f"    -> {result['status']}  ok={result['n_ok']} err={result['n_err']} "
              f"contaminated={result['n_contaminated']}  {cost_str}  ({result['elapsed_s']}s)")
        if result["contaminated_stages"]:
            print(f"       contaminated: {result['contaminated_stages'][:5]}"
                  + (" ..." if len(result["contaminated_stages"]) > 5 else ""))

    print("\n=== Batch summary ===")
    total_cost = 0.0
    for r in summary:
        cost_str = f"${r['cost_usd']:.3f}" if r["cost_usd"] is not None else "n/a"
        total_cost += r["cost_usd"] or 0.0
        print(f"  {r['model']:30s} {r['status']:11s} ok={r['n_ok']:>3} err={r['n_err']:>2} "
              f"contam={r['n_contaminated']:>2}  {cost_str:>8}")
    print(f"\nTotal estimated cost: ${total_cost:.3f}")

    out_path = Path(__file__).parent / "batch_run_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved -> {out_path}")
    print("\nModels with status != OK need attention (see contaminated_stages / setup_errors)")
    print("before rescoring. To rescore everything:")
    print("  > evals/task_agent/results.jsonl   # truncate first -- run_assertions.py appends")
    print("  python evals/task_agent/run_assertions.py --run-id all --output results.jsonl")


if __name__ == "__main__":
    main()
