"""
Run binary assertions against traces in traces.jsonl.
Appends results to results.jsonl and prints a pass/fail table.

Since traces.jsonl accumulates raw LLM responses across all runs, you can
re-score historical runs after updating assertion definitions without
re-calling the LLM.

Usage:
    .venv/bin/python evals/intelligence_agent/run_assertions.py                                          # latest run → results.jsonl
    .venv/bin/python evals/intelligence_agent/run_assertions.py --run-id 20260610_143022                 # one run → results.jsonl
    .venv/bin/python evals/intelligence_agent/run_assertions.py --run-id all --output results_v2.jsonl   # all runs → named file (required)

## How binary assertions work (Shankar et al. methodology)

Each assertion is an atomic binary question: True (pass), False (fail), or None (N/A).
The criterion is always binary — the *implementation* of checking it can vary:

  Keywords (current) — fast, transparent, brittle. A response passes "uses_shutter_context"
  if the word "shutter" appears anywhere in the text. Easy to inspect, easy to fool.

  Embeddings (next step) — cosine_similarity(response, concept) > threshold → True/False.
  More semantic, less brittle to synonyms, but threshold choice is arbitrary and failures
  are harder to explain.

  LLM-as-judge (most powerful) — ask a judge model "yes or no: did this response
  appropriately engage with the shutter context?" Still binary, fully within the
  methodology, but adds cost and the judge model's own biases.

The right time to move up the chain is when keyword failures become ambiguous — when
you can't tell from looking at the response whether it's a real failure or an assertion
gap. A continuous similarity score would be anti-binary; a thresholded one is not.

## TODOs

# TODO: Review all keyword lists against a sample of real traces to check for:
#   - False positives: keyword appears but response isn't actually doing the right thing
#     (e.g. "check" appears in "double-check your settings" vs a real action suggestion)
#   - False negatives: correct response uses synonyms not in the keyword list
#     (this has already happened once with "request" and "pause" in has_action)
#
# TODO: Consider LLM-as-judge for the "uses_*_context" assertions — keyword matching
#   can't distinguish "the shutter status is irrelevant" (mentions shutter, wrong conclusion)
#   from a response that correctly ignores a misleading shutter event. A judge model
#   asking "did the agent use this context appropriately?" would catch that distinction.
#
# TODO: Add assertions that test the misleading event_context tuples explicitly —
#   currently we assert the agent *uses* relevant context, but we don't assert it
#   *ignores* misleading context (e.g. SampleX move before intensity_drop).
#
# TODO: suggests_checking_autofocus / suggests_checking_zp_calibration (tuples id=25, id=26)
#   are expected to have low pass rates today — IntelligenceModule.on_scan_start()
#   (pystxmcontrol/controller/intelligence.py) doesn't forward scan['autofocus'] into the
#   scan_start event, so recent_events has no signal to distinguish "autofocus was off"
#   from "A0/A1 miscalibrated" from a plain energy change (id=15). That's intentional —
#   these two assertions establish an honest current-state baseline, not a target to
#   game with keyword tuning. Once autofocus is forwarded (and a way to signal A0/A1
#   drift exists), split these into properly disambiguated tuples with distinct inputs.
"""

import argparse
import json
from pathlib import Path

TRACES_PATH = Path(__file__).parent / "traces.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.jsonl"


def _contains(text: str, *keywords: str) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


# Each assertion has:
#   desc    — human-readable description for the report
#   applies — returns True if this assertion is relevant for this trace (else N/A)
#   check   — returns True (pass) or False (fail); only called when applies() is True
ASSERTIONS: dict[str, dict] = {
    "no_error": {
        "desc": "Response is not an API error",
        "applies": lambda tr: True,
        "check": lambda tr: not tr["response"].startswith("[Agent unavailable"),
    },
    "on_topic": {
        "desc": "Response addresses the anomaly domain",
        "applies": lambda tr: True,
        "check": lambda tr: (
            _contains(tr["response"], "intensity", "signal", "counts", "beam", "flux")
            if tr["tuple"]["anomaly_type"] == "intensity_drop"
            else _contains(tr["response"], "focus", "zone plate", "resolution", "sharp", "blur")
        ),
    },
    "has_action": {
        "desc": "Response includes at least one suggested action",
        "applies": lambda tr: True,
        "check": lambda tr: _contains(
            tr["response"],
            "check", "verify", "inspect", "adjust", "move", "open", "abort",
            "investigate", "consider", "try", "ensure", "confirm", "request", "pause",
        ),
    },
    "critical_urgent": {
        "desc": "Critical anomalies include urgency language",
        "applies": lambda tr: tr["tuple"]["severity"] == "critical",
        "check": lambda tr: _contains(
            tr["response"], "immediately", "urgent", "critical", "abort", "stop", "halt",
        ),
    },
    "uses_shutter_context": {
        "desc": "Mentions shutter when shutter_changed is in context",
        "applies": lambda tr: any(e.get("type") == "shutter_changed" for e in tr["recent_events"]),
        "check": lambda tr: _contains(tr["response"], "shutter", "beam block"),
    },
    "uses_zone_plate_context": {
        "desc": "Mentions zone plate when ZonePlateZ moved",
        "applies": lambda tr: any(e.get("motor") == "ZonePlateZ" for e in tr["recent_events"]),
        "check": lambda tr: _contains(tr["response"], "zone plate", "zoneplatz", "zp", "focus motor"),
    },
    "uses_energy_context": {
        "desc": "Mentions energy/focal length when Energy moved during focus_decline",
        "applies": lambda tr: (
            tr["tuple"]["anomaly_type"] == "focus_decline"
            and any(e.get("motor") == "Energy" for e in tr["recent_events"])
        ),
        "check": lambda tr: _contains(tr["response"], "energy", "focal length", "wavelength", "zone plate"),
    },
    "recognizes_energy_optics_coupling": {
        "desc": (
            "On intensity_drop with a recent Energy move, recognizes the need to "
            "re-check/re-align dependent optics (mirrors, aperture/OSA, slit, grating, "
            "harmonic) rather than attributing the drop elsewhere"
        ),
        "applies": lambda tr: (
            tr["tuple"]["anomaly_type"] == "intensity_drop"
            and any(e.get("motor") == "Energy" for e in tr["recent_events"])
        ),
        "check": lambda tr: _contains(
            tr["response"],
            "mirror", "aperture", "osa", "slit", "grating", "harmonic",
            "realign", "re-align", "reposition",
        ),
    },
    "flags_osa_collision_risk": {
        "desc": (
            "On intensity_drop with a large SampleZ move, flags possible OSA "
            "contact and recommends moving the sample away and running an OSA "
            "(focus) scan to verify"
        ),
        "applies": lambda tr: (
            tr["tuple"]["anomaly_type"] == "intensity_drop"
            and any(e.get("motor") == "SampleZ" for e in tr["recent_events"])
        ),
        "check": lambda tr: (
            _contains(tr["response"], "osa")
            and _contains(tr["response"], "move", "back", "away", "retract", "scan", "verify", "confirm")
        ),
    },
    "suggests_checking_autofocus": {
        "desc": (
            "On focus_decline with a recent Energy move, suggests checking whether "
            "autofocus/energy-tracking was enabled for the scan (honest baseline — "
            "see TODO above, agent has no direct telemetry for this yet)"
        ),
        "applies": lambda tr: (
            tr["tuple"]["anomaly_type"] == "focus_decline"
            and any(e.get("motor") == "Energy" for e in tr["recent_events"])
        ),
        "check": lambda tr: _contains(tr["response"], "autofocus", "auto-focus", "auto focus"),
    },
    "suggests_checking_zp_calibration": {
        "desc": (
            "On focus_decline with a recent Energy move, suggests checking/recalibrating "
            "the zone-plate-position-vs-energy equation (A0/A1) (honest baseline — see "
            "TODO above, agent has no direct telemetry for this yet)"
        ),
        "applies": lambda tr: (
            tr["tuple"]["anomaly_type"] == "focus_decline"
            and any(e.get("motor") == "Energy" for e in tr["recent_events"])
        ),
        "check": lambda tr: _contains(
            tr["response"], "a0", "a1", "calibrat", "equation",
        ),
    },
    "flags_diode_out_of_position": {
        "desc": (
            "On intensity_drop with the diode (Detector Y) parked out of its "
            "home position, recognizes the diode isn't in frame and recommends "
            "repositioning it"
        ),
        "applies": lambda tr: (
            tr["tuple"]["anomaly_type"] == "intensity_drop"
            and any(e.get("motor") == "Detector Y" for e in tr["recent_events"])
        ),
        "check": lambda tr: (
            _contains(tr["response"], "diode", "detector")
            and _contains(tr["response"], "frame", "home", "position", "align", "move", "reposition", "0,0", "0, 0")
        ),
    },
    "handles_ambiguous_total_loss": {
        # Empty-context intensity_drop covers two motivating scenarios that are
        # indistinguishable to the agent today: a beam dump (serious, no ring-current/
        # beam-status telemetry reaches the agent) and the sample holder frame entering
        # the field of view (benign — SampleX/SampleY moves are deliberately excluded
        # from the event stream, see controller.py _INTEL_MOTORS). Until beam status
        # becomes queryable (e.g. an EPICS PV, post-migration) the agent can't tell
        # these apart, so we only require it to hedge between both hypotheses rather
        # than commit to one. TODO: once a beam-status PV exists, split this back into
        # two tuples (one still "empty", one with the PV read in recent_events) and
        # assert the specific correct action for each.
        "desc": (
            "On unexplained critical intensity_drop, hedges between a serious cause "
            "(beam dump) and a benign one (sample holder frame in FOV) instead of "
            "committing to a single diagnosis"
        ),
        "applies": lambda tr: (
            tr["tuple"]["anomaly_type"] == "intensity_drop"
            and tr["tuple"]["event_context"] == "empty"
        ),
        "check": lambda tr: (
            _contains(tr["response"], "beam", "source", "ring", "status")
            and _contains(
                tr["response"],
                "sample", "holder", "frame", "benign", "expected",
                "could", "possible", "possibly", "either", "may be",
            )
        ),
    },
}


def evaluate(trace: dict) -> dict[str, bool | None]:
    """Return True (pass), False (fail), or None (N/A) per assertion.

    If no_error fails (API error), all content assertions are N/A — there is
    no response to evaluate.
    """
    api_ok = bool(ASSERTIONS["no_error"]["check"](trace))
    results = {}
    for name, a in ASSERTIONS.items():
        if name == "no_error":
            results[name] = api_ok
        elif not api_ok:
            results[name] = None
        elif not a["applies"](trace):
            results[name] = None
        else:
            results[name] = bool(a["check"](trace))
    return results


def _score_run(run_id: str, run_traces: list, output_path: Path) -> None:
    """Score one run's traces, print a table, and append results to output_path."""
    names = list(ASSERTIONS.keys())
    col_w = 24

    header = (f"{'ID':>3}  {'anomaly_type':15} {'sev':8} {'context':20}  "
              + "  ".join(f"{n[:col_w]:<{col_w}}" for n in names))
    print(f"Run: {run_id}  ({len(run_traces)} traces)\n")
    print(header)
    print("-" * len(header))

    totals = {n: {"pass": 0, "fail": 0, "na": 0} for n in names}

    with open(output_path, "a") as out:
        for tr in run_traces:
            results = evaluate(tr)

            out.write(json.dumps({
                "run_id": run_id,
                "id": tr["id"],
                "tuple": tr["tuple"],
                "model": tr["model"],
                "timestamp": tr["timestamp"],
                "response_chars": len(tr["response"]),
                "assertions": results,
            }) + "\n")

            cells = []
            for n in names:
                v = results[n]
                if v is None:
                    cells.append(f"{'—':<{col_w}}")
                    totals[n]["na"] += 1
                elif v:
                    cells.append(f"{'PASS':<{col_w}}")
                    totals[n]["pass"] += 1
                else:
                    cells.append(f"{'FAIL':<{col_w}}")
                    totals[n]["fail"] += 1

            print(f"{tr['id']:>3}  {tr['tuple']['anomaly_type']:15} "
                  f"{tr['tuple']['severity']:8} {tr['tuple']['event_context']:20}  "
                  + "  ".join(cells))

    print("-" * len(header))
    print()
    print("Assertions:")
    for name, a in ASSERTIONS.items():
        t = totals[name]
        applicable = t["pass"] + t["fail"]
        rate = t["pass"] / applicable * 100 if applicable else 0
        na_note = f"  ({t['na']} N/A)" if t["na"] else ""
        print(f"  {name}: {a['desc']} — {t['pass']}/{applicable} ({rate:.0f}%){na_note}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default="latest",
        help="Run ID to score, 'latest' (default), or 'all' to re-score every run in traces.jsonl.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output file for results (default: results.jsonl). "
            "Required when --run-id all is used — keeps experimental assertion snapshots "
            "separate from the canonical results.jsonl."
        ),
    )
    args = parser.parse_args()

    if args.run_id == "all" and args.output is None:
        parser.error("--output is required when --run-id all is used. "
                     "Choose a name that reflects the assertions being tested, "
                     "e.g. --output results_keyword_v2.jsonl")

    output_path = Path(args.output) if args.output else RESULTS_PATH

    traces = []
    with open(TRACES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    if not traces:
        print("No traces found in traces.jsonl — run run_eval.py first.")
        return

    all_run_ids = sorted({t["run_id"] for t in traces})

    if args.run_id == "latest":
        selected = [all_run_ids[-1]]
    elif args.run_id == "all":
        selected = all_run_ids
    else:
        if args.run_id not in all_run_ids:
            print(f"Run ID '{args.run_id}' not found. Available runs:")
            for r in all_run_ids:
                print(f"  {r}")
            return
        selected = [args.run_id]

    for run_id in selected:
        run_traces = [t for t in traces if t["run_id"] == run_id]
        _score_run(run_id, run_traces, output_path)
        if run_id != selected[-1]:
            print()

    print(f"\nResults   → {output_path}")
    print(f"Next: .venv/bin/python evals/intelligence_agent/report.py")


if __name__ == "__main__":
    main()
