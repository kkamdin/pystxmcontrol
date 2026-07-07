"""
Build tasks.jsonl from tuples.jsonl.

Each tuple defines a scenario along four dimensions:
  task_type   : elemental_map | image_scan | line_spectrum
  complication: happy_path | no_recs | scan_error | stale_params | given_particle
  element     : Fe | null
  plan_detail : specified | underspecified

`plan_detail` controls how much technical detail is in the INITIAL intent (see _intent()):
  specified     - the intent itself states the exact scan type, edge, energies, and point count
                  a scientist who already knows what they want would give.
  underspecified - the intent is a bare ask ("find the Fe particles") with no scan/energy detail.
                  The agent must supply the correct approach itself -- either by proposing it
                  directly with correct parameters, or by asking the user clarifying questions.
                  Both are acceptable; run_assertions.py's energy_in_range check is N/A (not a
                  fail) when the plan states no energies, and only checked when it does.

Once the plan is proposed, every stage's approval message is a plain "Yes, please proceed." --
the detail differentiation lives entirely in the intent, not in what the user says after.

This script expands each tuple into a concrete task definition:
  intent, particle coords, mock server overrides, and per-stage expected assertions.

No forbidden_tools are used. Assertions cover only positive behaviors:
what the agent must call and with what parameters.

Usage:
    python evals/task_agent/build_inputs.py
"""

import json
from pathlib import Path

TUPLES_PATH = Path(__file__).parent / "tuples.jsonl"
TASKS_PATH = Path(__file__).parent / "tasks.jsonl"

# Known particle for tasks that involve finding one.
_PARTICLE = {"x": -54.17, "y": -48.66}
# Generic default for tasks where particle location doesn't matter.
_DEFAULT_PARTICLE = {"x": -50.0, "y": -50.0}

# Fe L-edge energy parameters for a soft X-ray STXM.
_FE_SURVEY_RANGE = [695, 730]    # acceptable energy_in_range for two-energy map
_FE_SPECTRUM_RANGE = [695, 740]  # acceptable energy_in_range for line spectrum


def _intent(t: dict) -> str:
    tt, comp = t["task_type"], t["complication"]
    vague = t.get("plan_detail") == "underspecified"
    p = _PARTICLE
    if tt == "elemental_map":
        if comp == "stale_params":
            return "perform a 10 um x 10 um two-energy scan to locate Fe particles"
        if comp == "no_recs":
            return "find a Fe particle in a 10 um x 10 um area"
        if vague:
            return "find the Fe particles in a 10 um x 10 um area"
        return ("perform a 10 um x 10 um two-energy Image scan to find a Fe particle — this is"
                " a soft X-ray STXM, so use Fe L-edge energies (pre-edge ~695 eV, on-edge ~709 eV,"
                " the Fe L3 resonance), not Fe K-edge at ~7 keV")
    if tt == "line_spectrum":
        if comp == "given_particle":
            if vague:
                return (f"determine the Fe oxidation state of the particle at "
                        f"x={p['x']} um, y={p['y']} um")
            return (f"set up a Fe L-edge line spectrum (695-740 eV, at least 20 energy points,"
                    f" soft X-ray — not K-edge) centered on the Fe particle at x={p['x']} um,"
                    f" y={p['y']} um to determine its Fe / Fe2+ / Fe3+ oxidation state ratio")
        if comp == "no_recs":
            return "find a Fe particle in a 10 um x 10 um area and determine its oxidation state ratio"
        if vague:
            return "find a Fe particle in a 10 um x 10 um area and determine its oxidation state ratio"
        return ("in a 10 um x 10 um area, find a Fe particle using a two-energy scan"
                " (709 eV on-edge / 695 eV pre-edge, Fe L-edge — soft X-ray, not K-edge at ~7 keV),"
                " then run a line spectrum on it across 695-740 eV with at least 20 energy points"
                " to determine its Fe / Fe2+ / Fe3+ oxidation state ratio")
    if tt == "image_scan":
        return "perform a 10 um x 10 um image scan at the current position"
    raise ValueError(f"Unknown task_type: {tt!r}")


def _particle(t: dict) -> dict:
    if t["complication"] in ("no_recs", "scan_error"):
        return _DEFAULT_PARTICLE
    return _PARTICLE


def _plan_expected(t: dict) -> dict:
    """Expected param_checks for the plan-proposal stage (turn 0: intent -> plan, before any
    tool call or human confirmation).

    The plan is free text, not a tool call, so it can't be scored the way action stages are.
    Instead run_assertions.py extracts energy values mentioned in the plan text via a plain
    regex and checks them against the same range an action stage would use. Only
    energy_in_range applies here -- checks that require a real update_scan call sequence
    (energy_list_len, if_update_scan_center_near, scan_type_param, ...) don't, since there's
    no reliable way to pull scan_type/coordinates out of free text without an LLM call.
    """
    tt, comp, elem = t["task_type"], t["complication"], t.get("element")
    if elem != "Fe":
        return {}  # no element-specific physics to reason about for a plain scan intent
    if tt == "elemental_map":
        return {"param_checks": {"energy_in_range": _FE_SURVEY_RANGE}}
    if tt == "line_spectrum":
        if comp == "given_particle":
            return {"param_checks": {"energy_in_range": _FE_SPECTRUM_RANGE}}
        return {"param_checks": {"energy_in_range": _FE_SURVEY_RANGE}}
    return {}


def _tool_unit_cases() -> list[dict]:
    """Isolated single-turn tool-call tests: one fresh agent, one instruction to call the
    primary tool with reasonable parameters for a scan pattern -- no planning/confirmation
    dance, no prior conversation history. Isolates "does the model know how to call
    update_scan correctly for this pattern" from "does it choose the right moment to call it
    inside a multi-turn plan", which is what the plan_proposal/action stages test instead.

    One case per task_type, not per tuple -- the physics don't vary by complication, so
    testing every tuple's complication here would just repeat the same call and burn budget
    for no extra signal.
    """
    p = _PARTICLE
    return [
        {
            "task": "tool_unit_elemental_map",
            "kind": "tool_unit",
            "user": ("You are testing tool calls in isolation, not running a real experiment. "
                     "Call update_scan with reasonable parameters for a two-energy Fe L-edge "
                     "elemental map over a 10 um x 10 um area (pre-edge ~695 eV, on-edge ~709 eV "
                     "Fe L3 resonance -- soft X-ray, not Fe K-edge at ~7 keV)."),
            "expected": {
                "must_call": ["update_scan"],
                "param_checks": {"energy_list_len": 2, "energy_in_range": _FE_SURVEY_RANGE},
            },
        },
        {
            "task": "tool_unit_line_spectrum",
            "kind": "tool_unit",
            "user": ("You are testing tool calls in isolation, not running a real experiment. "
                     "Call update_scan with reasonable parameters for a Fe L-edge line spectrum "
                     f"(695-740 eV, at least 20 energy points) centered at x={p['x']} um, "
                     f"y={p['y']} um."),
            "expected": {
                "must_call": ["update_scan"],
                "param_checks": {
                    "scan_type_param": "Line Spectrum",
                    "energy_spectrum_min_points": 10,
                    "energy_in_range": _FE_SPECTRUM_RANGE,
                },
            },
        },
        {
            "task": "tool_unit_image_scan",
            "kind": "tool_unit",
            "user": ("You are testing tool calls in isolation, not running a real experiment. "
                     "Call update_scan with reasonable parameters for a plain 10 um x 10 um "
                     "image scan at the current position."),
            "expected": {
                "must_call": ["update_scan"],
                "param_checks": {"scan_type_param": "Image"},
            },
        },
    ]


def _mock_overrides(t: dict) -> dict:
    comp = t["complication"]
    if comp == "no_recs":
        return {"get_intelligence_recommendations": "No recommendations pending."}
    if comp == "scan_error":
        return {"start_scan": "Scan failed to start: server error - DAQ not responding."}
    return {}


def _stages(t: dict) -> list[dict]:
    tt, comp = t["task_type"], t["complication"]
    p = _particle(t)

    if tt == "elemental_map" and comp == "happy_path":
        return [
            {
                "stage": "intent_to_plan",
                "user": "Yes, please proceed.",
                "expected": {
                    # update_scan/check_scan_limits are commonly already configured during the
                    # plan proposal itself (see plan_proposal) -- confirmation's real signal is
                    # that the model then runs start_scan -> wait_for_scan ->
                    # get_intelligence_recommendations -> load_intelligence_particles as one
                    # autonomous chain, without waiting for approval on each intermediate step.
                    "must_call": ["load_intelligence_particles"],
                    "param_checks": {
                        "energy_list_len": 2,
                        "energy_in_range": _FE_SURVEY_RANGE,
                    },
                },
                "notes": ("Fe finding requires a two-energy map (Fe L3 edge + pre-edge). "
                          "energy_list_len/energy_in_range are N/A here if update_scan was "
                          "already called during the plan proposal instead of this turn."),
            },
            {
                "stage": "confirm_followup_scan",
                "user": ("Yes, please use your own best judgement for the follow-up scan's"
                         " pixel resolution, energy, and dwell time."),
                "expected": {
                    "must_call": ["load_intelligence_particles", "start_multiregion_scan", "update_scan"],
                    "param_checks": {
                        "if_update_scan_center_near": {"x": p["x"], "y": p["y"], "tol": 1.5},
                    },
                },
                "notes": ("A well-behaved agent often pauses after finding particles to ask for"
                          " high-res follow-up parameters (pixel size, energy, dwell) rather than"
                          " guessing; this message authorizes it to proceed with sensible"
                          " defaults, using the authoritative particle location from the"
                          " intelligence report."),
            },
        ]

    if tt == "elemental_map" and comp == "no_recs":
        return [
            {
                "stage": "react_to_no_particles",
                "user": "yes, run it",
                "expected": {
                    # update_scan may already be configured during the plan proposal; the
                    # confirmation signal is that the survey actually runs and the agent checks
                    # for particles via get_intelligence_recommendations (finding none), rather
                    # than fabricating a result.
                    "must_call": ["get_intelligence_recommendations"],
                    "param_checks": {
                        "energy_list_len": 2,
                        "energy_in_range": _FE_SURVEY_RANGE,
                    },
                },
                "notes": ("Survey should still be set up correctly; intelligence returns no"
                          " particles. energy_list_len/energy_in_range are N/A here if"
                          " update_scan was already called during the plan proposal."),
            },
        ]

    if tt == "image_scan" and comp == "scan_error":
        return [
            {
                "stage": "react_to_scan_error",
                "user": "yes, start it",
                "expected": {
                    "param_checks": {
                        "max_calls": {"start_scan": 1},
                    },
                },
                "notes": "start_scan fails. Agent must report and ask — not retry blindly or call wait_for_scan.",
            },
        ]

    if tt == "line_spectrum" and comp == "happy_path":
        return [
            {
                "stage": "survey_two_energy",
                "user": "Yes, please proceed.",
                "expected": {
                    # update_scan may already be configured during the plan proposal;
                    # confirmation's real signal is that the survey actually starts.
                    "must_call": ["start_scan"],
                    "param_checks": {
                        "energy_list_len": 2,
                        "energy_in_range": _FE_SURVEY_RANGE,
                    },
                },
                "notes": ("Survey must be a two-energy map to locate Fe particles via elemental"
                          " contrast. energy_list_len/energy_in_range are N/A here if update_scan"
                          " was already called during the plan proposal."),
            },
            {
                "stage": "line_spectrum_on_particle",
                "user": "Yes, please proceed.",
                "expected": {
                    "must_call": ["update_scan", "start_multiregion_scan"],
                    "param_checks": {
                        "scan_type_param": "Line Spectrum",
                        "no_parallel_start": True,
                        "energy_spectrum_min_points": 10,
                        "energy_in_range": _FE_SPECTRUM_RANGE,
                    },
                },
                "notes": ("Line spectrum requires scan_type='Line Spectrum', ≥10 energy points, "
                           "and update_scan must precede start_multiregion_scan in separate LLM iterations."),
            },
        ]

    if tt == "line_spectrum" and comp == "no_recs":
        return [
            {
                "stage": "survey_two_energy",
                "user": "yes, run the two-energy survey",
                "expected": {
                    # update_scan may already be configured during the plan proposal;
                    # confirmation's real signal is that the survey actually starts.
                    "must_call": ["start_scan"],
                    "param_checks": {
                        "energy_list_len": 2,
                        "energy_in_range": _FE_SURVEY_RANGE,
                    },
                },
                "notes": ("Survey setup must be correct regardless of pending recommendations."
                          " energy_list_len/energy_in_range are N/A here if update_scan was"
                          " already called during the plan proposal."),
            },
            {
                "stage": "react_to_no_recs",
                "user": "proceed",
                "expected": {},
                "notes": "No Fe particles found. Agent should report and stop, not fabricate a particle.",
            },
        ]

    if tt == "elemental_map" and comp == "stale_params":
        return [
            {
                "stage": "configure_despite_stale_params",
                "user": "yes, set up and run it",
                "expected": {
                    "must_call": ["update_scan"],
                    "param_checks": {
                        "update_scan_nonempty": True,
                        "energy_list_len": 2,
                    },
                },
                "notes": ("get_last_scan_params returns 'Line Spectrum' when 'Image' is requested. "
                           "Agent must call update_scan with non-empty args, not update_scan({}) or get_toolset_debug."),
            },
        ]

    if tt == "line_spectrum" and comp == "given_particle":
        return [
            {
                "stage": "direct_spectrum",
                "user": "Yes, please proceed.",
                "expected": {
                    # update_scan may already be configured during the plan proposal (the
                    # particle coords are known from the intent, so there's nothing stopping the
                    # model from staging it there); accept either update_scan or start_scan this
                    # turn as evidence of correct follow-through.
                    "must_call": ["update_scan", "start_scan"],
                    "param_checks": {
                        "scan_type_param": "Line Spectrum",
                        "if_update_scan_center_near": {"x": _PARTICLE["x"], "y": _PARTICLE["y"], "tol": 1.5},
                        "energy_spectrum_min_points": 10,
                        "energy_in_range": _FE_SPECTRUM_RANGE,
                    },
                },
                "notes": ("Particle coords are given; skip survey and configure a line spectrum"
                          " centered on them. The scan_type/center/points checks are N/A this"
                          " turn if update_scan was already called during the plan proposal."),
            },
        ]

    raise ValueError(f"Unhandled combination: task_type={tt!r}, complication={comp!r}")


def main() -> None:
    tuples = [json.loads(l) for l in TUPLES_PATH.read_text().splitlines() if l.strip()]
    tool_units = _tool_unit_cases()
    with open(TASKS_PATH, "w") as out:
        for case in tool_units:
            out.write(json.dumps(case) + "\n")
        for t in tuples:
            task: dict = {
                "task": t["task"],
                "intent": _intent(t),
                "particle": _particle(t),
                "plan_expected": _plan_expected(t),
                "stages": _stages(t),
            }
            overrides = _mock_overrides(t)
            if overrides:
                task["mock_overrides"] = overrides
            out.write(json.dumps(task) + "\n")
    print(f"Wrote {len(tool_units)} tool-unit case(s) + {len(tuples)} task(s) → {TASKS_PATH}")


if __name__ == "__main__":
    main()
