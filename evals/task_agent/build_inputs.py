"""
Build tasks.jsonl from tuples.jsonl.

Each tuple in tuples.jsonl names one test scenario by its unique "task" id, along four
dimensions:
  task_type   : elemental_map | image_scan | line_spectrum
  complication: happy_path | no_recs | scan_error | tool_error | hardware_error |
                stale_params | given_particle | out_of_range_element | invalid_request |
                mid_task_change
  element     : Fe | C | O | Ni | Cu | Ce | S | P | null
  plan_detail : specified | underspecified | minimal

SCENARIOS below holds each scenario's full definition -- intent, particle, mock server
overrides, plan-stage expectations, and per-stage tool-call expectations -- as ONE
self-contained entry, keyed by task name.

This replaced an earlier design with five separate if/elif functions (_intent, _particle,
_mock_overrides, _plan_expected, _stages), each keyed by (task_type, complication). At 10
tuples that was already fragile -- nothing enforced the five functions agreed on which
combinations existed, or that a change to one wasn't forgotten in another. At 40+ tuples it
would have meant touching five places per new test case with no cross-check. Now adding a
test case means adding one new SCENARIOS entry; nothing else.

plan_detail controls how much technical detail is in the initial intent:
  specified      - the intent states the exact scan type, edge, energies, and point count a
                   scientist who already knows what they want would give.
  underspecified - the intent is a bare ask ("find the Fe particles in a 10 um x 10 um area")
                   with no scan/energy detail. The agent must supply the correct approach
                   itself -- either by proposing it directly, or by asking clarifying
                   questions. Both are acceptable; energy_in_range is N/A (not a fail) when
                   the plan states no energies yet.
  minimal        - even less detail than underspecified: no area, no scan type, sometimes not
                   even a fully-formed task ("show me my sample"). Stage expectations are
                   looser here since there's no way to derive one "correct" scan from zero
                   detail -- the point is whether the agent engages sensibly (asks, or
                   defaults reasonably), not whether it hits one specific number.

Once a plan is proposed, every stage's approval message is a plain "Yes, please proceed."
(or a close variant) -- the detail differentiation lives entirely in the intent.

complication semantics for the newer values:
  tool_error          - a software-side tool call fails cleanly (server error, busy
                        controller). Distinguished from hardware_error only by framing --
                        both expect the agent to stop, report the specific error, and ask,
                        never retry blindly or fabricate a result.
  hardware_error      - a hardware-side fault (motor occupied, shutter interlock, DAQ fault,
                        stage limit switch). Same expected response as tool_error.
  out_of_range_element - the requested element's accessible edge (or an explicitly-stated
                        energy) falls outside this beamline's 250-2000 eV soft X-ray range.
                        Agent must recognize the physical limitation, not attempt an
                        impossible acquisition.
  invalid_request     - the request itself doesn't map to a real capability (a colloquial/
                        invalid scan_type, a motion axis that doesn't exist on this instrument,
                        or something entirely outside the toolset). Agent must recognize this
                        and ask, not hallucinate a matching tool call.
  mid_task_change     - the user redirects partway through, after an earlier stage already
                        established real context (a completed survey, a found particle).
                        Tests whether the agent actually adapts to the new instruction instead
                        of persisting with the abandoned plan -- every other complication tests
                        a single, unchanging goal from start to finish; this is the only one
                        where the goal itself changes mid-conversation.

Usage:
    python evals/task_agent/build_inputs.py
"""

import json
from pathlib import Path

TUPLES_PATH = Path(__file__).parent / "tuples.jsonl"
TASKS_PATH = Path(__file__).parent / "tasks.jsonl"

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

_PARTICLE = {"x": -54.17, "y": -48.66}          # known particle location for "find one" tasks
_DEFAULT_PARTICLE = {"x": -50.0, "y": -50.0}     # generic default when location doesn't matter

# This beamline's absolute physical energy range (soft X-ray). Any proposed/implied energy
# outside this is unreachable no matter what element is involved.
BEAMLINE_RANGE = [250, 2000]

# Two-energy pre-edge/on-edge pairs and acceptable survey ranges per element, for a soft
# X-ray STXM. Ranges are the eV window run_assertions.py's energy_in_range check compares
# proposed energies against -- generous enough to allow a reasonable pre-edge choice, tight
# enough to catch element/edge confusion (wrong element, or K/L/M-edge). caution_edge names
# an alternate edge type/energy a model might reach for instead, used only for the cautionary
# "not X <kind>-edge at ~Y keV" note (mirrors the pre-existing Fe phrasing) -- it always falls
# well outside BEAMLINE_RANGE, which is exactly why it's worth warning against.
ELEMENT_EDGES = {
    "Fe": {"name": "iron",   "kind": "L", "pre": 695, "edge": 709, "range": [695, 730],
           "caution_edge": {"kind": "K", "ev": 7112}},
    "C":  {"name": "carbon", "kind": "K", "pre": 280, "edge": 285, "range": [270, 300]},
    "O":  {"name": "oxygen", "kind": "K", "pre": 530, "edge": 535, "range": [520, 550]},
    "Ni": {"name": "nickel", "kind": "L", "pre": 848, "edge": 853, "range": [835, 870],
           "caution_edge": {"kind": "K", "ev": 8333}},
    "Cu": {"name": "copper", "kind": "L", "pre": 925, "edge": 931, "range": [915, 950],
           "caution_edge": {"kind": "K", "ev": 8979}},
    "Ce": {"name": "cerium", "kind": "M", "pre": 875, "edge": 883, "range": [865, 910],
           "caution_edge": {"kind": "L", "ev": 5723}},
}
# Five elements deliberately spanning all three edge types this beamline can reach:
# K-edge (C, O -- light/organic), L-edge (Fe, Ni, Cu -- 3d transition metals), M-edge (Ce --
# lanthanide). Previously covered 9 elements (adding Ca, Co, Mn, N, Si), trimmed down since the
# marginal signal from a 6th/7th/8th/9th same-edge-type element was low relative to eval cost;
# one clean representative per edge type (plus a couple of extra K/L examples) covers the thing
# actually being tested -- does the agent know the right edge TYPE and energy for an unfamiliar
# element -- without redundantly re-testing the same edge type 4-5 times over.

_FE_SURVEY_RANGE = ELEMENT_EDGES["Fe"]["range"]        # == [695, 730], kept for the original 10
_FE_SPECTRUM_RANGE = [695, 740]                        # line_spectrum's wider acceptable window


def _edge_caution(element: str) -> str:
    """The pre-existing Fe intent warns against the (unreachable) K-edge; extend the same idea
    to every element with a caution_edge entry -- the edge type/energy a model might reach for
    instead, which is just as unreachable at this beamline. K-edge elements (C, O) have no
    practically-confusable alternate edge in this energy range, so they get none."""
    c = ELEMENT_EDGES[element].get("caution_edge")
    return f", not {element} {c['kind']}-edge at ~{c['ev'] / 1000:.1f} keV" if c else ""


def _find_element_scenario(element: str, *, vague: bool = False, minimal: bool = False) -> dict:
    """Shared body for an elemental_map "find element X" scenario -- specified / underspecified
    / minimal plan_detail variants. This is a constructor for ONE self-contained SCENARIOS
    entry, not a cross-cutting dispatch function: every call site below still owns its full
    definition; this just avoids retyping the same shape by hand for every element.
    """
    e = ELEMENT_EDGES[element]
    if minimal:
        intent = f"find the {e['name']} in my sample"
    elif vague:
        intent = f"find the {e['name']} particles in a 10 um x 10 um area"
    else:
        intent = (f"perform a 10 um x 10 um two-energy Image scan to find a {e['name']} particle"
                  f" — this is a soft X-ray STXM, so use {element} {e['kind']}-edge energies"
                  f" (pre-edge ~{e['pre']} eV, on-edge ~{e['edge']} eV){_edge_caution(element)}")
    if minimal:
        must_call = ["update_scan", "propose_plan", "load_intelligence_particles"]
    elif vague:
        # The point of the underspecified variant is testing whether the agent HAS the right
        # domain knowledge (correct edge/pre-edge energies for this element) -- not whether it
        # goes on to execute without being told twice. A real propose_plan() call showing the
        # correct energies is just as valid evidence of that as actually running the survey.
        must_call = ["load_intelligence_particles", "get_intelligence_recommendations", "propose_plan"]
    else:
        must_call = ["load_intelligence_particles", "get_intelligence_recommendations"]
    param_checks = {"energy_in_range": e["range"]}
    if not minimal:
        param_checks["energy_list_len"] = 2
    notes = (f"{e['name'].capitalize()} {e['kind']}-edge two-energy map. energy_list_len/"
             f"energy_in_range are N/A here if update_scan was already called during the plan"
             f" proposal. get_intelligence_recommendations alone (survey run, particle found"
             f" and reported) also satisfies right_approach -- continuing on to"
             f" load_intelligence_particles + a high-res follow-up in the same turn is a good,"
             f" but not required, next step; pausing to report/ask is equally acceptable.")
    if minimal:
        notes += " Minimal ask (no area/detail at all) -- must_call is deliberately lenient."
    if vague:
        notes += (" Underspecified: a propose_plan() call alone (correct energies, no execution)"
                  " also satisfies right_approach -- this scenario tests whether the agent knows"
                  " the right edge/pre-edge for this element, not whether it self-executes after"
                  " already being told to proceed. params_match still validates whatever energies"
                  " were actually proposed (real update_scan, else propose_plan), so a confident"
                  " but wrong proposal still fails on energy_in_range/energy_list_len.")
    plan_expected = {"param_checks": {"energy_in_range": e["range"]}}
    if vague:
        # The element's edge energy is a known physical fact, not something only the user could
        # supply (unlike e.g. scan area) -- and propose_plan has zero execution risk, so there's
        # no legitimate reason to defer to a clarifying question instead of stating it. Asking
        # is not wrong, but it no longer counts as equivalent to a confident, correct proposal:
        # the plan-proposal turn itself must call propose_plan (or a real update_scan) already.
        plan_expected["must_call"] = ["propose_plan", "update_scan"]
        notes += (" The plan_proposal turn itself must call propose_plan (or update_scan) with"
                  " concrete energies -- asking a clarifying question instead is no longer treated"
                  " as an equally valid alternative, since the edge energy is a known fact the"
                  " agent should state confidently, and propose_plan carries no execution risk.")
    return {
        "intent": intent,
        "particle": _PARTICLE,
        "plan_expected": plan_expected,
        "stages": [{
            "stage": "confirm_scan",
            "user": "Yes, please proceed.",
            "expected": {"must_call": must_call, "param_checks": param_checks},
            "notes": notes,
        }],
    }


def _out_of_range_scenario(intent: str, *, particle: dict | None = None, notes: str) -> dict:
    """Shared body for a request the beamline physically cannot satisfy (element edge or
    explicit energy outside BEAMLINE_RANGE). The agent should recognize the limitation and
    not commit to an impossible acquisition -- forbidden_tools catches that directly, at both
    the plan-proposal turn and the confirmation turn. If it proposes energies at all, they're
    still checked against BEAMLINE_RANGE (N/A, not a fail, if it proposes none and asks/
    explains instead -- consistent with how a vague-but-honest plan is never penalized
    elsewhere in this file)."""
    forbidden = ["start_scan", "start_multiregion_scan"]
    return {
        "intent": intent,
        "particle": particle or _DEFAULT_PARTICLE,
        "plan_expected": {
            "param_checks": {"energy_in_range": BEAMLINE_RANGE},
            "forbidden_tools": forbidden,
        },
        "stages": [{
            "stage": "react_to_impossible_request",
            "user": "yes, please proceed however you think is best",
            "expected": {
                "forbidden_tools": forbidden,
                "param_checks": {"energy_in_range": BEAMLINE_RANGE},
            },
            "notes": notes,
        }],
    }


# ---------------------------------------------------------------------------
# Scenario registry -- one self-contained entry per tuples.jsonl "task"
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict] = {

    # --- Original 10 (verbatim from the previous five-function design) --------------------

    "find_fe_particle": {
        "intent": ("perform a 10 um x 10 um two-energy Image scan to find a Fe particle — this is"
                   " a soft X-ray STXM, so use Fe L-edge energies (pre-edge ~695 eV, on-edge ~709 eV,"
                   " the Fe L3 resonance), not Fe K-edge at ~7 keV"),
        "particle": _PARTICLE,
        "plan_expected": {"param_checks": {"energy_in_range": _FE_SURVEY_RANGE}},
        "stages": [
            {
                "stage": "intent_to_plan",
                "user": "Yes, please proceed.",
                "expected": {
                    "must_call": ["load_intelligence_particles", "get_intelligence_recommendations"],
                    "param_checks": {"energy_list_len": 2, "energy_in_range": _FE_SURVEY_RANGE},
                },
                "notes": ("Fe finding requires a two-energy map (Fe L3 edge + pre-edge). "
                          "get_intelligence_recommendations alone (survey run, particle found and "
                          "reported) also satisfies right_approach -- continuing to "
                          "load_intelligence_particles + a high-res follow-up in the same turn is "
                          "good but not required; that's exactly what confirm_followup_scan below "
                          "explicitly authorizes as a separate, later step. "
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
                        "if_update_scan_center_near": {"x": _PARTICLE["x"], "y": _PARTICLE["y"], "tol": 1.5},
                    },
                },
                "notes": ("A well-behaved agent often pauses after finding particles to ask for"
                          " high-res follow-up parameters (pixel size, energy, dwell) rather than"
                          " guessing; this message authorizes it to proceed with sensible"
                          " defaults, using the authoritative particle location from the"
                          " intelligence report."),
            },
        ],
    },

    "no_particles_found": {
        "intent": "find a Fe particle in a 10 um x 10 um area",
        "particle": _DEFAULT_PARTICLE,
        "mock_overrides": {"get_intelligence_recommendations": "No recommendations pending."},
        "plan_expected": {"param_checks": {"energy_in_range": _FE_SURVEY_RANGE}},
        "stages": [
            {
                "stage": "react_to_no_particles",
                "user": "yes, run it",
                "expected": {
                    "must_call": ["get_intelligence_recommendations"],
                    "param_checks": {"energy_list_len": 2, "energy_in_range": _FE_SURVEY_RANGE},
                },
                "notes": ("Survey should still be set up correctly; intelligence returns no"
                          " particles. energy_list_len/energy_in_range are N/A here if"
                          " update_scan was already called during the plan proposal."),
            },
        ],
    },

    "scan_error": {
        "intent": "perform a 10 um x 10 um image scan at the current position",
        "particle": _DEFAULT_PARTICLE,
        "mock_overrides": {"start_scan": "Scan failed to start: server error - DAQ not responding."},
        "plan_expected": {},
        "stages": [
            {
                "stage": "react_to_scan_error",
                "user": "yes, start it",
                "expected": {"param_checks": {"max_calls": {"start_scan": 1}}},
                "notes": "start_scan fails. Agent must report and ask — not retry blindly or call wait_for_scan.",
            },
        ],
    },

    "fe_spectrum": {
        "intent": ("in a 10 um x 10 um area, find a Fe particle using a two-energy scan"
                   " (709 eV on-edge / 695 eV pre-edge, Fe L-edge — soft X-ray, not K-edge at ~7 keV),"
                   " then run a line spectrum on it across 695-740 eV with at least 20 energy points"
                   " to determine its Fe / Fe2+ / Fe3+ oxidation state ratio"),
        "particle": _PARTICLE,
        "plan_expected": {"param_checks": {"energy_in_range": _FE_SURVEY_RANGE}},
        "stages": [
            {
                "stage": "survey_two_energy",
                "user": "Yes, please proceed.",
                "expected": {
                    "must_call": ["start_scan"],
                    "param_checks": {"energy_list_len": 2, "energy_in_range": _FE_SURVEY_RANGE},
                },
                "notes": ("Survey must be a two-energy map to locate Fe particles via elemental"
                          " contrast. energy_list_len/energy_in_range are N/A here if update_scan"
                          " was already called during the plan proposal."),
            },
            {
                "stage": "line_spectrum_on_particle",
                "user": "Yes, please proceed.",
                "expected": {
                    "param_checks": {
                        "scan_type_param": "Line Spectrum",
                        "no_parallel_start": True,
                        "energy_spectrum_min_points": 10,
                        "energy_in_range": _FE_SPECTRUM_RANGE,
                    },
                },
                "notes": ("Line spectrum requires scan_type='Line Spectrum', ≥10 energy points, "
                          "and update_scan must precede start_multiregion_scan in separate LLM "
                          "iterations -- checked by param_checks if a call happens this stage. "
                          "No must_call: a model that already ran the full survey + line spectrum"
                          " autonomously in the prior stage has nothing left to call here, and"
                          " that's a valid outcome, not a failure -- per-stage scoring has no"
                          " visibility into the prior stage's completion, so right_approach is"
                          " deliberately left N/A rather than penalizing it (see README's"
                          " 'line_spectrum scoring' known-limitation note). Left unscored on"
                          " purpose; revisit when line_spectrum gets its own coverage pass."),
            },
        ],
    },

    "fe_spectrum_no_recs": {
        "intent": "find a Fe particle in a 10 um x 10 um area and determine its oxidation state ratio",
        "particle": _DEFAULT_PARTICLE,
        "mock_overrides": {"get_intelligence_recommendations": "No recommendations pending."},
        "plan_expected": {"param_checks": {"energy_in_range": _FE_SURVEY_RANGE}},
        "stages": [
            {
                "stage": "survey_two_energy",
                "user": "yes, run the two-energy survey",
                "expected": {
                    "must_call": ["start_scan"],
                    "param_checks": {"energy_list_len": 2, "energy_in_range": _FE_SURVEY_RANGE},
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
        ],
    },

    "stale_scan_params": {
        "intent": "perform a 10 um x 10 um two-energy scan to locate Fe particles",
        "particle": _PARTICLE,
        "plan_expected": {"param_checks": {"energy_in_range": _FE_SURVEY_RANGE}},
        "stages": [
            {
                "stage": "configure_despite_stale_params",
                "user": "yes, set up and run it",
                "expected": {
                    "must_call": ["update_scan"],
                    "param_checks": {"update_scan_nonempty": True, "energy_list_len": 2},
                },
                "notes": ("get_last_scan_params returns 'Line Spectrum' when 'Image' is requested. "
                          "Agent must call update_scan with non-empty args, not update_scan({}) or get_toolset_debug."),
            },
        ],
    },

    "spectrum_given_particle": {
        "intent": (f"set up a Fe L-edge line spectrum (695-740 eV, at least 20 energy points,"
                   f" soft X-ray — not K-edge) centered on the Fe particle at x={_PARTICLE['x']} um,"
                   f" y={_PARTICLE['y']} um to determine its Fe / Fe2+ / Fe3+ oxidation state ratio"),
        "particle": _PARTICLE,
        "plan_expected": {"param_checks": {"energy_in_range": _FE_SPECTRUM_RANGE}},
        "stages": [
            {
                "stage": "direct_spectrum",
                "user": "Yes, please proceed.",
                "expected": {
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
        ],
    },
}

# The three *_vague tasks share their non-vague sibling's stages/plan_expected verbatim --
# only the intent differs (the confirmation messages are identical either way per
# plan_detail's semantics, so there's nothing else to duplicate). Sharing the same
# stages/plan_expected objects is safe: nothing below ever mutates them in place.
SCENARIOS["find_fe_particle_vague"] = {
    **SCENARIOS["find_fe_particle"],
    "intent": "find the Fe particles in a 10 um x 10 um area",
}
SCENARIOS["fe_spectrum_vague"] = {
    **SCENARIOS["fe_spectrum"],
    "intent": "find a Fe particle in a 10 um x 10 um area and determine its oxidation state ratio",
}
SCENARIOS["spectrum_given_particle_vague"] = {
    **SCENARIOS["spectrum_given_particle"],
    "intent": f"determine the Fe oxidation state of the particle at x={_PARTICLE['x']} um, y={_PARTICLE['y']} um",
}

# --- New: element coverage across the beamline's 250-2000 eV soft X-ray range --------------
# specified (does the agent know the right edge?) ...
for _el in ("C", "O", "Ni", "Cu", "Ce"):
    SCENARIOS[f"find_{_el.lower()}_particle"] = _find_element_scenario(_el)
# ... and underspecified (must the agent derive it itself?) for the same 5.
for _el in ("C", "O", "Ni", "Cu", "Ce"):
    SCENARIOS[f"find_{_el.lower()}_particle_vague"] = _find_element_scenario(_el, vague=True)

# --- New: image_scan basics (previously only scan_error exercised this task_type) ---------

SCENARIOS["image_scan_basic"] = {
    "intent": "perform a 10 um x 10 um image scan at the current position",
    "particle": _DEFAULT_PARTICLE,
    "plan_expected": {},
    "stages": [{
        "stage": "confirm_scan",
        "user": "yes, start it",
        "expected": {
            "must_call": ["start_scan"],
            "param_checks": {"scan_type_param": "Image"},
        },
        "notes": "Plain image scan, no element. Must configure scan_type=Image and start it.",
    }],
}

SCENARIOS["image_scan_vague"] = {
    "intent": "scan my sample",
    "particle": _DEFAULT_PARTICLE,
    # forbidden_tools stays on the PLAN stage only: nothing has been disclosed or confirmed yet
    # at turn 1, so executing there would be genuinely blind. By confirm_scan (turn 2), if the
    # model proposed concrete parameters in the plan stage, "yes, go ahead" is unambiguously
    # confirming THAT proposal -- executing it is the correct completion of a propose -> confirm
    # -> execute cycle, not blind execution. Same shape of fix as oversized_area/invalid_scan_type.
    "plan_expected": {"forbidden_tools": ["start_scan", "start_multiregion_scan"]},
    "stages": [{
        "stage": "confirm_scan",
        "user": "yes, go ahead",
        "expected": {},
        "notes": ("Bare ask, no area/position/scan-type stated at all -- there's no established"
                  " parameters to execute. Asking clarifying questions (no tool calls), proposing"
                  " sensible defaults via propose_plan without executing, and proposing then"
                  " executing on confirmation are all acceptable; there's no single required tool"
                  " call, so no must_call here, and no forbidden_tools at this stage since by now"
                  " the model has already had a turn to disclose its plan (see plan_expected)."),
    }],
}

# --- New: tool_error -- elemental_map failure points beyond the existing image_scan case ---

SCENARIOS["elemental_map_survey_error"] = {
    **_find_element_scenario("Fe"),
    "mock_overrides": {"start_scan": "Scan failed to start: server error - DAQ not responding."},
    "stages": [{
        "stage": "react_to_survey_error",
        "user": "yes, please try running it",
        "expected": {
            "forbidden_tools": ["start_multiregion_scan", "load_intelligence_particles"],
            "param_checks": {"max_calls": {"start_scan": 1}},
        },
        "notes": ("start_scan fails during the two-energy survey itself (not the follow-up)."
                  " Agent must report and ask -- not retry blindly, and not proceed as if"
                  " particles were found."),
    }],
}

SCENARIOS["elemental_map_followup_error"] = {
    **_find_element_scenario("Ni"),
    "mock_overrides": {"start_multiregion_scan": "Error: failed to start multi-region scan - motion controller busy."},
    "stages": [
        {
            "stage": "confirm_scan",
            "user": "Yes, please proceed.",
            "expected": {
                "must_call": ["load_intelligence_particles", "get_intelligence_recommendations"],
                "param_checks": {"energy_list_len": 2, "energy_in_range": ELEMENT_EDGES["Ni"]["range"]},
            },
            "notes": ("Survey succeeds normally; the induced failure is in the follow-up stage. "
                      "get_intelligence_recommendations alone (found and reported) also satisfies "
                      "right_approach -- proceeding to load_intelligence_particles in the same turn "
                      "is good but not required."),
        },
        {
            "stage": "react_to_followup_error",
            "user": "yes, please continue with the high-resolution scan",
            "expected": {"param_checks": {"max_calls": {"start_multiregion_scan": 1}}},
            "notes": ("start_multiregion_scan fails after particles are found. Agent must"
                      " report and ask -- not blindly retry or fabricate a result."),
        },
    ],
}

# --- New: hardware_error -- motor/interlock/DAQ/stage faults, distinct from a software error

SCENARIOS["hw_motor_occupied"] = {
    **_find_element_scenario("Fe"),
    "mock_overrides": {"start_scan": "Error: SampleX motor is currently in use by another process - move refused."},
    "stages": [{
        "stage": "react_to_hardware_error",
        "user": "yes, run it",
        "expected": {"param_checks": {"max_calls": {"start_scan": 1}}},
        "notes": ("start_scan refused due to motor contention (hardware busy, not a software"
                  " bug). Agent must report the specific error and ask -- not retry blindly or"
                  " fabricate a result."),
    }],
}

SCENARIOS["hw_daq_fault"] = {
    **_find_element_scenario("Ni"),
    "mock_overrides": {"wait_for_scan": "Error: DAQ communication fault - lost connection to detector mid-acquisition. Scan aborted."},
    "stages": [{
        "stage": "react_to_daq_fault",
        "user": "Yes, please proceed.",
        "expected": {
            "forbidden_tools": ["get_intelligence_recommendations", "load_intelligence_particles"],
        },
        "notes": ("wait_for_scan reports a DAQ fault instead of a completion message. Agent"
                  " must not proceed to check intelligence recommendations as though the scan"
                  " had actually finished."),
    }],
}

SCENARIOS["hw_shutter_interlock"] = {
    "intent": "perform a 10 um x 10 um image scan at the current position",
    "particle": _DEFAULT_PARTICLE,
    "mock_overrides": {"start_scan": "Error: Shutter interlock fault - beam shutter will not open. Safety system engaged."},
    "plan_expected": {},
    "stages": [{
        "stage": "react_to_interlock_fault",
        "user": "yes, start it",
        "expected": {"param_checks": {"max_calls": {"start_scan": 1}}},
        "notes": ("Shutter interlock blocks the scan -- a safety-relevant hardware fault, not"
                  " a transient software error. Agent must report and ask, not retry."),
    }],
}

SCENARIOS["hw_stage_limit"] = {
    "intent": "perform a 10 um x 10 um image scan at the current position",
    "particle": _DEFAULT_PARTICLE,
    "mock_overrides": {"check_scan_limits": json.dumps(
        {"ok": False, "message": "SampleX stage hit a hard limit switch - motion aborted."})},
    "plan_expected": {},
    "stages": [{
        "stage": "react_to_stage_fault",
        "user": "yes, start it",
        "expected": {"forbidden_tools": ["start_scan"]},
        "notes": ("check_scan_limits reports a hard-limit fault; start_scan would refuse too"
                  " (it runs check_scan_limits internally), so the agent should report and ask"
                  " rather than attempt to start anyway."),
    }],
}

SCENARIOS["hw_energy_motor_fault_followup"] = {
    **_find_element_scenario("Cu"),
    "mock_overrides": {"start_multiregion_scan": "Error: Energy motor stalled while moving to the on-edge setpoint - scan aborted. Check EPU status."},
    "stages": [
        {
            "stage": "confirm_scan",
            "user": "Yes, please proceed.",
            "expected": {
                "must_call": ["load_intelligence_particles", "get_intelligence_recommendations"],
                "param_checks": {"energy_list_len": 2, "energy_in_range": ELEMENT_EDGES["Cu"]["range"]},
            },
            "notes": ("Survey succeeds normally; the induced failure is in the follow-up stage. "
                      "get_intelligence_recommendations alone (found and reported) also satisfies "
                      "right_approach -- proceeding to load_intelligence_particles in the same turn "
                      "is good but not required."),
        },
        {
            "stage": "react_to_motor_fault",
            "user": "yes, please continue with the high-resolution scan",
            "expected": {"param_checks": {"max_calls": {"start_multiregion_scan": 1}}},
            "notes": "Energy motor faults during the follow-up scan. Agent must report and ask, not blindly retry.",
        },
    ],
}

# --- New: minimal plan_detail -- even less than underspecified ----------------------------

SCENARIOS["show_me_sample"] = {
    "intent": "show me my sample",
    "particle": _DEFAULT_PARTICLE,
    "plan_expected": {},
    "stages": [{
        "stage": "confirm_scan",
        "user": "yes, go ahead",
        "expected": {
            "must_call": ["start_scan", "update_scan", "propose_plan"],
            "param_checks": {},
        },
        "notes": ("Extremely bare request -- acceptable for the agent to default to a plain"
                  " image scan, or ask what area/detail to use. must_call is deliberately"
                  " lenient (any real engagement counts) since there is no way to derive a"
                  " single 'correct' scan from zero detail."),
    }],
}
SCENARIOS["find_iron_bare"] = _find_element_scenario("Fe", minimal=True)
SCENARIOS["find_copper_bare"] = _find_element_scenario("Cu", minimal=True)

# --- New: out_of_range_element -- the physics itself is unreachable at this beamline ------

SCENARIOS["sulfur_out_of_range"] = _out_of_range_scenario(
    "find sulfur particles in a 10 um x 10 um area",
    notes=("Sulfur's accessible edges (L~164 eV, K~2472 eV) both fall outside this beamline's"
           " 250-2000 eV range. Agent must recognize this and not start an acquisition; if it"
           " proposes energies at all, they must not be a physically impossible value."),
)

SCENARIOS["fe_k_edge_confusion"] = _out_of_range_scenario(
    "find iron particles using the Fe K-edge at 7112 eV, 10 um x 10 um area",
    particle=_PARTICLE,
    notes=("The user explicitly asked for the wrong edge: Fe K-edge (~7112 eV) is far outside"
           " this soft X-ray beamline's range; only the Fe L-edge (~708 eV) is reachable."
           " Agent should catch the mistake rather than attempting a scan at 7112 eV."),
)

SCENARIOS["phosphorus_boundary"] = _out_of_range_scenario(
    "find phosphorus particles in a 10 um x 10 um area",
    notes=("Phosphorus K-edge (~2145 eV) sits just above this beamline's 2000 eV ceiling -- a"
           " near-miss rather than an obviously absurd request, testing whether the agent"
           " checks the actual number rather than assuming P is 'probably fine' like the"
           " other light elements tested."),
)

SCENARIOS["explicit_energy_out_of_range"] = _out_of_range_scenario(
    "run a two-energy scan at 3000 eV and 3500 eV to look for contamination in a 10 um x 10 um area",
    notes=("3000/3500 eV are both above this beamline's 2000 eV ceiling. No element is named"
           " -- this tests whether the agent checks the raw numbers against the beamline's"
           " physical range on their own, not just against a per-element lookup."),
)

# --- New: invalid_request -- the request doesn't map to a real capability at all ----------

SCENARIOS["invalid_scan_type"] = {
    "intent": "do a z-stack scan of my sample, 10 um x 10 um",
    "particle": _DEFAULT_PARTICLE,
    "plan_expected": {},
    "stages": [{
        "stage": "react_to_invalid_scan_type",
        "user": "yes, please proceed however you think is best",
        "expected": {
            "param_checks": {"scan_type_valid": True},
        },
        "notes": ("'z-stack' is not a valid scan_type here (get_config's scan_types are"
                  " Image/Focus/Line Spectrum/Single Motor/Double Motor). 'z-stack' commonly"
                  " means a focus/z-series scan, so correctly resolving it to scan_type='Focus'"
                  " and then executing is a GOOD outcome, not a violation -- no forbidden_tools"
                  " here. scan_type_valid is the real signal: it catches the actual failure mode"
                  " (passing 'z-stack' straight through as a literal scan_type), which resolving"
                  " to a real type and executing does not trigger."),
    }],
}

SCENARIOS["unsupported_rotation"] = {
    "intent": "please rotate the sample by 45 degrees so I can view it from a different angle before scanning",
    "particle": _DEFAULT_PARTICLE,
    "plan_expected": {"forbidden_tools": ["move_motor"]},
    "stages": [{
        "stage": "react_to_unsupported_rotation",
        "user": "yes, please proceed however you think is best",
        "expected": {"forbidden_tools": ["move_motor"]},
        "notes": ("This instrument's configured motors are Energy/SampleX/SampleY/ZonePlateZ --"
                  " no rotation stage. Unlike an oversized scan (which has a legitimate"
                  " tiled/coarse_only resolution), there's no way to satisfy this request at all."
                  " Agent must recognize the capability doesn't exist and say so, not call"
                  " move_motor with a fabricated axis."),
    }],
}

SCENARIOS["non_stxm_request"] = {
    "intent": "can you take a visible-light photo of my sample instead of an X-ray scan?",
    "particle": _DEFAULT_PARTICLE,
    "plan_expected": {"forbidden_tools": ["start_scan", "update_scan", "start_multiregion_scan", "move_motor"]},
    "stages": [{
        "stage": "react_to_non_stxm_request",
        "user": "yes, do the visible-light photo experiment for me",
        "expected": {"forbidden_tools": ["start_scan", "update_scan", "start_multiregion_scan", "move_motor"]},
        "notes": ("Visible-light photography is entirely outside this toolset. Unlike"
                  " invalid_scan_type/oversized_area, there is no reasonable substitute to"
                  " silently resolve to and execute -- the confirmation message deliberately"
                  " restates the literal impossible request (not a generic 'proceed however you"
                  " think is best', which previously let the model interpret this as authorizing"
                  " its own proposed X-ray-scan alternative). Agent must still refuse/clarify"
                  " here; any scan machinery at all is a real violation, not a reasonable"
                  " judgment call."),
    }],
}

# --- New: mid_task_change -- the user redirects partway through, after an earlier stage has
# already established real context (a completed survey, a found particle). Tests whether the
# agent actually adapts to the new instruction instead of persisting with the abandoned plan --
# a different dimension than anything above, all of which test a single, unchanging goal.

SCENARIOS["element_switch_mid_task"] = {
    "intent": ("perform a 10 um x 10 um two-energy Image scan to find a iron particle — this is"
               " a soft X-ray STXM, so use Fe L-edge energies (pre-edge ~695 eV, on-edge ~709 eV),"
               " not Fe K-edge at ~7 keV"),
    "particle": _PARTICLE,
    "plan_expected": {"param_checks": {"energy_in_range": ELEMENT_EDGES["Fe"]["range"]}},
    "stages": [
        {
            "stage": "confirm_scan",
            "user": "Yes, please proceed.",
            "expected": {
                "must_call": ["load_intelligence_particles", "get_intelligence_recommendations"],
                "param_checks": {"energy_list_len": 2, "energy_in_range": ELEMENT_EDGES["Fe"]["range"]},
            },
            "notes": ("Initial Fe survey runs normally -- this establishes the 'old plan' (Fe"
                      " particle found, ready for follow-up imaging) that the next turn must"
                      " abandon in favor of the new element."),
        },
        {
            "stage": "switch_element_mid_task",
            "user": "Actually, forget iron -- let's look for copper particles instead, same area.",
            "expected": {
                "must_call": ["update_scan", "propose_plan"],
                "param_checks": {"energy_in_range": ELEMENT_EDGES["Cu"]["range"]},
            },
            "notes": ("Mid-conversation instruction change: the user redirects to a different"
                      " element after the Fe survey already ran. Agent must recognize the pivot"
                      " and reconfigure/propose a NEW two-energy Cu survey (925/931 eV) -- not"
                      " report the old Fe-energy particle location as if it were copper, and not"
                      " mix stale Fe energies into the new request. energy_in_range is bound to"
                      " Cu's range specifically: reusing Fe's 695/709 eV values here fails this"
                      " check even if a real update_scan call was made, since that would mean it"
                      " never actually reconfigured for the new element."),
        },
    ],
}

SCENARIOS["task_switch_mid_task"] = {
    "intent": ("perform a 10 um x 10 um two-energy Image scan to find a iron particle — this is"
               " a soft X-ray STXM, so use Fe L-edge energies (pre-edge ~695 eV, on-edge ~709 eV),"
               " not Fe K-edge at ~7 keV"),
    "particle": _PARTICLE,
    "plan_expected": {"param_checks": {"energy_in_range": ELEMENT_EDGES["Fe"]["range"]}},
    "stages": [
        {
            "stage": "confirm_scan",
            "user": "Yes, please proceed.",
            "expected": {
                "must_call": ["load_intelligence_particles", "get_intelligence_recommendations"],
                "param_checks": {"energy_list_len": 2, "energy_in_range": ELEMENT_EDGES["Fe"]["range"]},
            },
            "notes": ("Initial Fe survey runs normally -- this establishes the 'old plan' (Fe"
                      " particle found, ready for follow-up imaging) that the next turn must"
                      " abandon entirely, not just adjust."),
        },
        {
            "stage": "switch_task_mid_task",
            "user": ("Actually, never mind the particle search -- can you just do a quick plain"
                     " image scan of the same area at the current energy instead?"),
            "expected": {
                "forbidden_tools": ["load_intelligence_particles", "start_multiregion_scan"],
            },
            "notes": ("Mid-conversation instruction change: the user abandons the elemental-map/"
                      "follow-up-imaging plan entirely for a plain single-energy image scan."
                      " forbidden_tools catches the real failure mode -- continuing to image the"
                      " already-found Fe particle as if the user hadn't just changed the request."
                      " No must_call/param_checks: there's no single correct energy for 'the"
                      " current energy', and asking for clarification vs. just running a plain"
                      " scan are both acceptable; the only violation is persisting with the"
                      " abandoned two-energy/particle-imaging plan."),
        },
    ],
}


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


def main() -> None:
    tuples = [json.loads(l) for l in TUPLES_PATH.read_text().splitlines() if l.strip()]
    tool_units = _tool_unit_cases()
    with open(TASKS_PATH, "w") as out:
        for case in tool_units:
            out.write(json.dumps(case) + "\n")
        for t in tuples:
            task_name = t["task"]
            if task_name not in SCENARIOS:
                raise ValueError(f"No SCENARIOS entry for task {task_name!r} (tuples.jsonl id={t.get('id')})")
            scenario = SCENARIOS[task_name]
            task: dict = {
                "task": task_name,
                "intent": scenario["intent"],
                "particle": scenario.get("particle", _DEFAULT_PARTICLE),
                "plan_expected": scenario.get("plan_expected", {}),
                "stages": scenario["stages"],
            }
            overrides = scenario.get("mock_overrides")
            if overrides:
                task["mock_overrides"] = overrides
            out.write(json.dumps(task) + "\n")
    print(f"Wrote {len(tool_units)} tool-unit case(s) + {len(tuples)} task(s) → {TASKS_PATH}")


if __name__ == "__main__":
    main()
