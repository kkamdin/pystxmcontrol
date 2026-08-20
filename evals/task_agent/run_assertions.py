"""
Step 3: Score the recorded tool calls against each task stage's expectations.

Every assertion is a deterministic check on the agent's tool calls — no LLM judge.
The headline check is **dispatchable**: the (tool, parameters) the agent produced could
actually be executed by the production ToolSet (right name, known parameters, valid types,
and for update_scan the real ToolSet.update_scan accepts them).

Assertions (True / False / None=N/A):
  well_formed          - at least one tool call, arguments parse as JSON
  right_approach       - a chosen tool is in expected.must_call
  avoids_wrong         - no chosen tool is in expected.forbidden_tools
  params_match         - update_scan params match expected.param_checks
  dispatchable         - every tool exists, args match its schema, update_scan args accepted
                         by a real offline ToolSet (a no-args update_scan() call -- the tool's
                         documented "inspect current config" mode -- counts as accepted too)

param_checks types (evaluated in _params_match_strategy):
  energy_list_len: N          - update_scan energy_list has exactly N entries
  energy_in_range: [lo, hi]   - all implied energies fall within [lo, hi]
  if_update_scan_center_near  - update_scan x/y_center within tol of target (N/A if not called)
  max_calls: {tool: N}        - tool called at most N times (catches blind-retry loops)
  max_total_calls: N          - total tool calls in the stage <= N
  scan_type_param: str        - any update_scan call has scan_type == str
  no_parallel_start: true     - update_scan and start_multiregion_scan are in different LLM
                                iterations (i.e. agent saw update_scan result before firing
                                start_multiregion_scan; requires llm_iter field from run_eval)
  energy_spectrum_min_points: N - update_scan uses >= N energy points (energy_points field or
                                len(energy_list)), for distinguishing a spectrum from a 2-point map
  update_scan_nonempty: true  - update_scan called with at least one non-empty-args call
                                (catches update_scan({}) confusion from stale server state)
  scan_type_valid: true       - any update_scan scan_type is one of the instrument's real
                                configured types, not a colloquial/invalid guess (e.g.
                                "z-stack") -- the mock always accepts update_scan regardless
                                of scan_type, so this checks the AGENT avoided an invalid
                                type rather than relying on (mocked-away) server rejection

Usage:
    python evals/task_agent/run_assertions.py                 # latest run
    python evals/task_agent/run_assertions.py --run-id all --output res.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

from pystxmcontrol.controller.task_agent.tools import TOOL_SCHEMAS, ToolSet

import plan_probe

CFGDIR = Path(sys.prefix) / "pystxmcontrol_cfg"
TRACES_PATH = Path(__file__).parent / "traces.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.jsonl"

# propose_plan is an eval-only probe tool (see plan_probe.py), not part of the real
# production TOOL_SCHEMAS -- add it here so dispatchable/schema checks recognize it.
_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOL_SCHEMAS}
_SCHEMA[plan_probe.PROPOSE_PLAN_SCHEMA["function"]["name"]] = \
    plan_probe.PROPOSE_PLAN_SCHEMA["function"]["parameters"]
_TOL = 1e-6

_PYTYPE = {"number": (int, float), "integer": int, "string": str,
           "array": list, "boolean": bool, "object": dict}

# Must match the mock server's get_config() response in run_eval.py -- the real ToolSet
# rejects a colloquial/invalid scan_type (e.g. "z-stack") at update_scan() time, but the
# mock accepts any scan_type unconditionally, so scan_type_valid below is the only signal
# that the agent itself avoided an invalid guess rather than relying on server rejection.
_VALID_SCAN_TYPES = {"Image", "Focus", "Line Spectrum", "Single Motor", "Double Motor"}

# ---------------------------------------------------------------------------
# Plan-stage scoring: plan_proposal is free text, not a tool call, so it can't be
# scored against expected.param_checks the way an action stage's real update_scan()
# call is. run_eval.py injects an eval-only "propose_plan" tool (see plan_probe.py) the
# agent can call to declare parameters without executing anything -- that call, when
# present, is the primary source of truth, reused directly by the SAME
# _params_match_strategy() used for action stages. A fenced ```json block in prose
# (_plan_block()) is kept as a defensive fallback in case a model emits that format
# unprompted. The energy-regex below is now only a coarse fallback trigger ("did the
# prose commit to concrete numbers at all?") used to decide whether the agent skipped
# structured output entirely -- see _plan_signal(). No LLM call, no network
# dependency -- purely deterministic, same as every other check in this file.
# ---------------------------------------------------------------------------

_PLAN_BLOCK_RE = re.compile(r'```(?:json|plan|proposed_scan)?\s*\n(.*?)```', re.DOTALL | re.IGNORECASE)
_SCAN_PARAM_KEYS = set(_SCHEMA.get("update_scan", {}).get("properties", {}))


def _plan_block(text: str) -> dict | None:
    """Extract the agent's structured plan-summary block, if present: a fenced code block
    that parses as JSON and whose keys overlap update_scan's real parameters (disambiguates
    a genuine plan block from an unrelated fenced snippet the model might include)."""
    for m in _PLAN_BLOCK_RE.finditer(text or ""):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and _SCAN_PARAM_KEYS.intersection(data):
            return data
    return None


_ENERGY_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(keV|eV)\b', re.IGNORECASE)
# Words/symbols that, if they appear shortly before a matched number, mean it's a current
# reading, a safety/move threshold, or a step size/resolution rather than a proposed scan
# energy (e.g. "currently at 600 eV", "by >100 eV", "a 1 eV resolution", "0.5 eV steps").
# Best-effort local-context filter, not full comprehension.
_ENERGY_EXCLUDE_CONTEXT_RE = re.compile(
    r'current(ly)?|existing|more than|at least|greater than|exceeds|>'
    r'|resolution|step|spacing|increment|interval', re.IGNORECASE)
_ENERGY_CONTEXT_WINDOW = 30


def _energies_from_text(text: str) -> list[float]:
    """Unit-tagged energies proposed in free text, normalized to eV. Skips numbers whose
    local context marks them as a current reading or a threshold rather than a proposal.
    Checks both sides of the match: "currently at 600 eV" excludes via the prefix, while
    "a 1 eV resolution" / "0.5 eV steps" excludes via the suffix -- the disqualifying word
    describing a threshold usually leads, but one describing step size/resolution trails."""
    text = text or ""
    out = []
    for m in _ENERGY_RE.finditer(text):
        prefix = text[max(0, m.start() - _ENERGY_CONTEXT_WINDOW):m.start()]
        suffix = text[m.end():m.end() + _ENERGY_CONTEXT_WINDOW]
        if _ENERGY_EXCLUDE_CONTEXT_RE.search(prefix) or _ENERGY_EXCLUDE_CONTEXT_RE.search(suffix):
            continue
        v = float(m.group(1))
        out.append(v * 1000 if m.group(2).lower() == "kev" else v)
    return out


def _real_call(calls: list[dict], name: str) -> dict:
    """Merge every real (non-empty) call to `name` in this call list, in order -- later
    calls override earlier keys on conflict, matching production's own update_scan
    semantics (each real call merges onto the in-memory pending state:
    `{**self._scan, **kwargs}`). A call with empty arguments -- update_scan()'s documented
    no-args "inspect current config" mode -- contributes nothing and is skipped, rather
    than winning by virtue of being first. Previously this returned only the FIRST
    matching call's arguments, so a legitimate peek-then-configure turn (update_scan() to
    check state, then update_scan(...) with the real values) was checked against the
    peek's empty args instead of the real configuration that followed it. Always returns a
    dict, defaulting to {} when no real call exists -- callers that need a None sentinel
    for "wasn't really called" do `or None` at the call site.

    Takes a `calls` list rather than a trace so callers can scope the merge to one scan
    episode (see _first_scan_episode) instead of the whole turn -- merging update_scan
    calls from a SECOND, later, legitimate scan (e.g. a single-energy zoom-in confirmatory
    scan after a two-energy elemental map) into the first scan's checked state broke
    energy_list_len/energy_in_range/center_near for turns that correctly run more than one
    scan."""
    merged = {}
    for c in calls:
        if c["name"] != name or not isinstance(c["arguments"], dict) or not c["arguments"]:
            continue
        merged.update(c["arguments"])
    return merged


def _first_scan_episode(trace: dict) -> list[dict]:
    """Tool calls up to and including the first start_scan/start_multiregion_scan -- the
    episode that action-stage param_checks (energy_list_len, energy_in_range,
    if_update_scan_center_near) are meant to validate. A turn is free to run a second,
    later scan (a confirmatory zoom-in, a follow-up map) without that scan's update_scan
    args being merged into the first scan's checked state."""
    calls = trace.get("tool_calls") or []
    out = []
    for c in calls:
        out.append(c)
        if c["name"] in ("start_scan", "start_multiregion_scan"):
            break
    return out


def _has_energies(d: dict) -> bool:
    return bool(d.get("energy_list")) or d.get("energy_start") is not None or d.get("energy_stop") is not None


def _plan_signal(trace: dict) -> tuple[dict | None, bool | None]:
    """Resolve the plan stage's synthetic update_scan-shaped dict AND its format-compliance
    verdict together in one pass, so the two can never drift out of lockstep (they used to be
    two separately-maintained functions with parallel branching -- easy to update one and
    forget the other). Priority order for the dict:
    1. a real speculative update_scan() call the model made this turn (energy_list OR
       energy_start/energy_stop -- both are valid ToolSet inputs, see _energies_of()) --
       format_ok is N/A here, there's nothing to require structured output for.
    2. a real propose_plan() call (see plan_probe.py) -- an eval-injected tool the agent can
       use to declare parameters without executing anything -- merged onto any partial real
       update_scan() fields (real wins on conflicts, since it reflects what's actually staged).
    3. the agent's own structured ```json plan block in free text (see _plan_block()) --
       kept as a defensive fallback in case a model emits this format unprompted.
    4. legacy regex energy-scraping, only as a last resort when none of the above exists --
       used ONLY to populate `upd` for any param_checks that apply to this stage (e.g.
       energy_in_range on a plan-stage task). format_ok is always None (N/A) here, never
       False: not calling propose_plan/update_scan is equally consistent with the agent
       correctly asking a clarifying question or explaining a refusal by citing the
       instrument's current/general state (e.g. "Energy (600.0 eV)" read back from
       get_config(), or "photon energies typically between 100-2000 eV" describing the
       beamline's range) -- neither is a "plan" at all, so there's nothing to penalize as
       badly formatted. The regex can't reliably tell that apart from a genuine sloppy
       commitment to concrete numbers in prose, so it no longer tries; right_approach,
       avoids_wrong, and dispatchable already independently cover whether the agent's
       actual behavior was correct.

    Falling back to regex whenever a real call/block IS present would let the regex's
    best-effort extraction silently overwrite known-good structured data with whatever stray
    numbers it finds in the surrounding text (see the "1 eV resolution" / "0.5 eV steps" bugs
    this replaced).
    """
    real = _real_call(trace.get("tool_calls") or [], "update_scan")
    text = trace.get("final_text") or ""
    upd = dict(real)
    if _has_energies(real):
        return upd or None, None

    proposed = _real_call(trace.get("tool_calls") or [], "propose_plan")
    if proposed:
        for k, v in proposed.items():
            upd.setdefault(k, v)
        return upd or None, True

    block = _plan_block(text)
    if block:
        for k, v in block.items():
            upd.setdefault(k, v)
        return upd or None, True

    energies = _energies_from_text(text)
    if energies:
        upd["energy_list"] = energies
    return upd or None, None


def _offline_toolset():
    """A ToolSet seeded from the installed config — for real update_scan validation."""
    main_cfg = json.loads((CFGDIR / "main.json").read_text())

    class _Client:
        motorInfo = json.loads((CFGDIR / "motor.json").read_text())
        scanConfig = json.loads((CFGDIR / "scan.json").read_text())
        currentMotorPositions = {}
        main_config = main_cfg
    return ToolSet(_Client())


def _args_match_schema(name, args):
    """Generic JSON-schema-lite check: known tool, known keys, roughly-correct types."""
    if name not in _SCHEMA:
        return False, f"unknown tool '{name}'"
    if not isinstance(args, dict):
        return False, "arguments did not parse to an object"
    props = _SCHEMA[name].get("properties", {})
    for k, v in args.items():
        if k not in props:
            return False, f"unknown parameter '{k}'"
        jtype = props[k].get("type")
        pytype = _PYTYPE.get(jtype)
        # bool is a subclass of int in Python, so isinstance(v, pytype) alone would accept a
        # JSON boolean for a "number"/"integer" field -- exclude that explicitly.
        wrong_type = pytype and (not isinstance(v, pytype)
                                  or (jtype in ("number", "integer") and isinstance(v, bool)))
        if wrong_type:
            return False, f"param '{k}' expected {jtype}, got {type(v).__name__}"
    return True, None


def _dispatchable(name, args, toolset):
    ok, reason = _args_match_schema(name, args)
    if not ok:
        return False, reason
    if name == "update_scan":
        try:
            res = toolset.update_scan(**args)
        except Exception as e:
            return False, f"update_scan raised {e!r}"
        # update_scan's own tool-schema description explicitly documents a no-args "inspect
        # the current config" mode, which returns "Current scan definition: ..." instead of
        # "Scan updated: ...". That's a real, prompt-sanctioned success path, not a failure --
        # accept either prefix rather than only recognizing the with-args reply.
        if not (res.startswith("Scan updated") or res.startswith("Current scan definition")):
            return False, res.splitlines()[0][:120]
    return True, None


def _energies_of(upd):
    """All energies implied by an update_scan call: explicit list or start/stop endpoints."""
    el = upd.get("energy_list")
    if isinstance(el, list) and el:
        return [float(e) for e in el]
    es = [upd.get("energy_start"), upd.get("energy_stop")]
    return [float(e) for e in es if isinstance(e, (int, float))]


def _params_match_strategy(checks, upd, names, calls):
    """Deterministic strategy checks, evaluated individually.

    Returns (overall, detail) where:
      overall : True / False / None(N/A) — AND of all applicable sub-checks
      detail  : {sub_check_name: True|False|None} for diagnosis
    `upd`   is the arguments dict of the first update_scan call (or None).
    `names` is the ordered list of tool names called this stage.
    `calls` is the full list of {"name", "arguments", "llm_iter"} records.
    """
    if not checks:
        return None, {}
    detail = {}

    # Collect all update_scan calls for checks that need them all (not just the first).
    all_upd_calls = [c for c in calls if c["name"] == "update_scan"
                     and isinstance(c.get("arguments"), dict)]

    for name, spec in checks.items():
        if name == "max_calls":
            for tool, cap in spec.items():
                detail[f"max_calls:{tool}"] = names.count(tool) <= cap

        elif name == "max_total_calls":
            detail["max_total_calls"] = len(names) <= spec

        elif name == "energy_list_len":
            # N/A (not fail) when update_scan wasn't called this turn at all -- e.g. it was
            # already configured during the plan proposal and this turn just calls start_scan.
            # Consistent with energy_in_range's None-means-N/A convention below.
            el = (upd or {}).get("energy_list")
            detail["energy_list_len"] = (len(el) == spec) if isinstance(el, list) and el else None

        elif name == "energy_in_range":
            # N/A (not fail) when no energies are found at all -- e.g. a plan-stage trace
            # whose text states no concrete numbers yet. Consistent with every other
            # sub-check's None-means-N/A convention (see if_update_scan_center_near below).
            lo, hi = spec
            energies = _energies_of(upd) if upd else []
            detail["energy_in_range"] = (all(lo <= e <= hi for e in energies) if energies else None)

        elif name == "if_update_scan_center_near":
            # N/A when the agent positioned via load_intelligence_particles instead of
            # update_scan -- this comment described that exception for a while, but the code
            # never actually checked `names` for it, so a well-behaved agent that positioned
            # the follow-up scan via load_intelligence_particles() and only touched
            # energy/dwell in its update_scan call was scored a false center_near failure
            # (cx/cy were simply absent from that particular call, not wrong).
            if upd is None or "load_intelligence_particles" in names:
                detail["center_near"] = None
            else:
                cx, cy = upd.get("x_center"), upd.get("y_center")
                detail["center_near"] = bool(
                    isinstance(cx, (int, float)) and isinstance(cy, (int, float))
                    and abs(cx - spec["x"]) <= spec["tol"] and abs(cy - spec["y"]) <= spec["tol"])

        elif name == "scan_type_param":
            # Any update_scan call must include scan_type == spec.
            # Checks all update_scan calls because the agent may call update_scan multiple times
            # (once for spatial params, once for energy params); either may set scan_type.
            all_types = [c["arguments"].get("scan_type") for c in all_upd_calls]
            matching = [t for t in all_types if t is not None]
            if not matching:
                detail["scan_type_param"] = None  # N/A: update_scan never specified scan_type
            else:
                detail["scan_type_param"] = any(t == spec for t in matching)

        elif name == "no_parallel_start":
            # update_scan and start_multiregion_scan must be in different LLM iterations.
            # Same llm_iter means the LLM issued both in one completion — the agent fired
            # start_multiregion_scan before seeing the update_scan result.
            us_iters = {c.get("llm_iter") for c in calls if c["name"] == "update_scan"
                        and c.get("llm_iter") is not None}
            ms_iters = {c.get("llm_iter") for c in calls if c["name"] == "start_multiregion_scan"
                        and c.get("llm_iter") is not None}
            if not us_iters or not ms_iters:
                detail["no_parallel_start"] = None  # N/A if either tool wasn't called
            else:
                detail["no_parallel_start"] = us_iters.isdisjoint(ms_iters)

        elif name == "energy_spectrum_min_points":
            # Any update_scan call must use >= spec energy points (energy_points or len(energy_list)).
            # Distinguishes a true line spectrum from a two-point elemental map.
            max_pts = 0
            for c in all_upd_calls:
                a = c["arguments"]
                pts = a.get("energy_points")
                if isinstance(pts, int) and pts > 0:
                    max_pts = max(max_pts, pts)
                el = a.get("energy_list")
                if isinstance(el, list):
                    max_pts = max(max_pts, len(el))
            detail["energy_spectrum_min_points"] = (max_pts >= spec) if max_pts > 0 else None

        elif name == "update_scan_nonempty":
            # At least one update_scan call must have non-empty args.
            # update_scan({}) indicates the agent is confused by stale server state.
            if not all_upd_calls:
                detail["update_scan_nonempty"] = None  # N/A if update_scan not called
            else:
                detail["update_scan_nonempty"] = any(bool(c["arguments"]) for c in all_upd_calls)

        elif name == "scan_type_valid":
            # Any update_scan call's scan_type must be a real configured type, not a
            # colloquial/invalid guess (e.g. "z-stack", "tomo").
            all_types = [c["arguments"].get("scan_type") for c in all_upd_calls
                         if c["arguments"].get("scan_type") is not None]
            detail["scan_type_valid"] = (all(t in _VALID_SCAN_TYPES for t in all_types)
                                          if all_types else None)

        else:
            detail[name] = None  # unknown check type -> N/A

    applicable = [v for v in detail.values() if v is not None]
    overall = all(applicable) if applicable else None
    return overall, detail


def evaluate(trace, toolset):
    """Score one task stage: did the agent's action implement the correct strategy?

    For a plan_proposal stage (trace["kind"] == "plan"), there is usually no update_scan
    call to check -- the model's output is free text. run_eval.py offers it an eval-only
    "propose_plan" tool (see plan_probe.py) to declare parameters structurally instead;
    _plan_signal() resolves both the format-compliance verdict and a synthetic
    update_scan-shaped dict from the propose_plan call (or, failing that, the free text) in
    one pass, so the same _params_match_strategy() sub-checks apply to it as to a real call.
    """
    calls = trace.get("tool_calls") or []
    exp = trace.get("expected", {})
    names = [c["name"] for c in calls]
    r = {}

    # agent.py's run() loop catches the LLM call's own exceptions internally and returns them
    # as ordinary final_text ("LLM call failed: {e}") rather than raising -- run_eval.py's
    # try/except around agent.run() never sees it, so trace["error"] stays None even when
    # every underlying API call failed outright (e.g. a model/provider incompatibility).
    # Catch that sentinel directly so a fully broken run doesn't silently score well_formed=True.
    llm_call_failed = (trace.get("final_text") or "").startswith("LLM call failed:")
    r["well_formed"] = (not trace.get("error")) and not llm_call_failed \
        and all(c.get("arg_parse_error") is None for c in calls)
    if not r["well_formed"]:
        for k in ("right_approach", "avoids_wrong", "plan_formatted", "params_match", "dispatchable"):
            r[k] = None
        return r

    r["right_approach"] = (any(n in exp["must_call"] for n in names) if exp.get("must_call") else None)

    r["avoids_wrong"] = (not any(n in exp["forbidden_tools"] for n in names)
                         if exp.get("forbidden_tools") else None)

    if trace.get("kind") == "plan":
        upd, r["plan_formatted"] = _plan_signal(trace)
    else:
        r["plan_formatted"] = None
        # Real update_scan wins; if the agent never actually configured anything but did
        # propose_plan() with concrete parameters, validate THOSE against param_checks instead
        # of leaving everything N/A -- a confident-but-wrong proposal (e.g. the wrong element's
        # edge) must still fail energy_in_range/energy_list_len, not get a free pass just because
        # must_call allowed propose_plan as an alternative to real execution.
        episode = _first_scan_episode(trace)
        upd = _real_call(episode, "update_scan") or None
        if upd is None:
            upd = _real_call(episode, "propose_plan") or None
    pms, pms_detail = _params_match_strategy(exp.get("param_checks"), upd, names, calls)
    r["params_match"] = pms
    r["params_detail"] = pms_detail

    r["dispatchable"] = (all(_dispatchable(c["name"], c["arguments"], toolset)[0] for c in calls)
                         if calls else None)
    return r


ORDER = ["well_formed", "plan_formatted", "right_approach", "avoids_wrong", "params_match", "dispatchable"]


def _score(run_id, traces, toolset, out_path):
    cw = 20
    header = f"{'task':22}  {'stage':26}  " + "  ".join(f"{n[:cw]:<{cw}}" for n in ORDER)
    print(f"Run: {run_id}  ({len(traces)} stage(s))\n{header}\n" + "-" * len(header))
    totals = {n: {"pass": 0, "fail": 0, "na": 0} for n in ORDER}
    with open(out_path, "a") as out:
        for tr in traces:
            res = evaluate(tr, toolset)
            n_calls = len(tr.get("tool_calls") or [])
            out.write(json.dumps({"run_id": run_id, "task": tr.get("task"), "stage": tr.get("stage"),
                                  "kind": tr.get("kind"), "model": tr.get("model"),
                                  "timestamp": tr.get("timestamp"),
                                  "tool_calls": tr.get("tool_calls"), "tool_call_total": n_calls,
                                  "assertions": res}) + "\n")
            cells = []
            for n in ORDER:
                v = res.get(n)
                key = "na" if v is None else ("pass" if v else "fail")
                totals[n][key] += 1
                cells.append(f"{('—' if v is None else 'PASS' if v else 'FAIL'):<{cw}}")
            failed_sub = [k for k, v in (res.get("params_detail") or {}).items() if v is False]
            suffix = f"  PMS✗: {', '.join(failed_sub)}" if failed_sub else ""
            print(f"{tr.get('task', '')[:22]:22}  {tr.get('stage', '')[:26]:26}  "
                  + "  ".join(cells) + f"  [{n_calls} calls]" + suffix)
    print("-" * len(header) + "\nAssertions:")
    for n in ORDER:
        t = totals[n]; app = t["pass"] + t["fail"]
        rate = (t["pass"] / app * 100) if app else 0
        print(f"  {n}: {t['pass']}/{app} ({rate:.0f}%)" + (f"  ({t['na']} N/A)" if t["na"] else ""))


_ABBR = {"well_formed": "WF", "plan_formatted": "PF", "right_approach": "RA",
         "avoids_wrong": "AW", "params_match": "PMS", "dispatchable": "DISP"}


def _grouped_summary(traces, toolset):
    """Aggregate pass rates across ALL traces, grouped by model and task."""
    groups = {}
    for tr in traces:
        res = evaluate(tr, toolset)
        key = (tr.get("model", "?"), tr.get("task", "?"))
        g = groups.setdefault(key, {"n": 0, **{a: [0, 0] for a in ORDER}})
        g["n"] += 1
        for a in ORDER:
            v = res.get(a)
            if v is None:
                continue
            g[a][1] += 1
            g[a][0] += 1 if v else 0

    w_m, w_t = 18, 24
    header = (f"{'model':{w_m}} {'task':{w_t}} {'n':>3}  "
              + "  ".join(f"{_ABBR[a]:>7}" for a in ORDER) + "   score")
    print("\n=== Grouped summary — all results, by model × task ===")
    print(header + "\n" + "-" * len(header))

    def cells_and_score(g):
        cells, tp, ta = [], 0, 0
        for a in ORDER:
            p, app = g[a]
            tp += p; ta += app
            cells.append(f"{(f'{p}/{app}' if app else '-'):>7}")
        score = f"{tp/ta*100:.0f}%" if ta else "-"
        return cells, score

    for model in sorted({m for m, _ in groups}):
        mt = {"n": 0, **{a: [0, 0] for a in ORDER}}
        for (m, t), g in sorted(groups.items()):
            if m != model:
                continue
            cells, score = cells_and_score(g)
            print(f"{m[:w_m]:{w_m}} {t[:w_t]:{w_t}} {g['n']:>3}  " + "  ".join(cells) + f"   {score:>5}")
            mt["n"] += g["n"]
            for a in ORDER:
                mt[a][0] += g[a][0]; mt[a][1] += g[a][1]
        cells, score = cells_and_score(mt)
        print(f"{('  → ' + model)[:w_m]:{w_m}} {'(all tasks)':{w_t}} {mt['n']:>3}  "
              + "  ".join(cells) + f"   {score:>5}")
    print("\nLegend: " + "  ".join(f"{v}={k}" for k, v in _ABBR.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="latest")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.run_id == "all" and not args.output:
        ap.error("--output is required with --run-id all")
    out_path = Path(args.output) if args.output else RESULTS_PATH

    traces = [json.loads(l) for l in TRACES_PATH.read_text().splitlines() if l.strip()]
    if not traces:
        print("No traces — run run_eval.py first."); return
    run_ids = sorted({t["run_id"] for t in traces})
    selected = run_ids if args.run_id == "all" else [run_ids[-1] if args.run_id == "latest" else args.run_id]

    toolset = _offline_toolset()
    for rid in selected:
        _score(rid, [t for t in traces if t["run_id"] == rid], toolset, out_path)

    _grouped_summary(traces, toolset)
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
