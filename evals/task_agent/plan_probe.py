"""
Eval-only "propose_plan" probe.

Injects one extra tool into the LLM call so the agent has a STRUCTURED way to declare
concrete scan parameters during a plan-proposal turn, instead of relying on free prose
(which run_assertions.py had to regex-scrape, unreliably -- see README's "Plan-stage
scoring" for the false-fail bugs that caused) or a system-prompt instruction asking for a
fenced JSON block (measured at 0/7 compliance -- models are far more reliable at native
tool-calling than at following a "please format your text like this" instruction).

Nothing in pystxmcontrol/controller/task_agent/{agent,tools}.py changes. This is purely an
eval-harness technique: monkeypatch the LLM call to add one extra tool (same pattern
run_eval.py already uses for _TokenMeter / _install_iter_tracker), and give the mock server
a canned response for it. Reusable against any other agent under eval the same way, without
touching that agent's production source.

Usage (see run_eval.py):
    from plan_probe import PROPOSE_PLAN_SCHEMA, install as install_plan_probe
    install_plan_probe(agent)
    # mock server: add PROPOSE_PLAN_SCHEMA["function"]["name"] -> canned response
    # run_assertions.py: extend its tool schema dict with PROPOSE_PLAN_SCHEMA
"""

from pystxmcontrol.controller.task_agent.tools import TOOL_SCHEMAS

# Mirror update_scan's real parameter set exactly, so the two can never drift apart --
# copied at import time from the actual production schema, not hand-duplicated.
_UPDATE_SCAN_PROPERTIES = next(
    t["function"]["parameters"]["properties"] for t in TOOL_SCHEMAS
    if t["function"]["name"] == "update_scan"
)

PROPOSE_PLAN_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "propose_plan",
        "description": (
            "Declare the concrete scan parameters you intend to use, without configuring or "
            "starting anything. Call this whenever you state specific numbers to the user "
            "before they've confirmed the plan -- including your very first response to a new "
            "request. Use the exact same field names as update_scan. Omit fields you haven't "
            "decided yet. Do NOT call this once you're actually ready to configure the scan -- "
            "use update_scan() for that; propose_plan() has no effect on the pending scan "
            "definition. If you're asking a clarifying question instead of proposing concrete "
            "parameters, don't call this at all."
        ),
        "parameters": {
            "type": "object",
            "properties": dict(_UPDATE_SCAN_PROPERTIES),
            "required": [],
        },
    },
}

PROPOSE_PLAN_RESULT_PREFIX = "Plan noted (not configured or started): "


def install(agent) -> None:
    """Wrap agent._llm.chat.completions.create so every call also offers propose_plan.

    Stacks cleanly with run_eval.py's other create() wrappers (_TokenMeter.install,
    _install_iter_tracker) regardless of install order -- each just forwards *a, **k to the
    next layer, so mutating the `tools` kwarg here is visible all the way down to the real
    (or mocked-transport) API call.
    """
    create = agent._llm.chat.completions.create

    def wrapped(*a, **k):
        if "tools" in k:
            k = dict(k)
            k["tools"] = list(k["tools"]) + [PROPOSE_PLAN_SCHEMA]
        return create(*a, **k)

    agent._llm.chat.completions.create = wrapped
