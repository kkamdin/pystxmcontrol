"""
Step 4: Convert tuples into concrete synthetic inputs for AgentInterface.dispatch().

Reads tuples.jsonl, constructs a realistic (anomaly_dict, recent_events) pair
for each tuple, and writes the results to inputs.jsonl.

Anomaly dict field names match what AnomalyDetector actually produces.
Threshold values are read from config so generated values stay consistent
with whatever is deployed.

Usage:
    .venv/bin/python evals/intelligence_agent/build_inputs.py
"""

import json
import sys
import time
from pathlib import Path

from pystxmcontrol.controller.intelligence import (
    _CRITICAL_FOCUS_DECLINE_MULTIPLIER,
    _CRITICAL_ZSCORE_MULTIPLIER,
    AnomalyDetector,
)

CONFIG_PATH = Path(sys.prefix) / "pystxmcontrol_cfg/main.json"
TUPLES_PATH = Path(__file__).parent / "tuples.jsonl"
INPUTS_PATH = Path(__file__).parent / "inputs.jsonl"
INPUTS_META_PATH = Path(__file__).parent / "inputs_meta.json"

# Fixed base timestamp so inputs are reproducible across runs.
_T0 = 1_000_000.0


# ---------------------------------------------------------------------------
# Event builders — field names match what server.py / controller.py record
# ---------------------------------------------------------------------------

def _event(t_offset: float, event_type: str, **kwargs) -> dict:
    return {"type": event_type, "timestamp": _T0 + t_offset, **kwargs}


def _scan_start(t: float) -> dict:
    return _event(t, "scan_start", scan_type="Image", scan_id="TEST_EVAL_001")


def _region_complete(t: float, region: int = 0) -> dict:
    return _event(
        t, "region_complete",
        scan_type="Image", region=str(region), energy_index=0,
        frame_mean=7200.0, frame_min=400.0, frame_max=14000.0,
        frame_std=2100.0, focus_score=0.82,
    )


# Per-tuple event lists. Keys match tuple IDs in tuples.jsonl.
_EVENTS: dict[int, list[dict]] = {
    1:  [],
    2:  [_scan_start(0),
         _event(45, "shutter_changed", mode="close", during_scan=True)],
    4:  [_scan_start(0),
         _event(30, "motor_moved", motor="SampleX", target=0.5, actual=0.49, during_scan=False)],
    13: [_scan_start(0), _region_complete(60)],
    15: [_scan_start(0),
         _event(20, "motor_moved", motor="Energy", target=850.0, actual=850.0, during_scan=False)],
    16: [_scan_start(0),
         _event(20, "scan_paused"),
         _event(22, "shutter_changed", mode="close", during_scan=True),
         _event(55, "scan_resumed")],
    20: [_scan_start(0),
         _event(20, "motor_moved", motor="ZonePlateZ", target=-2.5, actual=-2.5, during_scan=False),
         _event(35, "motor_moved", motor="SampleZ", target=0.15, actual=0.15, during_scan=False)],
}


# ---------------------------------------------------------------------------
# Anomaly dict builders — field names match AnomalyDetector output exactly
# ---------------------------------------------------------------------------

def _anomaly(anomaly_type: str, severity: str, detector: AnomalyDetector) -> dict:
    """Build a value that just clears the detector's (critical-only) firing bar.

    All tuples are severity="critical" — the detector no longer has a "warn"
    tier — so these always compute a value a bit past the real critical bar
    rather than a separate synthetic edge case.
    """
    if anomaly_type == "intensity_drop":
        critical_z = detector.zscore_threshold * _CRITICAL_ZSCORE_MULTIPLIER
        z = -(critical_z + 0.5)
        baseline_mean = 8500.0
        baseline_std = 800.0
        line_mean = baseline_mean + z * baseline_std
        return {
            "type": "intensity_drop",
            "severity": severity,
            "line_mean": round(line_mean, 4),
            "baseline_mean": round(baseline_mean, 4),
            "baseline_std": round(baseline_std, 4),
            "z_score": round(z, 2),
        }

    elif anomaly_type == "focus_decline":
        critical_pct = detector.focus_decline_pct * _CRITICAL_FOCUS_DECLINE_MULTIPLIER
        prev_focus = 0.82
        pct = -(critical_pct + 0.5)
        focus_score = prev_focus * (1 + pct / 100.0)
        return {
            "type": "focus_decline",
            "severity": severity,
            "focus_score": round(focus_score, 4),
            "prev_focus_score": round(prev_focus, 4),
            "pct_change": round(pct, 1),
        }

    raise ValueError(f"Unknown anomaly_type: {anomaly_type}")


def main() -> None:
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    detector = AnomalyDetector(config.get("intelligence", {}).get("anomaly", {}))

    tuples = []
    with open(TUPLES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                tuples.append(json.loads(line))

    missing = {t["id"] for t in tuples} - set(_EVENTS)
    if missing:
        raise ValueError(f"No _EVENTS entry for tuple IDs: {sorted(missing)}")

    anomaly_cfg = config.get("intelligence", {}).get("anomaly", {})
    inputs_meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_tuples": len(tuples),
        "critical_zscore_multiplier": _CRITICAL_ZSCORE_MULTIPLIER,
        "critical_focus_decline_multiplier": _CRITICAL_FOCUS_DECLINE_MULTIPLIER,
        "anomaly_config": anomaly_cfg,
    }
    INPUTS_META_PATH.write_text(json.dumps(inputs_meta, indent=2))

    with open(INPUTS_PATH, "w") as out:
        for t in tuples:
            entry = {
                "id": t["id"],
                "tuple": t,
                "anomaly": _anomaly(t["anomaly_type"], t["severity"], detector),
                "recent_events": _EVENTS[t["id"]],
            }
            out.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(tuples)} inputs → {INPUTS_PATH}")
    print(f"Config snapshot → {INPUTS_META_PATH}")


if __name__ == "__main__":
    main()
