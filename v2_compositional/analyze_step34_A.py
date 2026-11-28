# -*- coding: utf-8 -*-
"""Paired analysis and hard gates for the step34 A experiment matrix.

The three arms must share the same seeds:

* B_budget: conservation layer with no pressure term;
* C_pressure: conservation plus measured optimization pressure;
* D_frozen: pressure is observed, but the allocation is frozen (trust=1).

Lower validation loss is better.  The primary contrast is paired C - B.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from bio_update import validate_trace


ARMS = ("B_budget", "C_pressure", "D_frozen")
_ARM_FROM_FILE = {
    "b_budget": "B_budget",
    "c_pressure": "C_pressure",
    "d_frozen": "D_frozen",
}

# Two-sided 95% Student-t critical values, indexed by degrees of freedom.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("empty sample")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _paired_summary(deltas: list[float]) -> dict[str, Any]:
    out = _summary(deltas)
    n = len(deltas)
    sd = float(out["sd"])
    if n > 1:
        critical = _T95.get(n - 1, 1.96)
        radius = critical * sd / math.sqrt(n)
        ci_low: float | None = float(out["mean"]) - radius
        ci_high: float | None = float(out["mean"]) + radius
    else:
        ci_low = ci_high = None
    mean = float(out["mean"])
    out.update(
        ci95_low=ci_low,
        ci95_high=ci_high,
        wins=sum(x < 0.0 for x in deltas),
        ties=sum(x == 0.0 for x in deltas),
        losses=sum(x > 0.0 for x in deltas),
        deltas=deltas,
    )
    return out


def _load_matrix(outdir: Path, prefix: str) -> dict[str, dict[int, dict[str, Any]]]:
    pattern = re.compile(
        rf"^{re.escape(prefix)}_(b_budget|c_pressure|d_frozen)_s(\d+)\.json$"
    )
    matrix: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for path in sorted(outdir.glob(f"{prefix}_*.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        arm = _ARM_FROM_FILE[match.group(1)]
        seed = int(match.group(2))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_result_path"] = str(path)
        payload["_trace_path"] = str(path.with_name(path.stem + "_bio.jsonl"))
        matrix[arm][seed] = payload
    return matrix


def analyze(outdir: Path, prefix: str, minimum_active_std: float) -> dict[str, Any]:
    matrix = _load_matrix(outdir, prefix)
    seed_sets = {arm: set(matrix[arm]) for arm in ARMS}
    errors: list[str] = []
    if any(not seeds for seeds in seed_sets.values()):
        errors.append(f"missing arm results: {seed_sets}")
    common = set.intersection(*seed_sets.values()) if all(seed_sets.values()) else set()
    if any(seeds != common for seeds in seed_sets.values()):
        errors.append(f"unpaired seeds: {seed_sets}")

    trace_checks: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for seed, result in sorted(matrix[arm].items()):
            label = f"{arm}/s{seed}"
            if not bool(result.get("telemetry_validation", {}).get("pass")):
                errors.append(f"{label}: telemetry failed")
            for key in ("val_loss", "ppl", "avg_step_ms"):
                value = float(result.get(key, math.nan))
                if not math.isfinite(value):
                    errors.append(f"{label}: non-finite {key}")
            verdict = validate_trace(
                result["_trace_path"],
                require_active=(arm == "C_pressure"),
                minimum_scale_std=minimum_active_std,
                require_resync=True,
            )
            trace_checks[label] = verdict
            if not verdict["ok"]:
                errors.extend(f"{label}: {msg}" for msg in verdict["errors"])

    arm_summary: dict[str, Any] = {}
    for arm in ARMS:
        rows = [matrix[arm][seed] for seed in sorted(common)]
        if rows:
            arm_summary[arm] = {
                "val_loss": _summary([float(row["val_loss"]) for row in rows]),
                "ppl": _summary([float(row["ppl"]) for row in rows]),
                "avg_step_ms": _summary([float(row["avg_step_ms"]) for row in rows]),
                "max_scale_std": _summary([
                    float(row["bio_update_summary"]["max_scale_std"]) for row in rows
                ]),
            }

    c_minus_b = [
        float(matrix["C_pressure"][seed]["val_loss"])
        - float(matrix["B_budget"][seed]["val_loss"])
        for seed in sorted(common)
    ]
    b_minus_d = [
        float(matrix["B_budget"][seed]["val_loss"])
        - float(matrix["D_frozen"][seed]["val_loss"])
        for seed in sorted(common)
    ]
    if b_minus_d and max(abs(x) for x in b_minus_d) > 1e-12:
        errors.append("B_budget and D_frozen diverged; frozen-control equivalence failed")

    paired = _paired_summary(c_minus_b) if c_minus_b else {}
    conclusion = "incomplete"
    if paired and paired["ci95_low"] is not None:
        if float(paired["ci95_high"]) < 0.0:
            conclusion = "C_pressure_better"
        elif float(paired["ci95_low"]) > 0.0:
            conclusion = "C_pressure_worse"
        else:
            conclusion = "inconclusive_ci_crosses_zero"

    return {
        "ok": not errors,
        "outdir": str(outdir),
        "prefix": prefix,
        "seeds": sorted(common),
        "arms": arm_summary,
        "paired_C_minus_B_val_loss": paired,
        "frozen_control_B_minus_D": {
            "deltas": b_minus_d,
            "max_abs_gap": max((abs(x) for x in b_minus_d), default=None),
        },
        "trace_checks": trace_checks,
        "conclusion": conclusion,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--prefix", default="st34local")
    parser.add_argument("--minimum-active-std", type=float, default=0.005)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    report = analyze(args.outdir, args.prefix, args.minimum_active_std)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(text)
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 7


if __name__ == "__main__":
    raise SystemExit(main())
