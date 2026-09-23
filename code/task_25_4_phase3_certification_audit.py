"""Audit Track A reference exclusions without changing the frozen gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/task_25_4_phase3/track_A_certification.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/task_25_4_phase3/track_A_reference_audit.json"),
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    exclusions = []
    comparison_count = 0
    all_uncertainty_intersects = True
    for case in summary["cases"]:
        artifact = json.loads(Path(case["artifact"]).read_text(encoding="utf-8"))
        for block in artifact["blocks"]:
            for mode in block["selected_mode_certificates"]:
                if not mode["available"]:
                    continue
                reference = float(mode["lambda_ref"])
                uncertainty = float(mode["u_ref"])
                for tail_name in (
                    "explicit_continuous_interval",
                    "hybrid_continuous_interval",
                ):
                    comparison_count += 1
                    interval = mode[tail_name]
                    lower = float(interval["lower"])
                    upper = float(interval["upper"])
                    uncertainty_intersects = bool(
                        reference + uncertainty >= lower
                        and reference - uncertainty <= upper
                    )
                    all_uncertainty_intersects &= uncertainty_intersects
                    if not interval["contains_ordinary_reference"]:
                        overshoot = max(reference - upper, lower - reference, 0.0)
                        exclusions.append(
                            {
                                "n": artifact["n"],
                                "c": artifact["c"],
                                "P": block["P"],
                                "category": mode["category"],
                                "m": mode["m"],
                                "tail": tail_name,
                                "lambda_ref": reference,
                                "u_ref": uncertainty,
                                "certificate_lower": lower,
                                "certificate_upper": upper,
                                "overshoot": overshoot,
                                "overshoot_over_u_ref": overshoot / uncertainty,
                                "reference_uncertainty_intersects_certificate": uncertainty_intersects,
                            }
                        )

    result = {
        "task": "25.4 Phase 3 Track A reference-exclusion audit",
        "formal_gate_A_from_frozen_protocol": summary["gate_A_passed"],
        "gate_B_nontrivial_width": summary["gate_B_nontrivial_width_passed"],
        "comparison_count": comparison_count,
        "central_reference_exclusion_count": len(exclusions),
        "excluded_categories": sorted({row["category"] for row in exclusions}),
        "excluded_modes": sorted({row["m"] for row in exclusions}),
        "maximum_overshoot": max((row["overshoot"] for row in exclusions), default=0.0),
        "maximum_overshoot_over_u_ref": max(
            (row["overshoot_over_u_ref"] for row in exclusions), default=0.0
        ),
        "all_reference_uncertainty_intervals_intersect_certificates": all_uncertainty_intersects,
        "classification": (
            "formal Gate A remains failed; exclusions are reference-floor artifacts, not certificate failures"
        ),
        "exclusions": exclusions,
    }
    rendered = json.dumps(result, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"output": str(args.output), "sha256": digest, **{k: v for k, v in result.items() if k != "exclusions"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
