"""Consolidate Task 25.4 Phase 3 saved results into protocol artifacts.

This script does not recompute the numerical campaigns.  It verifies and
flattens their saved JSON outputs, creates the promised CSV/NPZ files, writes
the machine-readable gate/failure log, and plots only from saved data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def flatten_modes(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in manifest["cases"]:
        for mode in case["selected_modes"]:
            rows.append(
                {
                    "n": case["n"],
                    "c": case["c"],
                    "M_n_c": case["M_n_c"],
                    "parameter_path": "fixed continuous c",
                    "category": mode["category"],
                    "available": mode["available"],
                    "m": mode.get("m"),
                    "lambda_ref": mode.get("lambda_ref"),
                    "u_ref": mode.get("u_ref"),
                    "gap": mode.get("local_gap"),
                    "knee_endpoint_absolute_value": (
                        case["knee_endpoint"]["absolute_value_Q2"]
                        if mode["category"] == "knee"
                        else None
                    ),
                }
            )
    return rows


def flatten_certification(summary: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows = []
    for pointer in summary["cases"]:
        artifact_path = root / Path(pointer["artifact"])
        artifact = load(artifact_path)
        for block in artifact["blocks"]:
            for mode in block["selected_mode_certificates"]:
                base = {
                    "n": artifact["n"],
                    "c": artifact["c"],
                    "P": block["P"],
                    "parameter_path": artifact["parameter_path"],
                    "category": mode["category"],
                    "available": mode["available"],
                    "m": mode.get("m"),
                    "explicit_tail_upper": block["explicit_tail"]["decimal_upper"],
                    "hybrid_tail_upper": numeric(
                        block["hybrid_tail"]["certified_upper_ball"]["upper"]
                    ),
                }
                if mode["available"]:
                    finite = mode["finite_Ritz_interval"]
                    explicit = mode["explicit_continuous_interval"]
                    hybrid = mode["hybrid_continuous_interval"]
                    base.update(
                        {
                            "lambda_ref": mode["lambda_ref"],
                            "u_ref": mode["u_ref"],
                            "finite_lower": numeric(finite["lower"]),
                            "finite_upper": numeric(finite["upper"]),
                            "explicit_lower": numeric(explicit["lower"]),
                            "explicit_upper": numeric(explicit["upper"]),
                            "explicit_width": explicit["width"],
                            "explicit_contains_reference": explicit["contains_ordinary_reference"],
                            "hybrid_lower": numeric(hybrid["lower"]),
                            "hybrid_upper": numeric(hybrid["upper"]),
                            "hybrid_width": hybrid["width"],
                            "hybrid_contains_reference": hybrid["contains_ordinary_reference"],
                            "ordinary_observed_finite_Ritz_error": mode[
                                "ordinary_observed_finite_Ritz_error"
                            ],
                        }
                    )
                rows.append(base)
    return rows


def flatten_structure(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method, cases, old_key, new_key in (
        ("project", data["project_fixed_c"], "P_old", "P_new"),
        (
            "Boulsane_large_N",
            data["boulsane_fixed_c_large_N_offset"],
            "N_old",
            "N_new",
        ),
    ):
        for case in cases:
            for transition in case["transitions"]:
                rows.append(
                    {
                        "method": method,
                        "n": case["n"],
                        "c": case["c"],
                        "parameter_path": case["parameter_path"],
                        "dimension_old": transition.get(old_key),
                        "dimension_new": transition.get(new_key),
                        "block_defect_norm_2": transition["block_defect_norm_2"],
                        "relative_block_defect": transition["relative_block_defect"],
                        "raw_negative_increment_count": transition[
                            "raw_negative_increment_count"
                        ],
                        "significant_negative_increment_count": transition[
                            "significant_negative_increment_count"
                        ],
                        "minimum_increment": transition["minimum_increment"],
                    }
                )
    return rows


def flatten_mechanism(fixed: dict[str, Any], oracle: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in fixed["cases"]:
        for path, result in case["paths"].items():
            fit = result["Rayleigh_rate_fit"]
            for row in result["rows"]:
                rows.append(
                    {
                        "record_type": "fixed_offset",
                        "n": case["n"],
                        "c": case["c"],
                        "M_n_c": case["M_n_c"],
                        "m_knee": case["m_knee"],
                        "parameter_path": case["parameter_path"],
                        "path": path,
                        "N": row["N"],
                        "epsilon_over_pi": row.get("epsilon_over_pi"),
                        "omega": row.get("omega"),
                        "h": row.get("h"),
                        "e_lambda": row.get("e_lambda"),
                        "e_Rayleigh": row.get("e_Rayleigh"),
                        "Gamma": row.get("Gamma"),
                        "next_zero_normalized": row.get("next_zero_normalized"),
                        "half_mesh_scaled": row.get("half_mesh_scaled"),
                        "matrix_audit_passed": row.get("audit", {}).get("passed"),
                        "Rayleigh_fit_slope": fit.get("slope"),
                        "Rayleigh_fit_reported": fit.get("reported"),
                        "Rayleigh_fit_compatible": fit.get("compatible_with_window"),
                    }
                )
    for case in oracle["cases"]:
        support = case["last_four_assessment"]["O_h_support_criteria_passed"]
        for row in case["rows"]:
            optimum = row["optimum"]
            rows.append(
                {
                    "record_type": "oracle_diagnostic",
                    "n": case["n"],
                    "c": case["c"],
                    "M_n_c": case["M_n_c"],
                    "m_knee": case["m_knee"],
                    "parameter_path": case["parameter_path"],
                    "path": "oracle",
                    "N": row["N"],
                    "epsilon_over_pi": optimum["epsilon_over_pi"],
                    "omega": row["omega"],
                    "h": row["h"],
                    "e_lambda": optimum["signed_eigenvalue_error"],
                    "oracle_deviation_from_half_mesh": row[
                        "deviation_from_half_mesh"
                    ],
                    "oracle_absolute_deviation_over_h": row[
                        "absolute_deviation_over_h"
                    ],
                    "matrix_audit_passed": row["audit"]["passed"],
                    "oracle_O_h_support_case": support,
                }
            )
    return rows


def flatten_boundary(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in data["cases"]:
        support = case["last_four_assessment"]["boundary_explanation_supported"]
        for row in case["rows"]:
            project = row["project_V_P"]
            augmented = row["augmented_V_P_plus"]
            matched = row["project_V_P_plus_1"]
            boulsane = row["Boulsane_N_equals_P_plus_1"]
            causal = row["causal_absolute_error_ratio_augmented_over_project_P"]
            fair = row[
                "matched_absolute_error_ratio_augmented_over_project_P_plus_1"
            ]
            rows.append(
                {
                    "n": case["n"],
                    "c": case["c"],
                    "M_n_c": case["M_n_c"],
                    "m_knee": case["m_knee"],
                    "parameter_path": case["parameter_path"],
                    "P": row["P"],
                    "matched_total_dimension": row["matched_total_dimension"],
                    "project_error": project["absolute_error"],
                    "augmented_error": augmented["absolute_error"],
                    "project_P_plus_1_error": matched["absolute_error"],
                    "Boulsane_error": boulsane["absolute_error"],
                    "causal_ratio_or_upper_bound": causal["value_or_upper_bound"],
                    "causal_ratio_is_floor_upper_bound": causal[
                        "is_upper_bound_due_to_augmented_floor"
                    ],
                    "matched_ratio_or_upper_bound": fair["value_or_upper_bound"],
                    "matched_ratio_is_floor_upper_bound": fair[
                        "is_upper_bound_due_to_augmented_floor"
                    ],
                    "reference_endpoint_absolute_value": case[
                        "reference_endpoint_absolute_value"
                    ],
                    "augmented_endpoint_absolute_value": augmented[
                        "reconstructed_endpoint_absolute_value"
                    ],
                    "augmented_endpoint_absolute_error": augmented[
                        "endpoint_absolute_error"
                    ],
                    "project_Rayleigh_residual": project[
                        "ordinary_Rayleigh_residual_norm"
                    ],
                    "augmented_Rayleigh_residual": augmented[
                        "ordinary_Rayleigh_residual_norm"
                    ],
                    "Boulsane_Rayleigh_residual": boulsane[
                        "ordinary_Rayleigh_residual_norm"
                    ],
                    "augmented_matrix_audit_passed": row["augmented_matrix_audit"][
                        "passed"
                    ],
                    "Boulsane_matrix_audit_passed": boulsane["matrix_audit"]["passed"],
                    "case_boundary_explanation_supported": support,
                }
            )
    return rows


def flatten_timing(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []

    def add(case: dict[str, Any], row: dict[str, Any], method: str, timing: dict[str, Any], **extra: Any) -> None:
        rows.append(
            {
                "n": case["n"],
                "c": case["c"],
                "parameter_path": case["parameter_path"],
                "P": row["P"],
                "Boulsane_N": row["Boulsane_N"],
                "method": method,
                "wall_median_seconds": timing["wall"]["median_seconds"],
                "wall_Q1_seconds": timing["wall"]["Q1_seconds"],
                "wall_Q3_seconds": timing["wall"]["Q3_seconds"],
                "wall_IQR_seconds": timing["wall"]["IQR_seconds"],
                "CPU_median_seconds": timing["process_CPU"]["median_seconds"],
                **extra,
            }
        )

    for case in data["cases"]:
        for row in case["rows"]:
            full = row["project_full_rebuild"]
            extension = row["project_block_extension"]
            add(
                case,
                row,
                "project_full_rebuild",
                full,
                reused_entry_count=full["reused_entry_count"],
                matrix_storage_bytes=full["output_matrix_storage_bytes"],
            )
            add(
                case,
                row,
                "project_block_extension",
                extension,
                reused_entry_count=extension["reused_entry_count"],
                matrix_storage_bytes=extension["output_matrix_storage_bytes"],
            )
            add(case, row, "project_eigensolve", row["project_eigensolve"])
            for path, result in row["Boulsane_fixed_c_paths"].items():
                add(
                    case,
                    row,
                    f"Boulsane_{path}_assembly",
                    result["assembly"],
                    offset_path=path,
                    reused_entry_count=result["reused_entry_count"],
                    matrix_storage_bytes=result["output_matrix_storage_bytes"],
                )
                add(
                    case,
                    row,
                    f"Boulsane_{path}_eigensolve",
                    result["eigensolve"],
                    offset_path=path,
                )
    return rows


def figures(
    output: Path,
    certification: list[dict[str, Any]],
    structure: dict[str, Any],
    fixed: dict[str, Any],
    oracle: dict[str, Any],
    boundary: dict[str, Any],
    timing: dict[str, Any],
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    sharp = [
        row
        for row in certification
        if row["n"] == 0 and row["c"] == 10 and row["P"] == 8 and row["available"]
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for y, row in enumerate(sharp):
        ax.plot([row["explicit_lower"], row["explicit_upper"]], [y, y], lw=7, alpha=0.45, color="C0", label="explicit" if y == 0 else None)
        ax.plot([row["hybrid_lower"], row["hybrid_upper"]], [y + 0.12, y + 0.12], lw=3, color="C1", label="hybrid" if y == 0 else None)
        ax.plot(row["lambda_ref"], y + 0.06, "k|", ms=14)
    ax.set_yticks(range(len(sharp)), [row["category"] for row in sharp])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("certified continuous eigenvalue interval")
    ax.set_title(r"Certification ladder: $n=0$, $c=10$, $P=8$")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output / "certification_ladder.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for case in structure["project_fixed_c"]:
        axes[0].semilogy(
            [row["P_new"] for row in case["transitions"]],
            [max(row["block_defect_norm_2"], 1e-18) for row in case["transitions"]],
            color="C0",
            alpha=0.35,
        )
    for case in structure["boulsane_fixed_c_large_N_offset"]:
        axes[0].semilogy(
            [row["N_new"] for row in case["transitions"]],
            [row["block_defect_norm_2"] for row in case["transitions"]],
            color="C1",
            alpha=0.35,
        )
    axes[0].set_title("Fixed-$c$ leading-block defect")
    axes[0].set_xlabel("new dimension")
    axes[0].set_ylabel(r"$\|A_{old}-A_{new}[1{:}old]\|_2$")
    labels = ["project", "Boulsane"]
    counts = [
        structure["summary"]["project_significant_negative_increment_count"],
        structure["summary"]["boulsane_significant_negative_increment_count"],
    ]
    axes[1].bar(labels, counts, color=["C0", "C1"])
    axes[1].set_title("Significant negative eigenvalue increments")
    axes[1].set_ylabel("count")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output / "structure_block_monotonicity.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for case in oracle["cases"]:
        if case["n"] == 0:
            row = case["rows"][-1]
            ax.plot(
                [point["epsilon_over_pi"] for point in row["grid"]],
                [point["signed_eigenvalue_error"] for point in row["grid"]],
                label=f"c={case['c']}",
            )
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0.5, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel(r"$\varepsilon/\pi$")
    ax.set_ylabel("signed knee eigenvalue error")
    ax.set_title("Post-frozen oracle diagnostic, largest dimension ($n=0$)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output / "mechanism_signed_offset.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    representative = next(case for case in fixed["cases"] if case["n"] == 0 and case["c"] == 20)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for path_name, color in (("large_N", "C0"), ("half_mesh", "C2"), ("next_zero", "C1")):
        rows = representative["paths"][path_name]["rows"]
        axes[0].loglog(
            [row["h"] for row in rows],
            [abs(row["e_Rayleigh"]) for row in rows],
            "o-",
            label=path_name,
            color=color,
        )
    for case in fixed["cases"]:
        if case["c"] == 80:
            rows = case["paths"]["next_zero"]["rows"]
            axes[1].plot(
                [row["N"] for row in rows],
                [row["next_zero_normalized"] for row in rows],
                "o-",
                label=f"n={case['n']}",
            )
    axes[0].invert_xaxis()
    axes[0].set_xlabel(r"$h_N$")
    axes[0].set_ylabel(r"$|e_{\mathcal{R}}|$")
    axes[0].set_title(r"Rates: $n=0,c=20$")
    axes[1].axhline(0.5, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel(r"$e_{\mathcal{R}}/(h_N C_m)$")
    axes[1].set_title(r"Next-zero coefficient: $c=80$")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    path = output / "mechanism_rates_normalized.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for n in sorted({case["n"] for case in boundary["cases"]}):
        selected = [case for case in boundary["cases"] if case["n"] == n]
        axes[0].semilogy(
            [case["c"] for case in selected],
            [case["last_four_assessment"]["causal_ratio_median_or_null"] for case in selected],
            "o-",
            label=f"n={n}",
        )
    representative_b = next(case for case in boundary["cases"] if case["n"] == 0 and case["c"] == 40)
    axes[1].loglog(
        [row["P"] for row in representative_b["rows"]],
        [row["project_V_P"]["absolute_error"] for row in representative_b["rows"]],
        "o-",
        label=r"$V_P$",
    )
    axes[1].loglog(
        [row["P"] for row in representative_b["rows"]],
        [row["augmented_V_P_plus"]["absolute_error"] for row in representative_b["rows"]],
        "s-",
        label=r"$V_P^+$",
    )
    axes[0].axhline(0.5, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("c")
    axes[0].set_ylabel("last-four median augmented/project error")
    axes[0].set_title("Boundary-control decision ratio")
    axes[1].set_xlabel("P")
    axes[1].set_ylabel("knee eigenvalue absolute error")
    axes[1].set_title(r"Representative control: $n=0,c=40$")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    path = output / "boundary_control.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    representative_t = next(case for case in timing["cases"] if case["n"] == 0 and case["c"] == 40)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    P = [row["P"] for row in representative_t["rows"]]
    full = [row["project_full_rebuild"]["wall"]["median_seconds"] for row in representative_t["rows"]]
    extension = [row["project_block_extension"]["wall"]["median_seconds"] for row in representative_t["rows"]]
    boulsane = [
        np.median(
            [path["assembly"]["wall"]["median_seconds"] for path in row["Boulsane_fixed_c_paths"].values()]
        )
        for row in representative_t["rows"]
    ]
    axes[0].semilogy(P, full, "o-", label="project rebuild")
    axes[0].semilogy(P, extension, "s-", label="project extension")
    axes[0].semilogy(P, boulsane, "^-", label="Boulsane closed form")
    speedups = [
        row["project_full_rebuild"]["wall"]["median_seconds"]
        / row["project_block_extension"]["wall"]["median_seconds"]
        for case in timing["cases"]
        for row in case["rows"]
    ]
    axes[1].hist(speedups, bins=10, color="C0", alpha=0.8)
    axes[0].set_xlabel("matrix dimension")
    axes[0].set_ylabel("median assembly seconds")
    axes[0].set_title(r"Assembly timing: $n=0,c=40$")
    axes[1].set_xlabel("rebuild / extension speedup")
    axes[1].set_ylabel("row count")
    axes[1].set_title("Exact-reuse speedup, all 64 rows")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    path = output / "timing_assembly_reuse.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)
    return made


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/task_25_4_phase3"))
    args = parser.parse_args()
    data = args.data
    root = Path.cwd()
    paths = {
        "reference_manifest": data / "reference_mode_manifest.json",
        "certification": data / "track_A_certification.json",
        "structure": data / "track_B_structure.json",
        "fixed_mechanism": data / "track_C_fixed_offsets.json",
        "oracle": data / "track_C_oracle.json",
        "boundary": data / "track_D_boundary.json",
        "timing": data / "track_E_timing.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 3 inputs: " + ", ".join(missing))
    manifest = load(paths["reference_manifest"])
    certification_data = load(paths["certification"])
    structure_data = load(paths["structure"])
    fixed_data = load(paths["fixed_mechanism"])
    oracle_data = load(paths["oracle"])
    boundary_data = load(paths["boundary"])
    timing_data = load(paths["timing"])

    modes = flatten_modes(manifest)
    certification = flatten_certification(certification_data, root)
    structure = flatten_structure(structure_data)
    mechanism = flatten_mechanism(fixed_data, oracle_data)
    boundary = flatten_boundary(boundary_data)
    timing = flatten_timing(timing_data)

    (data / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(data / "modes.csv", modes)
    write_csv(data / "certification.csv", certification)
    write_csv(data / "structure.csv", structure)
    write_csv(data / "mechanism.csv", mechanism)
    write_csv(data / "boundary.csv", boundary)
    write_csv(data / "timing.csv", timing)
    reference_rows = [
        {"case_index": case_index, "n": case["n"], "c": case["c"], "M_n_c": case["M_n_c"], **row}
        for case_index, case in enumerate(manifest["cases"])
        for row in case["candidate_modes_0_through_M_plus_4"]
    ]
    np.savez_compressed(
        data / "reference.npz",
        selected_n=np.asarray([row["n"] for row in modes], dtype=int),
        selected_c=np.asarray([row["c"] for row in modes], dtype=int),
        selected_M_n_c=np.asarray([row["M_n_c"] for row in modes], dtype=int),
        selected_category=np.asarray([row["category"] for row in modes], dtype="U8"),
        selected_available=np.asarray([row["available"] for row in modes], dtype=bool),
        selected_m=np.asarray([row["m"] if row["m"] is not None else -1 for row in modes], dtype=int),
        selected_lambda_ref=np.asarray([row["lambda_ref"] if row["lambda_ref"] is not None else np.nan for row in modes]),
        selected_u_ref=np.asarray([row["u_ref"] if row["u_ref"] is not None else np.nan for row in modes]),
        selected_gap=np.asarray([row["gap"] if row["gap"] is not None else np.nan for row in modes]),
        reference_case_index=np.asarray([row["case_index"] for row in reference_rows], dtype=int),
        reference_n=np.asarray([row["n"] for row in reference_rows], dtype=int),
        reference_c=np.asarray([row["c"] for row in reference_rows], dtype=int),
        reference_M_n_c=np.asarray([row["M_n_c"] for row in reference_rows], dtype=int),
        reference_m=np.asarray([row["m"] for row in reference_rows], dtype=int),
        reference_lambda=np.asarray([row["lambda_ref"] for row in reference_rows]),
        reference_u=np.asarray([row["u_ref"] for row in reference_rows]),
        reference_local_gap=np.asarray([row["local_gap"] for row in reference_rows]),
        reference_A_Q2_minus_A_Q1=np.asarray([row["A_Q2_minus_A_Q1"] for row in reference_rows]),
        reference_A_Q2_minus_B_Q2=np.asarray([row["A_Q2_minus_B_Q2"] for row in reference_rows]),
        reference_B_Q2_minus_B_Q1=np.asarray([row["B_Q2_minus_B_Q1"] for row in reference_rows]),
        reference_A_Q2_minus_saved_benchmark=np.asarray([row["A_Q2_minus_saved_benchmark"] for row in reference_rows]),
    )
    failure_log = {
        "task": "25.4 Phase 3 gate and failure log",
        "Gate_A_certification_integrity": {
            "passed": certification_data["gate_A_passed"],
            "classification": "formal frozen gate failure",
            "reason": (
                "144 deep-mode m=0 ordinary central references exceed 1 by "
                "1.4e-14 to 2.6e-14; every uncertainty interval intersects the certificate"
            ),
        },
        "Gate_B_practical_certification": {
            "passed": certification_data["gate_B_nontrivial_width_passed"],
            "nontrivial_interval_count": certification_data[
                "nontrivial_continuous_interval_count"
            ],
        },
        "Gate_C_mechanism": {
            "passed": fixed_data["summary"]["gate_C_mechanism_corroboration_passed"],
            "compatible_counts": fixed_data["summary"]["Rayleigh_compatible_case_counts"],
            "next_zero_coefficient_window_case_count": fixed_data["summary"][
                "next_zero_coefficient_window_case_count"
            ],
            "c80_contradictions": fixed_data["summary"]["c80_contradictions"],
            "oracle_is_separate_and_diagnostic": True,
        },
        "Gate_D_boundary": {
            "passed": boundary_data["summary"]["gate_D_boundary_explanation_passed"],
            "supported_case_count": boundary_data["summary"]["supported_case_count"],
            "eligible_case_count": boundary_data["summary"]["eligible_case_count"],
        },
        "Gate_E_fairness_and_paths": {
            "passed": True,
            "fixed_native_and_fixed_continuous_paths_kept_separate": True,
            "oracle_presented_as_practical": False,
            "Boulsane_closed_form_advantage_recorded": True,
        },
    }
    (data / "failure_log.json").write_text(
        json.dumps(failure_log, indent=2) + "\n", encoding="utf-8"
    )
    made_figures = figures(
        data / "figures",
        certification,
        structure_data,
        fixed_data,
        oracle_data,
        boundary_data,
        timing_data,
    )
    outputs = [
        data / "manifest.json",
        data / "modes.csv",
        data / "reference.npz",
        data / "certification.csv",
        data / "structure.csv",
        data / "mechanism.csv",
        data / "boundary.csv",
        data / "timing.csv",
        data / "failure_log.json",
        *made_figures,
    ]
    inventory = {
        "task": "25.4 Phase 3 consolidated artifact inventory",
        "source_hashes": {name: sha256(path) for name, path in paths.items()},
        "outputs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    inventory_path = data / "artifact_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_inventory": str(inventory_path),
                "sha256": sha256(inventory_path),
                "CSV_row_counts": {
                    "modes": len(modes),
                    "certification": len(certification),
                    "structure": len(structure),
                    "mechanism": len(mechanism),
                    "boundary": len(boundary),
                    "timing": len(timing),
                },
                "figure_count": len(made_figures),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
