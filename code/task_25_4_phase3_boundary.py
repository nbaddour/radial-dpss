"""Task 25.4 Phase 3, Track D boundary-augmented Ritz control.

The fixed augmentation is u_n(x)=x**n.  The orthonormal augmented Ritz
matrix and an independently quadrature-assembled generalized eigenproblem
are compared at every dimension.  Ordinary Nyström Rayleigh residuals are
diagnostics only, not certificates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.special import jn_zeros, jv

from task_12_1_assemble_matrix import _panels_nodes, assemble
from task_25_4_phase3_freeze_manifest import kernel_matrix
from task_25_4_phase3_mechanism import BUFFERS, audit_boulsane, offset_value
from task_25_4_phase3_structure import boulsane_matrix


EPS = np.finfo(float).eps
AUDIT_FACTOR = 1e-11


def frequency_blocks(
    n: int, c: int, zeros: np.ndarray, G: int
) -> tuple[np.ndarray, np.ndarray, float]:
    below = zeros[zeros < c]
    u, weights = _panels_nodes(float(c), below, G)
    denominator = u[:, None] ** 2 - zeros[None, :] ** 2
    transform_phi = (
        -math.sqrt(2) * zeros[None, :] * jv(n, u)[:, None] / denominator
    )
    transform_u = jv(n + 1, u) / u
    measure = u * weights
    b = transform_phi.T @ (measure * transform_u)
    d = float(np.sum(measure * transform_u * transform_u))
    direct = transform_phi.T @ (measure[:, None] * transform_phi)
    return 0.5 * (direct + direct.T), b, d


def orthogonal_augmented(
    project: np.ndarray,
    b: np.ndarray,
    d: float,
    coefficients: np.ndarray,
    norm_u_squared: float,
) -> tuple[np.ndarray, float]:
    residual_squared = float(norm_u_squared - coefficients @ coefficients)
    if residual_squared <= 0:
        raise ArithmeticError("nonpositive boundary residual norm")
    residual = math.sqrt(residual_squared)
    cross = (b - project @ coefficients) / residual
    corner = float(
        (d - 2 * coefficients @ b + coefficients @ project @ coefficients)
        / residual_squared
    )
    matrix = np.block(
        [[project, cross[:, None]], [cross[None, :], np.asarray([[corner]])]]
    )
    return 0.5 * (matrix + matrix.T), residual


def nonorthogonal_problem(
    direct_project: np.ndarray,
    b: np.ndarray,
    d: float,
    coefficients: np.ndarray,
    norm_u_squared: float,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.block(
        [[direct_project, b[:, None]], [b[None, :], np.asarray([[d]])]]
    )
    gram = np.block(
        [
            [np.eye(len(coefficients)), coefficients[:, None]],
            [coefficients[None, :], np.asarray([[norm_u_squared]])],
        ]
    )
    return 0.5 * (matrix + matrix.T), 0.5 * (gram + gram.T)


def basis_values(n: int, zeros: np.ndarray, x: np.ndarray, absolute: bool) -> np.ndarray:
    denominator = jv(n + 1, zeros)
    if absolute:
        denominator = np.abs(denominator)
    return math.sqrt(2) * jv(n, np.outer(x, zeros)) / denominator[None, :]


def rayleigh_residual(
    operator: np.ndarray, scale: np.ndarray, function: np.ndarray
) -> dict[str, float]:
    vector = scale * function
    norm = float(np.linalg.norm(vector))
    vector = vector / norm
    applied = operator @ vector
    quotient = float(vector @ applied)
    residual = float(np.linalg.norm(applied - quotient * vector))
    return {
        "quadrature_norm_before_normalization": norm,
        "Rayleigh_quotient": quotient,
        "ordinary_Rayleigh_residual_norm": residual,
    }


def eigpair_descending(matrix: np.ndarray, m: int) -> tuple[float, np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    return float(values[m]), vectors[:, m], values


def error_record(lambda_ref: float, value: float) -> dict[str, float]:
    signed = float(lambda_ref - value)
    return {
        "eigenvalue": float(value),
        "signed_error": signed,
        "absolute_error": abs(signed),
    }


def conservative_ratio(
    numerator: float, denominator: float, u_ref: float
) -> dict[str, object]:
    eligible = bool(denominator >= 10 * u_ref)
    if not eligible:
        return {
            "eligible": False,
            "value_or_upper_bound": None,
            "is_upper_bound_due_to_augmented_floor": None,
        }
    floor_limited = bool(numerator < 10 * u_ref)
    value = (10 * u_ref if floor_limited else numerator) / denominator
    return {
        "eligible": True,
        "value_or_upper_bound": float(value),
        "is_upper_bound_due_to_augmented_floor": floor_limited,
    }


def case_run(case: dict[str, object]) -> dict[str, object]:
    n, c, M = int(case["n"]), int(case["c"]), int(case["M_n_c"])
    knee = next(row for row in case["selected_modes"] if row["category"] == "knee")
    m = int(knee["m"])
    lambda_ref = float(knee["lambda_ref"])
    u_ref = float(knee["u_ref"])
    endpoint_ref = float(case["knee_endpoint"]["absolute_value_Q2"])
    dimensions = [M + buffer for buffer in BUFFERS]
    largest = max(dimensions) + 1
    zeros = jn_zeros(n, largest + 1)

    project48 = assemble(n, float(c), P=largest, G=48)
    project32 = assemble(n, float(c), P=largest, G=32)
    direct64, b64, d64 = frequency_blocks(n, c, zeros[:largest], 64)
    _, b48, d48 = frequency_blocks(n, c, zeros[:largest], 48)

    Q = int(case["quadrature_orders"]["Q2"])
    t, w = np.polynomial.legendre.leggauss(Q)
    x = 0.5 * (t + 1)
    w = 0.5 * w
    scale = np.sqrt(x * w)
    continuous_operator = (
        scale[:, None] * kernel_matrix(n, float(c), x) * scale[None, :]
    )
    continuous_operator = 0.5 * (continuous_operator + continuous_operator.T)
    signed_basis = basis_values(n, zeros[:largest], x, absolute=False)
    absolute_basis = basis_values(n, zeros[:largest], x, absolute=True)

    norm_u_squared = 1 / (2 * n + 2)
    rows = []
    for P in dimensions:
        started = time.perf_counter()
        project = project48[:P, :P]
        project_next = project48[: P + 1, : P + 1]
        coefficients = math.sqrt(2) / zeros[:P]
        augmented, residual_norm = orthogonal_augmented(
            project, b48[:P], d48, coefficients, norm_u_squared
        )
        nonorth, gram = nonorthogonal_problem(
            direct64[:P, :P], b64[:P], d64, coefficients, norm_u_squared
        )
        generalized_values = sla.eigvalsh(
            nonorth, gram, check_finite=False, driver="gvd"
        )[::-1]
        augmented_value, augmented_vector, augmented_values = eigpair_descending(augmented, m)
        project_value, project_vector, _ = eigpair_descending(project, m)
        next_value, next_vector, _ = eigpair_descending(project_next, m)

        transform = np.zeros((P + 1, P + 1))
        transform[:P, :P] = np.eye(P)
        transform[:P, P] = -coefficients / residual_norm
        transform[P, P] = 1 / residual_norm
        transformed_direct = transform.T @ nonorth @ transform
        tolerance = AUDIT_FACTOR * max(1.0, float(np.linalg.norm(augmented, 2)))
        audit_disagreements = {
            "orthogonal_vs_transformed_direct_norm_2": float(
                np.linalg.norm(augmented - transformed_direct, 2)
            ),
            "orthogonal_vs_generalized_spectrum_max_abs": float(
                np.max(np.abs(augmented_values - generalized_values))
            ),
            "project_G48_vs_G32_norm_2": float(
                np.linalg.norm(project - project32[:P, :P], 2)
            ),
            "project_G48_vs_direct_G64_norm_2": float(
                np.linalg.norm(project - direct64[:P, :P], 2)
            ),
            "symmetry_defect": float(np.max(np.abs(augmented - augmented.T))),
        }
        augmented_audit = {
            "tolerance": tolerance,
            **audit_disagreements,
            "maximum_disagreement": max(audit_disagreements.values()),
            "passed": bool(max(audit_disagreements.values()) <= tolerance),
            "gram_minimum_eigenvalue": float(np.linalg.eigvalsh(gram)[0]),
            "orthogonalization_residual_norm": residual_norm,
            "ill_conditioned_threshold": float(1e4 * EPS),
            "declared_ill_conditioned": bool(residual_norm < 1e4 * EPS),
        }

        project_function = signed_basis[:, :P] @ project_vector
        next_function = signed_basis[:, : P + 1] @ next_vector
        boundary_function = (
            x**n - signed_basis[:, :P] @ coefficients
        ) / residual_norm
        augmented_endpoint = float(augmented_vector[-1] / residual_norm)
        if augmented_endpoint < 0:
            augmented_vector = -augmented_vector
            augmented_endpoint = -augmented_endpoint
        augmented_function = (
            signed_basis[:, :P] @ augmented_vector[:P]
            + augmented_vector[-1] * boundary_function
        )

        N = P + 1
        epsilon = offset_value("large_N", n, N, zeros)
        D = float(zeros[N - 1] + epsilon)
        omega = float(c / D)
        boulsane = boulsane_matrix(n, N, zeros, omega)
        boulsane_value, boulsane_vector, _ = eigpair_descending(boulsane, m)
        boulsane_function = absolute_basis[:, :N] @ boulsane_vector
        boulsane_audit = audit_boulsane(n, N, zeros, omega, boulsane)

        project_error = error_record(lambda_ref, project_value)
        augmented_error = error_record(lambda_ref, augmented_value)
        next_error = error_record(lambda_ref, next_value)
        boulsane_error = error_record(lambda_ref, boulsane_value)
        causal_ratio = conservative_ratio(
            augmented_error["absolute_error"], project_error["absolute_error"], u_ref
        )
        matched_ratio = conservative_ratio(
            augmented_error["absolute_error"], next_error["absolute_error"], u_ref
        )
        rows.append(
            {
                "P": P,
                "matched_total_dimension": P + 1,
                "project_V_P": {
                    **project_error,
                    "reconstructed_endpoint_absolute_value": 0.0,
                    "endpoint_absolute_error": endpoint_ref,
                    **rayleigh_residual(continuous_operator, scale, project_function),
                },
                "augmented_V_P_plus": {
                    **augmented_error,
                    "reconstructed_endpoint_absolute_value": augmented_endpoint,
                    "endpoint_absolute_error": abs(augmented_endpoint - endpoint_ref),
                    **rayleigh_residual(continuous_operator, scale, augmented_function),
                },
                "project_V_P_plus_1": {
                    **next_error,
                    "reconstructed_endpoint_absolute_value": 0.0,
                    "endpoint_absolute_error": endpoint_ref,
                    **rayleigh_residual(continuous_operator, scale, next_function),
                },
                "Boulsane_N_equals_P_plus_1": {
                    **boulsane_error,
                    "offset_path": "large-N fixed-c",
                    "epsilon": epsilon,
                    "epsilon_over_pi": float(epsilon / np.pi),
                    "omega": omega,
                    "reconstructed_endpoint_absolute_value": 0.0,
                    "endpoint_absolute_error": endpoint_ref,
                    **rayleigh_residual(continuous_operator, scale, boulsane_function),
                    "matrix_audit": boulsane_audit,
                },
                "causal_absolute_error_ratio_augmented_over_project_P": causal_ratio,
                "matched_absolute_error_ratio_augmented_over_project_P_plus_1": matched_ratio,
                "augmented_matrix_audit": augmented_audit,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )

    last = rows[-4:]
    eligible_causal = [
        row["causal_absolute_error_ratio_augmented_over_project_P"]
        for row in last
        if row["causal_absolute_error_ratio_augmented_over_project_P"]["eligible"]
    ]
    eligible_matched = [
        row["matched_absolute_error_ratio_augmented_over_project_P_plus_1"]
        for row in last
        if row["matched_absolute_error_ratio_augmented_over_project_P_plus_1"]["eligible"]
    ]
    causal_median = (
        float(np.median([row["value_or_upper_bound"] for row in eligible_causal]))
        if len(eligible_causal) == 4
        else None
    )
    matched_median = (
        float(np.median([row["value_or_upper_bound"] for row in eligible_matched]))
        if len(eligible_matched) == 4
        else None
    )
    causal_pass = bool(causal_median is not None and causal_median < 0.5)
    matched_pass = bool(matched_median is not None and matched_median < 0.75)
    return {
        "n": n,
        "c": c,
        "M_n_c": M,
        "m_knee": m,
        "lambda_ref": lambda_ref,
        "u_ref": u_ref,
        "reference_endpoint_absolute_value": endpoint_ref,
        "parameter_path": "fixed continuous c",
        "augmentation": "u_n(x)=x^n",
        "rows": rows,
        "last_four_assessment": {
            "dimensions_P": [row["P"] for row in last],
            "causal_eligible_count": len(eligible_causal),
            "matched_eligible_count": len(eligible_matched),
            "causal_ratio_median_or_null": causal_median,
            "matched_ratio_median_or_null": matched_median,
            "causal_threshold_below_one_half_passed": causal_pass,
            "matched_threshold_below_three_quarters_passed": matched_pass,
            "boundary_explanation_supported": bool(causal_pass and matched_pass),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/task_25_4_phase3/reference_mode_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/task_25_4_phase3/track_D_boundary.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    started = time.perf_counter()
    cases = []
    for case in manifest["cases"]:
        result = case_run(case)
        cases.append(result)
        assessment = result["last_four_assessment"]
        print(
            f"n={result['n']} c={result['c']}: "
            f"causal={assessment['causal_ratio_median_or_null']} "
            f"matched={assessment['matched_ratio_median_or_null']} "
            f"supported={assessment['boundary_explanation_supported']}",
            flush=True,
        )
    supported = sum(
        bool(case["last_four_assessment"]["boundary_explanation_supported"])
        for case in cases
    )
    eligible = sum(
        case["last_four_assessment"]["causal_eligible_count"] == 4
        and case["last_four_assessment"]["matched_eligible_count"] == 4
        for case in cases
    )
    payload = {
        "task": "25.4 Phase 3 Track D boundary-augmented Ritz control",
        "evidence_label": "ordinary numerical control; not certification",
        "parameter_path": "fixed continuous c",
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "eligible_case_count": eligible,
            "supported_case_count": supported,
            "gate_D_boundary_explanation_passed": bool(
                eligible > 0 and supported >= 0.75 * eligible
            ),
            "all_augmented_matrix_audits_passed": all(
                row["augmented_matrix_audit"]["passed"]
                for case in cases
                for row in case["rows"]
            ),
            "all_Boulsane_matrix_audits_passed": all(
                row["Boulsane_N_equals_P_plus_1"]["matrix_audit"]["passed"]
                for case in cases
                for row in case["rows"]
            ),
            "no_ill_conditioned_rows": all(
                not row["augmented_matrix_audit"]["declared_ill_conditioned"]
                for case in cases
                for row in case["rows"]
            ),
        },
        "elapsed_seconds_total": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"output": str(args.output), "sha256": digest, **payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
