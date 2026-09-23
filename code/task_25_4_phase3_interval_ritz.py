"""Rigorous interval pilots for Task 25.4 Phase 3.

The implementation encloses the exact project Ritz matrix through the scalar
partial-fraction integrals

    I_p = integral_0^c u J_n(u)^2 / (u^2-j_p^2) du,
    D_p = integral_0^c u J_n(u)^2 / (u^2-j_p^2)^2 du.

Each real cell is integrated by an interval range enclosure times its exact
width, so there is no uncomputed quadrature remainder.  On the cell containing
j_p, the quotient J_n(u)/(u^2-j_p^2) is bounded with the derivative-average
identity required by the Phase 3 protocol.  The method is intentionally
conservative; it is a pilot and fallback certificate, not yet an optimized
full-campaign integrator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np
from flint import acb, acb_mat, arb, ctx

from task_12_1_assemble_matrix import assemble, assemble_routeA
from task_25_4_phase3_gate0_interval import (
    REFERENCE_DPS,
    _ball_record,
    _bessel_derivative,
    _certified_bessel_zero,
)


BALL_BITS = 160


def certified_bessel_zeros(n: int, P: int) -> tuple[list[arb], list[dict[str, object]]]:
    roots: list[arb] = []
    records: list[dict[str, object]] = []
    for k in range(1, P + 1):
        root, record = _certified_bessel_zero(n, k)
        if not record["passed"]:
            raise RuntimeError(f"failed to certify j_{{{n},{k}}}")
        if roots and not roots[-1].upper() < root.lower():
            raise RuntimeError(f"Bessel-zero intervals {k-1} and {k} are not ordered")
        roots.append(root)
        records.append(record)
    return roots, records


def removable_quotient_enclosure(n: int, X: arb, JX: arb, root: arb) -> tuple[arb, bool]:
    """Enclose J_n(u)/(u^2-j^2) for all u in X and the certified root j."""
    denominator = X * X - root * root
    # Keep the stable identity on the crossing cell and its immediate
    # neighbours, but let this neighbourhood shrink with refinement.  A fixed
    # physical radius would leave a nonvanishing range-enclosure floor.
    identity_radius = 4 * arb(X.rad())
    near_root = bool(
        X.lower() < root.upper() + identity_radius
        and X.upper() > root.lower() - identity_radius
    )
    if not denominator.contains(0) and not near_root:
        return JX / denominator, False

    # For the exact zero j and every u in X,
    # J_n(u)/(u^2-j^2) = average_{t in [0,1]} J_n'(j+t(u-j))/(u+j).
    # The real interval hull encloses the full line segment, so its derivative
    # range encloses that average.  This avoids division by an interval through 0.
    segment = X.union(root)
    derivative_range = _bessel_derivative(n, segment)
    safe_denominator = X + root
    if safe_denominator.contains(0):
        raise RuntimeError("unexpected zero in u+j denominator")
    quotient = derivative_range / safe_denominator
    if not quotient.is_finite():
        raise RuntimeError("non-finite removable quotient enclosure")
    return quotient, True


def square_enclosure(X: arb) -> arb:
    """Dependency-aware enclosure of {x^2 : x in X}."""
    lower_abs = X.abs_lower()
    upper_abs = X.abs_upper()
    lower_square = (lower_abs * lower_abs).abs_lower()
    upper_square = (upper_abs * upper_abs).abs_upper()
    midpoint = (lower_square + upper_square) / 2
    radius = (upper_square - lower_square) / 2
    return arb(midpoint, radius.abs_upper())


def interval_riemann_ID(
    n: int,
    c: int,
    roots: list[arb],
    cells: int,
) -> tuple[list[arb], list[arb], list[list[arb]], dict[str, object]]:
    """Rigorous range-sum enclosures for I_p, D_p, and the direct Gram matrix."""
    if c <= 0 or cells <= 0:
        raise ValueError("c and cells must be positive")

    P = len(roots)
    I = [arb(0) for _ in range(P)]
    D = [arb(0) for _ in range(P)]
    gram = [[arb(0) for _ in range(P)] for _ in range(P)]
    gram_factors = [
        [2 * roots[m] * roots[k] for k in range(P)]
        for m in range(P)
    ]
    width = arb(c) / cells
    half_width = arb(c) / (2 * cells)
    removable_hits = [0 for _ in range(P)]
    started = time.perf_counter()

    for cell in range(cells):
        midpoint = arb((2 * cell + 1) * c) / (2 * cells)
        X = arb(midpoint, half_width)
        JX = X.bessel_j(n)
        quotients: list[arb] = []
        for p, root in enumerate(roots):
            quotient, used_removable = removable_quotient_enclosure(n, X, JX, root)
            quotients.append(quotient)
            if used_removable:
                removable_hits[p] += 1
            I[p] += width * X * JX * quotient
            D[p] += width * X * square_enclosure(quotient)

        cell_measure = width * X
        for m in range(P):
            for k in range(m, P):
                quotient_product = (
                    square_enclosure(quotients[m])
                    if m == k
                    else quotients[m] * quotients[k]
                )
                gram[m][k] += (
                    gram_factors[m][k]
                    * cell_measure
                    * quotient_product
                )

    for m in range(P):
        for k in range(m + 1, P):
            gram[k][m] = gram[m][k]

    elapsed = time.perf_counter() - started
    if not all(value.is_finite() for value in I + D):
        raise RuntimeError("non-finite scalar integral enclosure")
    if not all(value.is_finite() for row in gram for value in row):
        raise RuntimeError("non-finite direct Gram enclosure")
    return I, D, gram, {
        "cells": cells,
        "cell_width": _ball_record(width),
        "elapsed_seconds": elapsed,
        "removable_cell_hits_by_root": removable_hits,
    }


def assemble_interval_matrix(roots: list[arb], I: list[arb], D: list[arb]) -> list[list[arb]]:
    """Assemble a symmetric interval matrix from certified scalar integrals."""
    P = len(roots)
    matrix = [[arb(0) for _ in range(P)] for _ in range(P)]
    for m in range(P):
        matrix[m][m] = 2 * roots[m] * roots[m] * D[m]
        for k in range(m + 1, P):
            factor = 2 * roots[m] * roots[k]
            forward = factor * (I[m] - I[k]) / (roots[m] * roots[m] - roots[k] * roots[k])
            reverse = factor * (I[k] - I[m]) / (roots[k] * roots[k] - roots[m] * roots[m])
            # The two directed evaluations represent the independently evaluated
            # (m,k) and (k,m) entries.  Exact symmetry permits intersection.
            entry = forward.intersection(reverse)
            matrix[m][k] = entry
            matrix[k][m] = entry
    return matrix


def intersect_interval_matrices(
    first: list[list[arb]], second: list[list[arb]]
) -> tuple[list[list[arb]], bool]:
    """Intersect two rigorous matrix enclosures entrywise."""
    P = len(first)
    result = [[arb(0) for _ in range(P)] for _ in range(P)]
    overlap = True
    for i in range(P):
        for j in range(P):
            if not first[i][j].overlaps(second[i][j]):
                overlap = False
                result[i][j] = first[i][j].union(second[i][j])
            else:
                result[i][j] = first[i][j].intersection(second[i][j])
    return result, overlap


def _matrix_contains(inner: list[list[arb]], outer: list[list[arb]]) -> bool:
    """Return whether every entry of outer contains the corresponding inner ball."""
    return all(
        outer[i][j].contains(inner[i][j])
        for i in range(len(inner))
        for j in range(len(inner))
    )


def _ordinary_matrix_contained(matrix: list[list[arb]], ordinary: np.ndarray) -> tuple[bool, float]:
    failures = 0
    worst_distance = 0.0
    for i in range(ordinary.shape[0]):
        for j in range(ordinary.shape[1]):
            point = arb(float(ordinary[i, j]))
            if not matrix[i][j].contains(point):
                failures += 1
                lo = float(matrix[i][j].lower())
                hi = float(matrix[i][j].upper())
                worst_distance = max(worst_distance, lo - ordinary[i, j], ordinary[i, j] - hi)
    return failures == 0, worst_distance


def _matrix_radius_infinity_norm(matrix: list[list[arb]]) -> arb:
    eta = arb(0)
    for row in matrix:
        row_sum = arb(0)
        for value in row:
            row_sum += arb(value.rad())
        eta = eta.max(row_sum)
    return eta


def certified_weyl_eigen_intervals(matrix: list[list[arb]]) -> tuple[list[arb], dict[str, object]]:
    """Certify midpoint eigenvalues with Rump and add a rigorous interval-matrix Weyl radius."""
    P = len(matrix)
    midpoint_matrix = acb_mat([[acb(matrix[i][j].mid()) for j in range(P)] for i in range(P)])
    try:
        eigenvalues = midpoint_matrix.eig(algorithm="rump")
        clustered = False
    except ValueError:
        eigenvalues = midpoint_matrix.eig(algorithm="rump", multiple=True)
        clustered = True
    eigenvalues = sorted(eigenvalues, key=lambda z: float(z.real.mid()), reverse=True)

    eta = _matrix_radius_infinity_norm(matrix)
    padding = arb(0, eta.abs_upper())
    intervals = [value.real + padding for value in eigenvalues]
    finite = all(value.is_finite() for value in intervals)

    clusters: list[list[int]] = []
    if intervals:
        current = [0]
        for i in range(1, len(intervals)):
            if intervals[i - 1].overlaps(intervals[i]):
                current.append(i)
            else:
                clusters.append(current)
                current = [i]
        clusters.append(current)

    return intervals, {
        "midpoint_eigensolver": "acb_mat.eig(algorithm='rump')",
        "midpoint_solver_required_multiple_mode": clustered,
        "weyl_radius_infinity_bound": _ball_record(eta),
        "clusters_descending_indices": clusters,
        "all_intervals_finite": finite,
    }


def _matrix_records(matrix: list[list[arb]]) -> list[list[dict[str, str]]]:
    return [[_ball_record(value) for value in row] for row in matrix]


def run_pilot(n: int, c: int, P: int, coarse_cells: int) -> dict[str, object]:
    ctx.prec = BALL_BITS
    mp.mp.dps = REFERENCE_DPS
    started = time.perf_counter()
    roots, root_records = certified_bessel_zeros(n, P)

    I0, D0, direct0, coarse_meta = interval_riemann_ID(n, c, roots, coarse_cells)
    I1, D1, direct1, refined_meta = interval_riemann_ID(n, c, roots, 2 * coarse_cells)
    scalar_refinement_contained = all(
        I0[p].contains(I1[p]) and D0[p].contains(D1[p]) for p in range(P)
    )

    partial0 = assemble_interval_matrix(roots, I0, D0)
    partial1 = assemble_interval_matrix(roots, I1, D1)
    coarse_matrix, coarse_routes_overlap = intersect_interval_matrices(direct0, partial0)
    refined_matrix, refined_routes_overlap = intersect_interval_matrices(direct1, partial1)
    direct_refinement_contained = _matrix_contains(direct1, direct0)
    partial_refinement_contained = _matrix_contains(partial1, partial0)
    matrix_refinement_contained = _matrix_contains(refined_matrix, coarse_matrix)

    ordinary_B, cache = assemble(n, float(c), P=P, G=48, return_cache=True)
    ordinary_A = assemble_routeA(n, float(c), P, cache)
    contains_B, distance_B = _ordinary_matrix_contained(refined_matrix, ordinary_B)
    contains_A, distance_A = _ordinary_matrix_contained(refined_matrix, ordinary_A)

    eigen_intervals, eigen_meta = certified_weyl_eigen_intervals(refined_matrix)
    ordinary_eigenvalues = np.linalg.eigvalsh(0.5 * (ordinary_B + ordinary_B.T))[::-1]
    ordinary_eigenvalues_contained = all(
        eigen_intervals[m].contains(arb(float(ordinary_eigenvalues[m]))) for m in range(P)
    )
    exact_symmetry = all(
        refined_matrix[i][j].contains(refined_matrix[j][i])
        and refined_matrix[j][i].contains(refined_matrix[i][j])
        for i in range(P)
        for j in range(P)
    )

    all_passed = all(
        [
            all(bool(record["passed"]) for record in root_records),
            scalar_refinement_contained,
            direct_refinement_contained,
            partial_refinement_contained,
            matrix_refinement_contained,
            coarse_routes_overlap,
            refined_routes_overlap,
            contains_A,
            contains_B,
            bool(eigen_meta["all_intervals_finite"]),
            ordinary_eigenvalues_contained,
            exact_symmetry,
        ]
    )

    result: dict[str, object] = {
        "task": "25.4 Phase 3 interval Ritz pilot",
        "evidence_label": "certified numerical pilot",
        "parameter_path": "fixed continuous c",
        "n": n,
        "c": c,
        "P": P,
        "ball_precision_bits": BALL_BITS,
        "quadrature": "uniform interval range-sum; no omitted remainder",
        "coarse": coarse_meta,
        "refined": refined_meta,
        "checks": {
            "all_bessel_zeros_uniquely_enclosed": all(bool(record["passed"]) for record in root_records),
            "refined_scalar_integrals_contained_in_coarse": scalar_refinement_contained,
            "refined_direct_Gram_contained_in_coarse": direct_refinement_contained,
            "refined_partial_fraction_matrix_contained_in_coarse": partial_refinement_contained,
            "refined_matrix_contained_in_coarse": matrix_refinement_contained,
            "coarse_certified_routes_overlap": coarse_routes_overlap,
            "refined_certified_routes_overlap": refined_routes_overlap,
            "ordinary_route_A_matrix_contained": contains_A,
            "ordinary_route_B_matrix_contained": contains_B,
            "ordinary_route_A_worst_outside_distance": distance_A,
            "ordinary_route_B_worst_outside_distance": distance_B,
            "exact_saved_symmetry": exact_symmetry,
            "finite_indexed_Ritz_intervals": bool(eigen_meta["all_intervals_finite"]),
            "ordinary_Ritz_eigenvalues_contained": ordinary_eigenvalues_contained,
        },
        "bessel_zero_certificates": root_records,
        "refined_scalar_I": [_ball_record(value) for value in I1],
        "refined_scalar_D": [_ball_record(value) for value in D1],
        "refined_interval_matrix": _matrix_records(refined_matrix),
        "eigenvalue_method": eigen_meta,
        "Ritz_intervals_descending": [_ball_record(value) for value in eigen_intervals],
        "ordinary_Ritz_eigenvalues_descending": ordinary_eigenvalues.tolist(),
        "elapsed_seconds_total": time.perf_counter() - started,
        "all_passed": all_passed,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--c", type=int, required=True)
    parser.add_argument("--P", type=int, required=True)
    parser.add_argument("--coarse-cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_pilot(args.n, args.c, args.P, args.coarse_cells)
    rendered = json.dumps(result, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    # Hash the bytes actually written: pathlib applies the platform newline
    # convention on Windows, so hashing the pre-write LF string is not portable.
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary = {
        "output": os.fspath(args.output),
        "sha256": digest,
        "n": args.n,
        "c": args.c,
        "P": args.P,
        "elapsed_seconds_total": result["elapsed_seconds_total"],
        "checks": result["checks"],
        "eigenvalue_method": result["eigenvalue_method"],
        "all_passed": result["all_passed"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
