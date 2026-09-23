"""Freeze the Task 25.4 Phase 3 reference and mode-selection manifest.

Only ordinary Nyström reference calculations and the pre-existing saved
benchmarks are read.  No project-matrix or Boulsane error is computed here.
The selection rules are exactly those accepted in Phase 3 Protocol Sections
3--4.  Changing the resulting manifest requires a dated protocol amendment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg as sla
from scipy.special import jn_zeros, jv, jvp


ORDERS = (0, 1, 2, 4)
BANDLIMITS = (10, 20, 40, 80)
EPS = np.finfo(float).eps


def shannon_count(n: int, c: float) -> int:
    return int(np.sum(jn_zeros(n, int(c / np.pi) + 30) < c))


def kernel_matrix(n: int, c: float, x: np.ndarray) -> np.ndarray:
    X = x[:, None]
    Y = x[None, :]
    cX = c * X
    cY = c * Y
    JX = jv(n, cX)
    JY = jv(n, cY)
    JpX = jvp(n, cX)
    JpY = jvp(n, cY)
    denominator = Y * Y - X * X
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = c * (X * JY * JpX - Y * JX * JpY) / denominator
    cx = c * x
    diagonal = (c * c / 2) * (
        jvp(n, cx) ** 2 + (1 - (n / cx) ** 2) * jv(n, cx) ** 2
    )
    np.fill_diagonal(kernel, diagonal)
    return kernel


def nystrom_A(
    n: int, c: float, Q: int, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, w = np.polynomial.legendre.leggauss(Q)
    x = 0.5 * (t + 1)
    w = 0.5 * w
    scale = np.sqrt(w * x)
    matrix = (scale[:, None] * kernel_matrix(n, c, x)) * scale[None, :]
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = sla.eigh(
        matrix,
        subset_by_index=[Q - count, Q - 1],
        driver="evr",
        check_finite=False,
    )
    values = values[::-1]
    vectors = vectors[:, ::-1]
    functions = vectors / scale[:, None]
    return values, x, w, functions


def nystrom_B(n: int, c: float, Q: int, count: int) -> np.ndarray:
    t, omega = np.polynomial.legendre.leggauss(Q)
    u = 0.5 * (t + 1)
    omega = 0.5 * omega
    x = np.sqrt(u)
    weights = omega / 2
    scale = np.sqrt(weights)
    matrix = (scale[:, None] * kernel_matrix(n, c, x)) * scale[None, :]
    matrix = 0.5 * (matrix + matrix.T)
    values = sla.eigvalsh(
        matrix,
        subset_by_index=[Q - count, Q - 1],
        driver="evr",
        check_finite=False,
    )
    return values[::-1]


def endpoint_extension(
    n: int,
    c: float,
    eigenvalue: float,
    x: np.ndarray,
    w: np.ndarray,
    function: np.ndarray,
) -> float:
    """Nyström extension to x=1; no polynomial extrapolation."""
    cx = c * x
    JY = jv(n, cx)
    JpY = jvp(n, cx)
    J1 = jv(n, c)
    Jp1 = jvp(n, c)
    denominator = x * x - 1
    row = c * (JY * Jp1 - x * J1 * JpY) / denominator
    return float(np.sum(row * function * x * w) / eigenvalue)


def local_gap(values: np.ndarray, m: int) -> float:
    candidates = []
    if m > 0:
        candidates.append(float(values[m - 1] - values[m]))
    if m + 1 < len(values):
        candidates.append(float(values[m] - values[m + 1]))
    return min(candidates) if candidates else math.inf


def mode_record(
    category: str,
    m: int | None,
    values: np.ndarray,
    uncertainties: np.ndarray,
) -> dict[str, object]:
    if m is None:
        return {"category": category, "available": False, "m": None}
    return {
        "category": category,
        "available": True,
        "m": int(m),
        "lambda_ref": float(values[m]),
        "u_ref": float(uncertainties[m]),
        "local_gap": local_gap(values, m),
        "isolated_at_10_u_ref": bool(local_gap(values, m) > 10 * uncertainties[m]),
    }


def build_case(n: int, c: int, benchmark_dir: Path) -> dict[str, object]:
    M = shannon_count(n, float(c))
    count = M + 6
    Q0 = max(240, 4 * c)
    Q1 = Q0 + 80
    Q2 = Q0 + 160

    A0, x0, w0, psi0 = nystrom_A(n, float(c), Q0, count)
    A1, x1, w1, psi1 = nystrom_A(n, float(c), Q1, count)
    A2, x2, w2, psi2 = nystrom_A(n, float(c), Q2, count)
    B1 = nystrom_B(n, float(c), Q1, count)
    B2 = nystrom_B(n, float(c), Q2, count)

    uncertainty = 10 * np.maximum.reduce(
        [np.abs(A2 - A1), np.abs(A2 - B2), np.full(count, 50 * EPS)]
    )

    plateau = np.flatnonzero(1 - A2 <= 1e-6)
    if len(plateau) >= 2:
        deep = 0
        edge = int(plateau[-1])
    elif len(plateau) == 1:
        deep = None
        edge = int(plateau[0])
    else:
        deep = None
        edge = None

    candidate_stop = M + 4
    knee = int(np.argmin(np.abs(A2[: candidate_stop + 1] - 0.5)))
    tail = next(
        (
            m
            for m in range(knee + 1, candidate_stop + 1)
            if A2[m] <= 1e-3 and A2[m] >= 100 * uncertainty[m]
        ),
        None,
    )

    endpoint1 = endpoint_extension(n, float(c), A1[knee], x1, w1, psi1[:, knee])
    endpoint2 = endpoint_extension(n, float(c), A2[knee], x2, w2, psi2[:, knee])
    endpoint_abs = abs(endpoint2)
    endpoint_uncertainty = 10 * abs(abs(endpoint2) - abs(endpoint1))

    benchmark_path = benchmark_dir / f"cpswf_n{n}_c{c}.npz"
    if not benchmark_path.exists():
        raise FileNotFoundError(benchmark_path)
    benchmark = np.load(benchmark_path)
    benchmark_values = benchmark["lam"]
    if len(benchmark_values) < candidate_stop + 1:
        raise RuntimeError(
            f"saved benchmark {benchmark_path} does not cover modes 0..M+4"
        )
    benchmark_overlap = min(count, len(benchmark_values))

    candidates = []
    for m in range(candidate_stop + 1):
        candidates.append(
            {
                "m": m,
                "lambda_ref": float(A2[m]),
                "u_ref": float(uncertainty[m]),
                "local_gap": local_gap(A2, m),
                "A_Q2_minus_A_Q1": float(A2[m] - A1[m]),
                "A_Q2_minus_B_Q2": float(A2[m] - B2[m]),
                "B_Q2_minus_B_Q1": float(B2[m] - B1[m]),
                "A_Q2_minus_saved_benchmark": float(A2[m] - benchmark_values[m]),
            }
        )

    selected = [
        mode_record("deep", deep, A2, uncertainty),
        mode_record("edge", edge, A2, uncertainty),
        mode_record("knee", knee, A2, uncertainty),
        mode_record("tail", tail, A2, uncertainty),
    ]
    return {
        "n": n,
        "c": c,
        "parameter_path": "fixed continuous c",
        "M_n_c": M,
        "quadrature_orders": {"Q0": Q0, "Q1": Q1, "Q2": Q2},
        "reference_central": "Nyström Solver A on x-grid at Q2",
        "uncertainty_formula": "10*max(|A_Q2-A_Q1|,|A_Q2-B_Q2|,50*eps_machine)",
        "plateau_indices": plateau.tolist(),
        "selected_modes": selected,
        "knee_endpoint": {
            "m": knee,
            "method": "Nyström integral-equation extension at x=1",
            "absolute_value_Q2": endpoint_abs,
            "inflated_Q1_Q2_uncertainty": endpoint_uncertainty,
            "resolved_at_10_times_uncertainty": bool(
                endpoint_abs > 10 * endpoint_uncertainty
            ),
        },
        "candidate_modes_0_through_M_plus_4": candidates,
        "reference_diagnostics": {
            "max_abs_A_Q2_minus_A_Q1": float(np.max(np.abs(A2 - A1))),
            "max_abs_A_Q2_minus_B_Q2": float(np.max(np.abs(A2 - B2))),
            "max_abs_B_Q2_minus_B_Q1": float(np.max(np.abs(B2 - B1))),
            "max_abs_A_Q2_minus_saved_benchmark": float(
                np.max(
                    np.abs(
                        A2[:benchmark_overlap] - benchmark_values[:benchmark_overlap]
                    )
                )
            ),
            "saved_benchmark_overlap_count": benchmark_overlap,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/task_25_4_phase3/reference_mode_manifest.json"),
    )
    args = parser.parse_args()

    started = time.perf_counter()
    cases = []
    for n in ORDERS:
        for c in BANDLIMITS:
            case_started = time.perf_counter()
            case = build_case(n, c, args.benchmark_dir)
            case["elapsed_seconds"] = time.perf_counter() - case_started
            cases.append(case)
            selected = {row["category"]: row["m"] for row in case["selected_modes"]}
            print(f"n={n} c={c}: M={case['M_n_c']} selected={selected}", flush=True)

    manifest = {
        "task": "25.4 Phase 3 frozen reference and mode manifest",
        "status": "FROZEN before project/Boulsane campaign errors",
        "date_frozen": "2026-08-24",
        "evidence_label": "ordinary numerical reference and preregistered selection",
        "parameter_path": "fixed continuous c",
        "orders": list(ORDERS),
        "bandlimits": list(BANDLIMITS),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "machine_epsilon": EPS,
        "case_count": len(cases),
        "cases": cases,
        "elapsed_seconds_total": time.perf_counter() - started,
    }

    rendered = json.dumps(manifest, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    hash_path = args.output.with_suffix(args.output.suffix + ".sha256")
    hash_path.write_text(f"{digest}  {args.output.name}\n", encoding="ascii")
    print(f"manifest={args.output}")
    print(f"sha256={digest}")
    print(f"hash_file={hash_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
