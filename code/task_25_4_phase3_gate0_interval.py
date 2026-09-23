"""Task 25.4 Phase 3, Gate 0: validate the rigorous ball-arithmetic stack.

This program does not certify any project Ritz matrix.  It only exercises the
primitives preregistered in Section 2.3 of the Phase 3 protocol:

* a scalar Bessel value;
* a uniquely enclosed positive Bessel zero (interval Newton);
* the removable-quotient identity at that zero;
* a certified Bessel integral; and
* a certified 3 x 3 symmetric eigenproblem.

Every Arb/FLINT ball is checked against a separately recomputed 90-decimal
mpmath value.  Passing this program makes the interval implementation eligible
for the two matrix pilots; it is not by itself a matrix or CPSWF certificate.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import flint
import mpmath as mp
from flint import acb, acb_mat, arb, ctx
from scipy.special import jn_zeros


BALL_BITS = 192
REFERENCE_DPS = 90


def _ball_record(x: arb) -> dict[str, str]:
    """Return human-readable ball and outward endpoint representations."""
    return {
        "ball": str(x),
        "lower": x.lower().str(80, radius=False),
        "upper": x.upper().str(80, radius=False),
        "radius": x.rad().str(20, radius=False),
    }


def _mp_string(x: mp.mpf | mp.mpc) -> str:
    return mp.nstr(x, REFERENCE_DPS)


def _contains_mpf(x: arb, y: mp.mpf) -> bool:
    # Compare with the nearest 192-bit point, not with the auxiliary radius
    # introduced when Arb parses a decimal string.  The source has 90 decimal
    # digits versus about 58 digits in the target balls.
    independent_point = arb(_mp_string(y)).mid()
    return bool(x.contains(independent_point))


def _bessel_derivative(n: int, x: arb | acb) -> arb | acb:
    return (x.bessel_j(n - 1) - x.bessel_j(n + 1)) / 2


def _strict_sign(x: arb) -> int:
    """Return +1/-1 when the whole ball has that sign, and 0 otherwise."""
    if x.lower() > 0:
        return 1
    if x.upper() < 0:
        return -1
    return 0


def _certified_bessel_zero(n: int, k: int) -> tuple[arb, dict[str, object]]:
    """Enclose one positive zero and prove local existence and uniqueness.

    SciPy supplies only a seed.  The certificate is the interval-Newton
    inclusion N(X) subset int(X), together with a sign change at the endpoints.
    The seed interval is far narrower than the separation of consecutive Bessel
    zeros, and the requested index is retained only after these directed tests.
    """
    seed = float(jn_zeros(n, k)[-1])
    X = arb(format(seed, ".17g"), "1e-10")
    midpoint = X.mid()
    derivative_X = _bessel_derivative(n, X)
    if derivative_X.contains(0):
        raise RuntimeError("interval derivative contains zero")

    newton = midpoint - midpoint.bessel_j(n) / derivative_X
    newton_interior = bool(X.contains_interior(newton))
    left_value = X.lower().bessel_j(n)
    right_value = X.upper().bessel_j(n)
    left_sign = _strict_sign(left_value)
    right_sign = _strict_sign(right_value)
    sign_change = bool(left_sign * right_sign == -1)

    # A second interval-Newton step makes the saved zero ball compact while
    # retaining the first strict inclusion as the existence/uniqueness proof.
    X1 = X.intersection(newton)
    derivative_X1 = _bessel_derivative(n, X1)
    newton2 = X1.mid() - X1.mid().bessel_j(n) / derivative_X1
    refined_interior = bool(X1.contains_interior(newton2))
    root = X1.intersection(newton2)

    reference = mp.besseljzero(n, k)
    contains_reference = _contains_mpf(root, reference)
    passed = newton_interior and refined_interior and sign_change and contains_reference
    details: dict[str, object] = {
        "name": "bessel_zero_interval_newton",
        "n": n,
        "k": k,
        "seed_only_scipy": format(seed, ".17g"),
        "initial_interval": _ball_record(X),
        "root_interval": _ball_record(root),
        "left_sign_certified": {1: "positive", -1: "negative", 0: "unresolved"}[left_sign],
        "right_sign_certified": {1: "positive", -1: "negative", 0: "unresolved"}[right_sign],
        "interval_newton_strict_inclusion": newton_interior,
        "refined_interval_newton_strict_inclusion": refined_interior,
        "endpoint_sign_change": sign_change,
        "mpmath_reference": _mp_string(reference),
        "contains_independent_reference": contains_reference,
        "passed": passed,
    }
    return root, details


def _scalar_bessel_test() -> dict[str, object]:
    n = 2
    x_text = "1.25"
    ball = arb(x_text).bessel_j(n)
    reference = mp.besselj(n, mp.mpf(x_text))
    contains = _contains_mpf(ball, reference)
    return {
        "name": "scalar_bessel_value",
        "n": n,
        "x": x_text,
        "enclosure": _ball_record(ball),
        "mpmath_reference": _mp_string(reference),
        "contains_independent_reference": contains,
        "passed": contains and ball.is_finite(),
    }


def _removable_quotient_test(n: int, k: int, root: arb) -> dict[str, object]:
    # Both copies contain the exact zero.  The identity stays finite even though
    # the two interval objects do not preserve their equality as a dependency.
    j = acb(root)
    u = acb(root)
    numerator = acb.integral(
        lambda t, _analytic: _bessel_derivative(n, j + t * (u - j)),
        0,
        1,
    )
    quotient = numerator / (u + j)

    j_ref = mp.besseljzero(n, k)
    derivative_ref = (mp.besselj(n - 1, j_ref) - mp.besselj(n + 1, j_ref)) / 2
    reference = derivative_ref / (2 * j_ref)
    contains_real = _contains_mpf(quotient.real, reference)
    contains_zero_imag = bool(quotient.imag.contains(0))
    passed = quotient.is_finite() and contains_real and contains_zero_imag
    return {
        "name": "removable_quotient_identity",
        "n": n,
        "k": k,
        "evaluation": "u=j_{n,k}",
        "real_enclosure": _ball_record(quotient.real),
        "imaginary_enclosure": _ball_record(quotient.imag),
        "mpmath_reference": _mp_string(reference),
        "contains_independent_reference": contains_real,
        "imaginary_part_contains_zero": contains_zero_imag,
        "passed": passed,
    }


def _integral_test() -> dict[str, object]:
    integral = acb.integral(lambda x, _analytic: x.bessel_j(0), 0, 1)
    # Independent closed hypergeometric representation of integral_0^1 J_0(x) dx.
    reference = mp.hyper([mp.mpf(1) / 2], [1, mp.mpf(3) / 2], -mp.mpf(1) / 4)
    contains_real = _contains_mpf(integral.real, reference)
    contains_zero_imag = bool(integral.imag.contains(0))
    return {
        "name": "certified_bessel_integral",
        "integrand": "J_0(x)",
        "interval": "[0,1]",
        "real_enclosure": _ball_record(integral.real),
        "imaginary_enclosure": _ball_record(integral.imag),
        "mpmath_reference": _mp_string(reference),
        "contains_independent_reference": contains_real,
        "imaginary_part_contains_zero": contains_zero_imag,
        "passed": integral.is_finite() and contains_real and contains_zero_imag,
    }


def _eigenvalue_test() -> dict[str, object]:
    # Exact integer matrix; exact eigenvalues are 2-sqrt(2), 2, 2+sqrt(2).
    matrix = acb_mat([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
    eigenvalues = matrix.eig(algorithm="rump")
    eigenvalues = sorted(eigenvalues, key=lambda z: float(z.real.mid()))
    references = [mp.mpf(2) - mp.sqrt(2), mp.mpf(2), mp.mpf(2) + mp.sqrt(2)]

    rows = []
    passed = True
    for index, (value, reference) in enumerate(zip(eigenvalues, references)):
        contains_real = _contains_mpf(value.real, reference)
        contains_zero_imag = bool(value.imag.contains(0))
        row_passed = value.is_finite() and contains_real and contains_zero_imag
        passed = passed and row_passed
        rows.append(
            {
                "ascending_index": index,
                "real_enclosure": _ball_record(value.real),
                "imaginary_enclosure": _ball_record(value.imag),
                "mpmath_reference": _mp_string(reference),
                "contains_independent_reference": contains_real,
                "imaginary_part_contains_zero": contains_zero_imag,
                "passed": row_passed,
            }
        )

    return {
        "name": "certified_3_by_3_symmetric_eigenproblem",
        "matrix": [[2, 1, 0], [1, 2, 1], [0, 1, 2]],
        "algorithm": "acb_mat.eig(algorithm='rump')",
        "eigenvalues_ascending": rows,
        "passed": passed,
    }


def run_gate0() -> dict[str, object]:
    ctx.prec = BALL_BITS
    mp.mp.dps = REFERENCE_DPS

    scalar = _scalar_bessel_test()
    root, zero = _certified_bessel_zero(n=2, k=3)
    removable = _removable_quotient_test(n=2, k=3, root=root)
    integral = _integral_test()
    eigenproblem = _eigenvalue_test()
    tests = [scalar, zero, removable, integral, eigenproblem]

    version_fields = {
        name: getattr(flint, name)
        for name in ("__version__", "__FLINT_VERSION__", "__ARB_VERSION__")
        if hasattr(flint, name)
    }
    return {
        "task": "25.4 Phase 3 Gate 0 interval-stack validation",
        "evidence_scope": "implementation primitive tests only; no Ritz/CPSWF certification",
        "implementation": "python-flint Arb/FLINT ball arithmetic",
        "python": sys.version,
        "platform": platform.platform(),
        "versions": version_fields,
        "ball_precision_bits": BALL_BITS,
        "independent_reference": f"mpmath at {REFERENCE_DPS} decimal digits",
        "tests": tests,
        "all_passed": all(bool(test["passed"]) for test in tests),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    result = run_gate0()
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
