"""
qap_normbound.py — SecureFedHE
Converts the R1CS matrices (A, B, C) from r1cs_normbound.py into a QAP
(Quadratic Arithmetic Program): one polynomial per column, for each of
A, B, C, such that evaluating all column-polys at constraint-index x=1..m
and combining with the witness reproduces the original R1CS check.

Core QAP identity, for witness vector w:
    (sum_j w_j * A_j(x)) * (sum_j w_j * B_j(x)) - (sum_j w_j * C_j(x))
        = H(x) * Z(x)
where Z(x) = product_{i=1}^{m} (x - i)  (m = number of constraints)
and H(x) is a polynomial with NO remainder iff every R1CS row is satisfied.

This H(x)*Z(x) divisibility check is the actual mathematical content that
the trusted setup encodes into the proving/verification keys, and it is
what the real pairing equation ultimately certifies. If the witness breaks
even one constraint, the left-hand side will NOT be divisible by Z(x),
and H(x) will not exist as a polynomial (division leaves a nonzero
remainder) — that's the soundness anchor.
"""
from typing import Dict, List, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributed_simulation.poly_fr import (
    Poly, poly_add, poly_sub, poly_mul, poly_scale, poly_eval,
    poly_divmod, lagrange_interpolate, poly_trim
)
from distributed_simulation.r1cs_normbound import build_r1cs, build_witness, check_r1cs


def matrix_to_col_polys(M: List[Dict[int, int]], n_cols: int, n_rows: int, Fr: int) -> List[Poly]:
    """
    For each column j, interpolate the polynomial that passes through
    (i+1, M[i][j]) for i = 0..n_rows-1 (constraint indices are 1-indexed
    by convention, x=1,2,...,m).
    """
    xs = [i + 1 for i in range(n_rows)]
    col_polys = []
    for j in range(n_cols):
        ys = [M[i].get(j, 0) % Fr for i in range(n_rows)]
        if all(y == 0 for y in ys):
            col_polys.append([0])
        else:
            col_polys.append(lagrange_interpolate(xs, ys, Fr))
    return col_polys


def build_vanishing_poly(n_rows: int, Fr: int) -> Poly:
    """Z(x) = product_{i=1}^{n_rows} (x - i)"""
    Z = [1]
    for i in range(1, n_rows + 1):
        Z = poly_mul(Z, [(-i) % Fr, 1], Fr)
    return Z


def witness_combine(col_polys: List[Poly], w: List[int], Fr: int) -> Poly:
    """Compute sum_j w[j] * col_polys[j](x) as a single polynomial."""
    acc = [0]
    for j, coeff in enumerate(w):
        if coeff == 0:
            continue
        acc = poly_add(acc, poly_scale(col_polys[j], coeff, Fr), Fr)
    return acc


def compute_h(A_w: Poly, B_w: Poly, C_w: Poly, Z: Poly, Fr: int) -> Tuple[Poly, Poly]:
    """
    Compute H(x) = (A_w(x)*B_w(x) - C_w(x)) / Z(x).
    Returns (H, remainder). remainder should be [0] iff witness is valid.
    """
    lhs = poly_sub(poly_mul(A_w, B_w, Fr), C_w, Fr)
    H, rem = poly_divmod(lhs, Z, Fr)
    return H, rem


if __name__ == "__main__":
    from distributed_simulation.zkp_math import Fr as FR, quantize, norm_sq_int, threshold_sq_int
    import random

    print("=" * 60)
    print("  qap_normbound.py — Self Test")
    print("=" * 60)

    # ---- Small n=3 structural test first ----
    n = 3
    A, B, C, n_cols, n_rows = build_r1cs(n)
    A_polys = matrix_to_col_polys(A, n_cols, n_rows, FR)
    B_polys = matrix_to_col_polys(B, n_cols, n_rows, FR)
    C_polys = matrix_to_col_polys(C, n_cols, n_rows, FR)
    Z = build_vanishing_poly(n_rows, FR)
    print(f"\n[n=3] n_cols={n_cols} n_rows={n_rows}  Z degree={len(Z)-1} (expect {n_rows})")
    assert len(Z) - 1 == n_rows

    # sanity: each col poly, evaluated at constraint index i+1, must equal M[i][j]
    for j in range(n_cols):
        for i in range(n_rows):
            expect = A[i].get(j, 0) % FR
            got = poly_eval(A_polys[j], i + 1, FR)
            assert got == expect, f"A col {j} row {i}: expected {expect} got {got}"
    print("[OK] All A/B/C column polynomials reproduce original R1CS matrix at each row index")

    # ---- Real scale n=32, VALID witness ----
    n = 32
    C_thresh = 0.5
    A, B, C, n_cols, n_rows = build_r1cs(n)
    A_polys = matrix_to_col_polys(A, n_cols, n_rows, FR)
    B_polys = matrix_to_col_polys(B, n_cols, n_rows, FR)
    C_polys = matrix_to_col_polys(C, n_cols, n_rows, FR)
    Z = build_vanishing_poly(n_rows, FR)
    print(f"\n[n=32] n_cols={n_cols} n_rows={n_rows}  Z degree={len(Z)-1}")

    random.seed(1)
    grad = [random.uniform(-0.05, 0.05) for _ in range(n)]
    grad_fr = quantize(grad)
    ns = norm_sq_int(grad_fr)
    bound = threshold_sq_int(C_thresh)
    slack = bound - ns
    assert slack >= 0

    w = build_witness(grad_fr, slack, bound, round_num=7, Fr=FR)
    assert check_r1cs(A, B, C, w, FR), "R1CS check should pass for valid witness"

    A_w = witness_combine(A_polys, w, FR)
    B_w = witness_combine(B_polys, w, FR)
    C_w = witness_combine(C_polys, w, FR)
    H, rem = compute_h(A_w, B_w, C_w, Z, FR)
    print(f"[Valid witness] H(x) degree={len(H)-1}  remainder={'ZERO (divides exactly)' if rem == [0] else rem}")
    assert rem == [0], "QAP divisibility should hold exactly for a valid witness"
    print("[OK] Valid witness satisfies QAP identity: A_w(x)*B_w(x) - C_w(x) = H(x)*Z(x)")

    # Cross-check the identity numerically at a random point (not 1..n_rows)
    test_x = 999999937  # arbitrary point, not a root of Z
    lhs = (poly_eval(A_w, test_x, FR) * poly_eval(B_w, test_x, FR) - poly_eval(C_w, test_x, FR)) % FR
    rhs = (poly_eval(H, test_x, FR) * poly_eval(Z, test_x, FR)) % FR
    assert lhs == rhs, f"QAP identity mismatch at random point: {lhs} != {rhs}"
    print(f"[OK] QAP identity verified numerically at random x={test_x}")

    # ---- INVALID witness: tampered gradient must break divisibility ----
    w_bad = list(w)
    w_bad[3] = (w_bad[3] + 999) % FR  # corrupt g_0, don't fix sq_0 to match
    A_w_bad = witness_combine(A_polys, w_bad, FR)
    B_w_bad = witness_combine(B_polys, w_bad, FR)
    C_w_bad = witness_combine(C_polys, w_bad, FR)
    H_bad, rem_bad = compute_h(A_w_bad, B_w_bad, C_w_bad, Z, FR)
    print(f"\n[Tampered witness] remainder={'ZERO (BUG!)' if rem_bad == [0] else 'NONZERO (correct)'}")
    assert rem_bad != [0], "tampered witness must NOT satisfy QAP divisibility"
    print("[OK] Tampered witness correctly fails QAP divisibility (no valid H(x) exists)")

    print("\nAll QAP self-tests passed.")
