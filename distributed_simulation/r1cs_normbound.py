"""
r1cs_normbound.py — SecureFedHE
R1CS matrices for the scoped norm-bound circuit (Option 2 / Path B).

Proves: given a private gradient of dimension n,
    sum(g_i^2) + slack = norm_sq_bound
Does NOT prove the hash commitment — that is checked outside the SNARK,
same as today's slack >= 0 check (honest-prover-side, not in-circuit).

Witness layout (column order):
  [0]             = 1 (constant)                    public
  [1]             = norm_sq_bound                    public
  [2]             = round_num                        public
  [3 .. 3+n-1]    = gradient values g_0..g_{n-1}      private
  [3+n .. 3+2n-1] = squares sq_0..sq_{n-1}            private
  [3+2n]          = slack                             private

n_public  = 3   (constant counts as public in Groth16 convention)
n_private = 2n + 1
n_rows    = n + 1   (n square constraints + 1 sum/bound constraint)
"""
from typing import Dict, List, Tuple


def build_r1cs(n: int) -> Tuple[List[Dict[int, int]], List[Dict[int, int]], List[Dict[int, int]], int, int]:
    """
    Returns A, B, C — each a list of rows, each row a sparse dict {col_index: coeff}.
    Satisfies: for every row r, (A[r] . w) * (B[r] . w) == (C[r] . w)
    """
    n_cols = 3 + 2 * n + 1
    n_rows = n + 1

    A = [dict() for _ in range(n_rows)]
    B = [dict() for _ in range(n_rows)]
    C = [dict() for _ in range(n_rows)]

    def g(i):  return 3 + i
    def sq(i): return 3 + n + i
    SLACK = 3 + 2 * n
    CONST = 0
    BOUND = 1

    # Rows 0..n-1: g_i * g_i = sq_i
    for i in range(n):
        A[i][g(i)] = 1
        B[i][g(i)] = 1
        C[i][sq(i)] = 1

    # Row n: 1 * (sum(sq_i) + slack - bound) = 0
    row = n
    A[row][CONST] = 1
    for i in range(n):
        B[row][sq(i)] = 1
    B[row][SLACK] = 1
    B[row][BOUND] = -1
    # C[row] stays all-zero (empty dict = zero row)

    return A, B, C, n_cols, n_rows


def build_witness(gradient_fr: List[int], slack: int, norm_sq_bound: int,
                   round_num: int, Fr: int) -> List[int]:
    """
    Build the full witness vector w matching the column layout in build_r1cs,
    reduced mod Fr. gradient_fr must already be quantized Fr elements.
    """
    n = len(gradient_fr)
    sq = [(gi * gi) % Fr for gi in gradient_fr]
    w = [1, norm_sq_bound % Fr, round_num % Fr]
    w += [gi % Fr for gi in gradient_fr]
    w += sq
    w += [slack % Fr]
    assert len(w) == 3 + 2 * n + 1
    return w


def check_r1cs(A, B, C, w: List[int], Fr: int) -> bool:
    """Verify every row satisfies (A.w)*(B.w) == (C.w) mod Fr."""
    for r in range(len(A)):
        av = sum(coeff * w[col] for col, coeff in A[r].items()) % Fr
        bv = sum(coeff * w[col] for col, coeff in B[r].items()) % Fr
        cv = sum(coeff * w[col] for col, coeff in C[r].items()) % Fr
        if (av * bv) % Fr != cv:
            print(f"  Row {r} FAILED: A.w={av} B.w={bv} A.w*B.w={(av*bv)%Fr} C.w={cv}")
            return False
    return True


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from distributed_simulation.zkp_math import Fr as FR, quantize, norm_sq_int, threshold_sq_int

    print("=" * 60)
    print("  R1CS Norm-Bound Circuit — Self Test")
    print("=" * 60)

    # --- Small structural test, n=3 ---
    n = 3
    A, B, C, n_cols, n_rows = build_r1cs(n)
    print(f"\n[Structural] n={n}  n_cols={n_cols}  n_rows={n_rows}")
    print("A:", A)
    print("B:", B)
    print("C:", C)

    # --- Real-scale test, n=32 (matches actual fc2 gradient_dim) ---
    n = 32
    C_thresh = 0.5
    A, B, C, n_cols, n_rows = build_r1cs(n)
    print(f"\n[Real scale] n={n}  n_cols={n_cols}  n_rows={n_rows}  "
          f"(expected n_cols={3+2*n+1}, n_rows={n+1})")
    assert n_cols == 3 + 2 * n + 1
    assert n_rows == n + 1

    # Valid gradient (within norm bound)
    import random
    random.seed(0)
    grad = [random.uniform(-0.05, 0.05) for _ in range(n)]
    grad_fr = quantize(grad)
    ns = norm_sq_int(grad_fr)
    bound = threshold_sq_int(C_thresh)
    slack = bound - ns
    print(f"\n[Valid witness] norm_sq={ns}  bound={bound}  slack={slack}  "
          f"(slack>=0: {slack >= 0})")
    assert slack >= 0, "test gradient exceeds bound, adjust random range"

    w = build_witness(grad_fr, slack, bound, round_num=1, Fr=FR)
    ok = check_r1cs(A, B, C, w, FR)
    print(f"[Valid witness] R1CS check: {'PASS' if ok else 'FAIL'}")
    assert ok

    # Invalid witness: tampered gradient (should fail R1CS check)
    w_bad = list(w)
    w_bad[3] = (w_bad[3] + 12345) % FR  # corrupt g_0 without updating sq_0
    ok_bad = check_r1cs(A, B, C, w_bad, FR)
    print(f"[Tampered witness] R1CS check (expect FAIL): {'PASS' if ok_bad else 'FAIL'}")
    assert not ok_bad, "tampered witness should NOT satisfy constraints"

    # Invalid witness: slack forged negative-equivalent via wraparound (should fail)
    w_bad2 = list(w)
    w_bad2[3 + 2 * n] = (w_bad2[3 + 2 * n] - slack - 1) % FR  # break the sum+bound row
    ok_bad2 = check_r1cs(A, B, C, w_bad2, FR)
    print(f"[Forged slack] R1CS check (expect FAIL): {'PASS' if ok_bad2 else 'FAIL'}")
    assert not ok_bad2, "forged slack should NOT satisfy constraints"

    print("\nAll R1CS self-tests passed.")
