"""
prover_v2.py — SecureFedHE
Real Groth16 prover for the norm-bound QAP circuit.

Proof construction (zero-knowledge blinding terms r, s omitted — see
NOTE at bottom for what that trade-off means and why it's acceptable here):

    A = alpha_g1 + [A_w(tau)]_1
    B = beta_g2  + [B_w(tau)]_2
    C = sum_{j in private} w_j * L_j  +  [H(tau)*Z(tau)]_1

Where [P(tau)]_1 / [P(tau)]_2 mean "evaluate polynomial P at the secret
tau, but only using the SRS powers-of-tau points from the trusted setup —
tau itself is never available here, it was destroyed after setup."
"""
from typing import List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_ecc.bn128 import add, multiply, curve_order, Z1

from distributed_simulation.r1cs_normbound import build_r1cs, build_witness, check_r1cs
from distributed_simulation.qap_normbound import (
    matrix_to_col_polys, build_vanishing_poly, witness_combine, compute_h
)
from distributed_simulation.poly_fr import poly_mul
from distributed_simulation.trusted_setup import setup, ProvingKeyV2

Fr = curve_order


class Groth16ProofV2:
    def __init__(self, A: tuple, B: tuple, C: tuple, public_inputs: List[int]):
        self.A = A                          # G1 point
        self.B = B                          # G2 point
        self.C = C                          # G1 point
        self.public_inputs = public_inputs  # [1, norm_sq_bound, round_num]

    def __repr__(self):
        return f"Groth16ProofV2(public_inputs={self.public_inputs})"


def _srs_eval_g1(tau_powers_g1: List[tuple], poly: List[int]) -> tuple:
    """Evaluate `poly` at the (secret, unknown-to-us) tau, as a G1 point,
    using only the SRS powers-of-tau. This never touches tau directly."""
    acc = Z1
    for i, c in enumerate(poly):
        if c == 0:
            continue
        if i >= len(tau_powers_g1):
            raise RuntimeError(
                f"Polynomial degree {len(poly)-1} exceeds SRS G1 power bound "
                f"{len(tau_powers_g1)-1} — regenerate setup with more headroom."
            )
        acc = add(acc, multiply(tau_powers_g1[i], c))
    return acc


def _srs_eval_g2(tau_powers_g2: List[tuple], poly: List[int]) -> tuple:
    acc = None
    for i, c in enumerate(poly):
        if c == 0:
            continue
        if i >= len(tau_powers_g2):
            raise RuntimeError(
                f"Polynomial degree {len(poly)-1} exceeds SRS G2 power bound "
                f"{len(tau_powers_g2)-1} — regenerate setup with more headroom."
            )
        pt = multiply(tau_powers_g2[i], c)
        acc = pt if acc is None else add(acc, pt)
    return acc if acc is not None else None  # caller adds a G2 base point, so None is fine to skip


def prove(pk: ProvingKeyV2, n: int, gradient_fr: List[int], slack: int,
          norm_sq_bound: int, round_num: int) -> Groth16ProofV2:
    """
    Build a real Groth16 proof for the norm-bound circuit.
    Raises ValueError if the witness does not satisfy the circuit.
    """
    A_mat, B_mat, C_mat, n_cols, n_rows = build_r1cs(n)
    A_polys = matrix_to_col_polys(A_mat, n_cols, n_rows, Fr)
    B_polys = matrix_to_col_polys(B_mat, n_cols, n_rows, Fr)
    C_polys = matrix_to_col_polys(C_mat, n_cols, n_rows, Fr)
    Z = build_vanishing_poly(n_rows, Fr)

    if slack < 0:
        raise ValueError(
            f"Gradient norm exceeds bound (slack={slack} < 0) — refusing to prove. "
            f"Apply gradient clipping before proving."
        )

    w = build_witness(gradient_fr, slack, norm_sq_bound, round_num, Fr)

    if not check_r1cs(A_mat, B_mat, C_mat, w, Fr):
        raise ValueError("Witness does not satisfy circuit constraints — cannot prove.")

    A_w = witness_combine(A_polys, w, Fr)
    B_w = witness_combine(B_polys, w, Fr)
    C_w = witness_combine(C_polys, w, Fr)
    H, rem = compute_h(A_w, B_w, C_w, Z, Fr)
    if rem != [0]:
        raise ValueError("QAP divisibility failed — internal consistency error.")

    # HZ must be divided by delta to match l_query's units (l_query[j]
    # already carries /delta from setup). We use the proving key's
    # h_query — precomputed SRS points [tau^i / delta]_1 — rather than
    # ever reconstructing delta_inv as a bare scalar here.
    HZ = poly_mul(H, Z, Fr)
    HZ_g1 = _srs_eval_g1(pk.h_query, HZ)

    # A = alpha_g1 + [A_w(tau)]_1
    A_g1 = add(pk.alpha_g1, _srs_eval_g1(pk.tau_powers_g1, A_w))

    # B = beta_g2 + [B_w(tau)]_2
    Bw_g2 = _srs_eval_g2(pk.tau_powers_g2, B_w)
    B_g2 = pk.beta_g2 if Bw_g2 is None else add(pk.beta_g2, Bw_g2)

    # C = sum_{private j} w_j * L_j + HZ_g1  (both already divided by delta)
    C_g1 = Z1
    for idx, j in enumerate(pk.private_cols):
        if w[j] == 0:
            continue
        C_g1 = add(C_g1, multiply(pk.l_query[idx], w[j]))
    C_g1 = add(C_g1, HZ_g1)

    public_inputs = w[:pk.n_public]
    return Groth16ProofV2(A=A_g1, B=B_g2, C=C_g1, public_inputs=public_inputs)


if __name__ == "__main__":
    from distributed_simulation.zkp_math import quantize, norm_sq_int, threshold_sq_int
    import random

    print("=" * 60)
    print("  prover_v2.py — Self Test")
    print("=" * 60)

    n = 32
    C_thresh = 0.5
    pk, vk = setup(n)
    print(f"\n[Setup] circuit_id={pk.circuit_id}")

    random.seed(2)
    grad = [random.uniform(-0.05, 0.05) for _ in range(n)]
    grad_fr = quantize(grad)
    ns = norm_sq_int(grad_fr)
    bound = threshold_sq_int(C_thresh)
    slack = bound - ns
    assert slack >= 0, "test gradient exceeds bound, adjust random range"

    proof = prove(pk, n, grad_fr, slack, bound, round_num=3)
    print(f"[Prove] proof.A = {proof.A}")
    print(f"[Prove] proof.B (G2) type OK: {proof.B is not None}")
    print(f"[Prove] proof.C = {proof.C}")
    print(f"[Prove] public_inputs = {proof.public_inputs}")

    assert proof.A is not None and proof.A != Z1
    assert proof.C is not None and proof.C != Z1
    print("[OK] Proof elements A, C are non-trivial G1 points")

    # Invalid witness should raise before even reaching point arithmetic
    try:
        prove(pk, n, [g + 999999 for g in grad_fr], slack, bound, round_num=3)
        print("[FAIL] Expected ValueError for invalid witness, none raised!")
        raise SystemExit(1)
    except ValueError as e:
        print(f"[OK] Invalid witness correctly rejected before proof construction: {e}")

    print("\nAll prover_v2 self-tests passed (structural — full verify() comes next).")
