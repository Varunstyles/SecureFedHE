"""
verifier_v2.py — SecureFedHE
Real Groth16 verifier for the norm-bound circuit — uses an ACTUAL bilinear
pairing check, unlike the old zkp_engine.py verifier which just compared
G1 point additions (and explicitly admitted in its own comments that this
was not sound).

Groth16 verification equation:

    e(A, B) == e(alpha_g1, beta_g2) * e(vk_x, gamma_g2) * e(C, delta_g2)

Where vk_x = IC[0] + sum_{i=1}^{n_public-1} public_inputs[i] * IC[i]
(IC[0] corresponds to the constant "1" wire).

This holds if and only if the prover knew a valid witness satisfying every
R1CS constraint. Forging A, B, C without a valid witness requires solving
the discrete log problem on BN128 — not feasible classically. This is the
soundness guarantee that was completely absent from the old implementation.
"""
from typing import List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_ecc.bn128 import add, multiply, pairing, curve_order, Z1, eq

from distributed_simulation.trusted_setup import VerificationKeyV2
from distributed_simulation.prover_v2 import Groth16ProofV2

import sys
sys.setrecursionlimit(10000)

Fr = curve_order


def verify(vk: VerificationKeyV2, proof: Groth16ProofV2) -> bool:
    """
    Returns True iff the proof is valid under vk — via a REAL pairing check.
    """
    if len(proof.public_inputs) != vk.n_public:
        return False

    # vk_x = IC[0] * 1 + sum_{i>=1} public_inputs[i] * IC[i]
    vk_x = vk.ic[0]  # public_inputs[0] is always the constant 1
    for i in range(1, vk.n_public):
        vk_x = add(vk_x, multiply(vk.ic[i], proof.public_inputs[i]))

    # e(A, B) should equal e(alpha, beta) * e(vk_x, gamma) * e(C, delta)
    # py_ecc's pairing(Q, P) computes e(P, Q) with Q in G2, P in G1.
    lhs = pairing(proof.B, proof.A)

    rhs = (
        pairing(vk.beta_g2, vk.alpha_g1)
        * pairing(vk.gamma_g2, vk_x)
        * pairing(vk.delta_g2, proof.C)
    )

    return lhs == rhs


def _run_self_test():
    from distributed_simulation.trusted_setup import setup
    from distributed_simulation.prover_v2 import prove
    from distributed_simulation.zkp_math import quantize, norm_sq_int, threshold_sq_int
    from py_ecc.bn128 import G1, G2
    import random

    print("=" * 60)
    print("  verifier_v2.py — Self Test (the real moment of truth)")
    print("=" * 60)

    n = 32
    C_thresh = 0.5
    pk, vk = setup(n)

    random.seed(3)
    grad = [random.uniform(-0.05, 0.05) for _ in range(n)]
    grad_fr = quantize(grad)
    ns = norm_sq_int(grad_fr)
    bound = threshold_sq_int(C_thresh)
    slack = bound - ns
    assert slack >= 0

    # ---- Test 1: honest proof, correct public inputs -> must PASS ----
    proof = prove(pk, n, grad_fr, slack, bound, round_num=5)
    ok = verify(vk, proof)
    print(f"\n[Test 1] Honest proof, correct verification key: {'PASS' if ok else 'FAIL'}")
    assert ok, "Honest proof MUST verify — if this fails, the whole scheme is broken"

    # ---- Test 2: tampered public input (claim a different bound) -> must FAIL ----
    proof_bad_pub = Groth16ProofV2(proof.A, proof.B, proof.C,
                                    [1, bound + 12345, 5])
    ok2 = verify(vk, proof_bad_pub)
    print(f"[Test 2] Proof with tampered public input (expect FAIL): "
          f"{'PASS (BUG!)' if ok2 else 'FAIL (correct)'}")
    assert not ok2, "Tampering public inputs must invalidate the proof"

    # ---- Test 3: proof from a DIFFERENT trusted setup (wrong vk) -> must FAIL ----
    pk_other, vk_other = setup(n)
    ok3 = verify(vk_other, proof)
    print(f"[Test 3] Valid proof checked against WRONG verification key (expect FAIL): "
          f"{'PASS (BUG!)' if ok3 else 'FAIL (correct)'}")
    assert not ok3, "Proof must not verify under a mismatched setup"

    # ---- Test 4: THE REAL ATTACK — forge (A,B,C) as random curve points
    # without ever running prove(). This is exactly the attack the OLD
    # verifier (G1-only addition check) was vulnerable to. ----
    forged_A = multiply(G1, 999999937)
    forged_B = multiply(G2, 123456789)
    forged_C = multiply(G1, 555555555)
    forged_proof = Groth16ProofV2(forged_A, forged_B, forged_C, [1, bound, 5])
    ok4 = verify(vk, forged_proof)
    print(f"[Test 4] FORGED proof (random points, no real witness, expect FAIL): "
          f"{'PASS (CRITICAL BUG!)' if ok4 else 'FAIL (correct — forgery rejected)'}")
    assert not ok4, "CRITICAL: forged proof must be rejected — this is the whole point of Path B"

    # ---- Test 5: another naive forged proof, different random points ----
    naive_forged_A = multiply(G1, 42)
    naive_forged_B_g2 = multiply(G2, 43)
    naive_forged_C = multiply(G1, 44)
    naive_proof = Groth16ProofV2(naive_forged_A, naive_forged_B_g2, naive_forged_C, [1, bound, 5])
    ok5 = verify(vk, naive_proof)
    print(f"[Test 5] Second forged proof attempt, different points (expect FAIL): "
          f"{'PASS (BUG!)' if ok5 else 'FAIL (correct)'}")
    assert not ok5

    # ---- Test 6: gradient that violates norm bound must be refused at
    #      prove() time and never reach the verifier at all ----
    bad_grad = [5.0] * n  # way over norm bound
    bad_grad_fr = quantize(bad_grad)
    bad_ns = norm_sq_int(bad_grad_fr)
    try:
        bad_slack = bound - bad_ns
        prove(pk, n, bad_grad_fr, bad_slack, bound, round_num=5)
        print("[Test 6] FAIL — over-norm gradient should not produce a proof!")
        raise SystemExit(1)
    except ValueError:
        print("[Test 6] Over-norm-bound gradient correctly refused at prove() stage (correct)")

    print("\n" + "=" * 60)
    print("  ALL VERIFIER SELF-TESTS PASSED")
    print("  Honest proofs verify. Forged proofs (including attack patterns")
    print("  the OLD additive-check verifier was vulnerable to) are")
    print("  correctly rejected by the real pairing check.")
    print("=" * 60)


if __name__ == "__main__":
    import sys, threading
    sys.setrecursionlimit(100000)
    threading.stack_size(64 * 1024 * 1024)
    t = threading.Thread(target=_run_self_test)
    t.start()
    t.join()