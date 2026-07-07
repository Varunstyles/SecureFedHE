"""
trusted_setup.py — SecureFedHE
Real Groth16 trusted setup for the norm-bound QAP circuit, using py_ecc's
BN128 (alt_bn128) implementation for actual G1/G2 points and pairings.

This REPLACES the old fake setup in zkp_engine.py's TrustedSetup class,
which derived "toxic waste" via SHA256(string) instead of real field
sampling, and never touched G2 at all.

Standard Groth16 setup (per the original paper, simplified to our QAP
with n_public public inputs, n_rows constraints):

  Secret (toxic waste, must be destroyed after setup):
      tau, alpha, beta, gamma, delta  <- random Fr

  Proving key (given to every prover):
      alpha_g1 = alpha * G1
      beta_g1  = beta  * G1
      beta_g2  = beta  * G2
      delta_g1 = delta * G1
      delta_g2 = delta * G2
      { tau^i * G1 : i = 0..degree(H) }                  (powers of tau, G1)
      { tau^i * G2 : i = 0..degree(H) }                  (powers of tau, G2, only up to needed degree)
      L_query: for each PRIVATE witness column j:
          ((beta*A_j(tau) + alpha*B_j(tau) + C_j(tau)) / delta) * G1

  Verification key (public):
      alpha_g1, beta_g2, gamma_g2, delta_g2
      IC: for each PUBLIC witness column j (incl. constant):
          ((beta*A_j(tau) + alpha*B_j(tau) + C_j(tau)) / gamma) * G1
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import secrets as pysecrets
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_ecc.bn128 import (
    G1, G2, add, multiply, neg, curve_order, FQ, FQ2, pairing, eq, Z1, Z2
)

from distributed_simulation.r1cs_normbound import build_r1cs
from distributed_simulation.qap_normbound import matrix_to_col_polys, build_vanishing_poly
from distributed_simulation.poly_fr import poly_eval

import sys
import threading

sys.setrecursionlimit(100000)
threading.stack_size(64 * 1024 * 1024)  # 64 MB stack, well above default

Fr = curve_order  # same field as our QAP, confirmed equal to zkp_math.Fr

def rand_fr() -> int:
    return pysecrets.randbelow(Fr - 1) + 1  # nonzero


@dataclass
class ProvingKeyV2:
    alpha_g1: tuple
    beta_g1: tuple
    beta_g2: tuple
    delta_g1: tuple
    delta_g2: tuple
    tau_powers_g1: List[tuple]   # [tau^0 G1, tau^1 G1, ..., tau^d G1]
    tau_powers_g2: List[tuple]   # [tau^0 G2, tau^1 G2, ..., tau^d G2]
    h_query: List[tuple]         # [tau^i / delta * G1] — for the H(tau)*Z(tau) term
    l_query: List[tuple]         # G1 points, one per PRIVATE column
    private_cols: List[int]      # which witness column each l_query entry corresponds to
    n_cols: int
    n_public: int
    n_rows: int
    circuit_id: str


@dataclass
class VerificationKeyV2:
    alpha_g1: tuple
    beta_g2: tuple
    gamma_g2: tuple
    delta_g2: tuple
    ic: List[tuple]               # G1 points, one per PUBLIC column (incl constant)
    n_public: int
    circuit_id: str


def setup(n: int) -> Tuple[ProvingKeyV2, VerificationKeyV2]:
    """
    Run the trusted setup for the norm-bound circuit with gradient dim n.
    Returns (proving_key, verification_key). The secret values
    (tau, alpha, beta, gamma, delta) are sampled here and go out of scope
    immediately after use — nothing about them is retained in the return
    value, matching the "toxic waste must be destroyed" requirement.
    """
    A, B, C, n_cols, n_rows = build_r1cs(n)
    A_polys = matrix_to_col_polys(A, n_cols, n_rows, Fr)
    B_polys = matrix_to_col_polys(B, n_cols, n_rows, Fr)
    C_polys = matrix_to_col_polys(C, n_cols, n_rows, Fr)
    Z = build_vanishing_poly(n_rows, Fr)

    # --- sample toxic waste ---
    tau   = rand_fr()
    alpha = rand_fr()
    beta  = rand_fr()
    gamma = rand_fr()
    delta = rand_fr()

    gamma_inv = pow(gamma, Fr - 2, Fr)
    delta_inv = pow(delta, Fr - 2, Fr)

    n_public = 3  # constant, norm_sq_bound, round_num (matches r1cs_normbound layout)

    # --- powers of tau needed for H(x)*Z(x) term ---
    # H has degree up to (n_rows - 1) - 1 roughly; A_w*B_w has degree up to
    # 2*(n_rows-1), Z has degree n_rows, so H has degree up to (n_rows - 2).
    # Be generous: cap at n_rows to be safe.
    max_h_degree = max(0, 2 * (n_rows - 1) - n_rows)
    # Need enough powers for both H*Z (degree up to len(H)+len(Z)-2, bounded
    # by max_h_degree + n_rows) and for the column polys A_j/B_j/C_j
    # (degree up to n_rows - 1 each). Take the max plus headroom.
    max_power_needed = max(max_h_degree + n_rows, n_rows - 1) + 2
    tau_powers_g1 = []
    tau_powers_g2 = []
    h_query = []
    acc = 1
    for i in range(max_power_needed):
        tau_powers_g1.append(multiply(G1, acc))
        tau_powers_g2.append(multiply(G2, acc))
        h_query.append(multiply(G1, (acc * delta_inv) % Fr))
        acc = (acc * tau) % Fr

    # --- IC (public columns) and L (private columns) ---
    ic = []
    l_query = []
    private_cols = []
    for j in range(n_cols):
        a_tau = poly_eval(A_polys[j], tau, Fr)
        b_tau = poly_eval(B_polys[j], tau, Fr)
        c_tau = poly_eval(C_polys[j], tau, Fr)
        combo = (beta * a_tau + alpha * b_tau + c_tau) % Fr
        if j < n_public:
            val = (combo * gamma_inv) % Fr
            ic.append(multiply(G1, val))
        else:
            val = (combo * delta_inv) % Fr
            l_query.append(multiply(G1, val))
            private_cols.append(j)

    pk = ProvingKeyV2(
        alpha_g1=multiply(G1, alpha),
        beta_g1=multiply(G1, beta),
        beta_g2=multiply(G2, beta),
        delta_g1=multiply(G1, delta),
        delta_g2=multiply(G2, delta),
        tau_powers_g1=tau_powers_g1,
        tau_powers_g2=tau_powers_g2,
        h_query=h_query,
        l_query=l_query,
        private_cols=private_cols,
        n_cols=n_cols,
        n_public=n_public,
        n_rows=n_rows,
        circuit_id=f"NormBound_d{n}_v2",
    )
    vk = VerificationKeyV2(
        alpha_g1=multiply(G1, alpha),
        beta_g2=multiply(G2, beta),
        gamma_g2=multiply(G2, gamma),
        delta_g2=multiply(G2, delta),
        ic=ic,
        n_public=n_public,
        circuit_id=f"NormBound_d{n}_v2",
    )
    # tau, alpha, beta, gamma, delta fall out of scope here — not stored anywhere.
    return pk, vk

def _g1_to_list(pt):
    if pt is None:
        return None  # point at infinity
    return [int(pt[0].n), int(pt[1].n)]

def _g1_from_list(data):
    if data is None:
        return None
    from py_ecc.fields import bn128_FQ as FQ
    return (FQ(data[0]), FQ(data[1]))

def _g2_to_list(pt):
    if pt is None:
        return None  # point at infinity
    return [
        [int(c) for c in pt[0].coeffs],
        [int(c) for c in pt[1].coeffs],
    ]

def _g2_from_list(data):
    if data is None:
        return None
    from py_ecc.fields import bn128_FQ2 as FQ2
    return (FQ2(data[0]), FQ2(data[1]))


def save_setup(pk: "ProvingKeyV2", vk: "VerificationKeyV2", path: str):
    """Serialize (pk, vk) to a JSON file. Contains no secret material —
    tau/alpha/beta/gamma/delta were already discarded inside setup()."""
    import json as _json

    data = {
        "pk": {
            "alpha_g1": _g1_to_list(pk.alpha_g1),
            "beta_g1": _g1_to_list(pk.beta_g1),
            "beta_g2": _g2_to_list(pk.beta_g2),
            "delta_g1": _g1_to_list(pk.delta_g1),
            "delta_g2": _g2_to_list(pk.delta_g2),
            "tau_powers_g1": [_g1_to_list(p) for p in pk.tau_powers_g1],
            "tau_powers_g2": [_g2_to_list(p) for p in pk.tau_powers_g2],
            "h_query": [_g1_to_list(p) for p in pk.h_query],
            "l_query": [_g1_to_list(p) for p in pk.l_query],
            "private_cols": list(pk.private_cols),
            "n_cols": pk.n_cols,
            "n_public": pk.n_public,
            "n_rows": pk.n_rows,
            "circuit_id": pk.circuit_id,
        },
        "vk": {
            "alpha_g1": _g1_to_list(vk.alpha_g1),
            "beta_g2": _g2_to_list(vk.beta_g2),
            "gamma_g2": _g2_to_list(vk.gamma_g2),
            "delta_g2": _g2_to_list(vk.delta_g2),
            "ic": [_g1_to_list(p) for p in vk.ic],
            "n_public": vk.n_public,
            "circuit_id": vk.circuit_id,
        },
    }
    with open(path, "w") as f:
        _json.dump(data, f)


def load_setup(path: str):
    """Deserialize (pk, vk) from a JSON file produced by save_setup()."""
    import json as _json

    with open(path) as f:
        data = _json.load(f)

    p = data["pk"]
    pk = ProvingKeyV2(
        alpha_g1=_g1_from_list(p["alpha_g1"]),
        beta_g1=_g1_from_list(p["beta_g1"]),
        beta_g2=_g2_from_list(p["beta_g2"]),
        delta_g1=_g1_from_list(p["delta_g1"]),
        delta_g2=_g2_from_list(p["delta_g2"]),
        tau_powers_g1=[_g1_from_list(x) for x in p["tau_powers_g1"]],
        tau_powers_g2=[_g2_from_list(x) for x in p["tau_powers_g2"]],
        h_query=[_g1_from_list(x) for x in p["h_query"]],
        l_query=[_g1_from_list(x) for x in p["l_query"]],
        private_cols=list(p["private_cols"]),
        n_cols=p["n_cols"],
        n_public=p["n_public"],
        n_rows=p["n_rows"],
        circuit_id=p["circuit_id"],
    )

    v = data["vk"]
    vk = VerificationKeyV2(
        alpha_g1=_g1_from_list(v["alpha_g1"]),
        beta_g2=_g2_from_list(v["beta_g2"]),
        gamma_g2=_g2_from_list(v["gamma_g2"]),
        delta_g2=_g2_from_list(v["delta_g2"]),
        ic=[_g1_from_list(x) for x in v["ic"]],
        n_public=v["n_public"],
        circuit_id=v["circuit_id"],
    )

    return pk, vk

def _run_self_test():
    print("=" * 60)
    print("  trusted_setup.py — Self Test")
    print("=" * 60)

    n = 32
    pk, vk = setup(n)
    print(f"\n[Setup] n={n}  n_cols={pk.n_cols}  n_public={pk.n_public}  n_rows={pk.n_rows}")
    print(f"[Setup] tau_powers_g1 count: {len(pk.tau_powers_g1)}")
    print(f"[Setup] tau_powers_g2 count: {len(pk.tau_powers_g2)}")
    print(f"[Setup] h_query count: {len(pk.h_query)}")
    assert len(pk.tau_powers_g1) == len(pk.tau_powers_g2) == len(pk.h_query)
    print(f"[Setup] l_query count: {len(pk.l_query)}  (expect n_cols - n_public = {pk.n_cols - pk.n_public})")
    print(f"[Setup] ic count: {len(vk.ic)}  (expect n_public = {vk.n_public})")

    assert len(pk.l_query) == pk.n_cols - pk.n_public
    assert len(vk.ic) == vk.n_public

    # Sanity: alpha_g1 must match between pk and vk (same alpha used)
    assert eq(pk.alpha_g1, vk.alpha_g1), "alpha_g1 mismatch between pk/vk"
    print("[OK] alpha_g1 consistent between proving key and verification key")

    # Sanity: beta_g1 and beta_g2 must correspond to the SAME beta scalar.
    # We can check this via pairing: e(beta_g1, G2) == e(G1, beta_g2)
    lhs = pairing(G2, pk.beta_g1)
    rhs = pairing(pk.beta_g2, G1)
    assert lhs == rhs, "beta_g1/beta_g2 do not correspond to the same scalar!"
    print("[OK] beta_g1 and beta_g2 verified (via pairing) to encode the same secret beta")

    # Sanity: delta_g1/delta_g2 same check
    lhs2 = pairing(G2, pk.delta_g1)
    rhs2 = pairing(pk.delta_g2, G1)
    assert lhs2 == rhs2, "delta_g1/delta_g2 mismatch"
    print("[OK] delta_g1 and delta_g2 verified (via pairing) to encode the same secret delta")

    # Confirm every point is actually on-curve / well-formed (py_ecc raises on garbage automatically
    # during `add`, but let's explicitly re-add each to itself and back to sanity check)
    for pt in [pk.alpha_g1, pk.beta_g1, pk.delta_g1] + pk.tau_powers_g1[:3] + pk.l_query[:3]:
        doubled = add(pt, pt)
        back = add(doubled, neg(pt))
        assert eq(back, pt), "point arithmetic sanity check failed"
    print("[OK] Sampled G1 points pass basic curve arithmetic sanity (double then subtract back)")

    print("\nTwo independent setups must NOT produce the same keys (toxic waste is random):")
    pk2, vk2 = setup(n)
    assert not eq(pk.alpha_g1, pk2.alpha_g1), "two setups produced identical alpha — RNG broken!"
    print("[OK] Independent setup runs produce different (fresh) toxic waste, as required")

    print("\nAll trusted_setup self-tests passed.")


if __name__ == "__main__":
    import threading
    sys.setrecursionlimit(100000)
    threading.stack_size(64 * 1024 * 1024)
    t = threading.Thread(target=_run_self_test)
    t.start()
    t.join()