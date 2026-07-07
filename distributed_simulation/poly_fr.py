"""
poly_fr.py — SecureFedHE
Dense polynomial arithmetic over Fr, represented as coefficient lists
(index i = coefficient of x^i, ascending order).

Needed for: Lagrange interpolation (R1CS -> QAP), polynomial evaluation
at the trusted-setup secret tau, and the prover's H(x) computation
( (A.w)(x) * (B.w)(x) - (C.w)(x) = H(x) * Z(x) ).

Kept deliberately simple (O(n^2) multiply) since our circuit is small
(33 constraints) — no need for FFT-based multiplication at this scale.
"""
from typing import List

Poly = List[int]  # coeff[i] = coefficient of x^i


def poly_trim(p: Poly) -> Poly:
    """Remove trailing zero coefficients (keep at least [0])."""
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def poly_add(a: Poly, b: Poly, Fr: int) -> Poly:
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        av = a[i] if i < len(a) else 0
        bv = b[i] if i < len(b) else 0
        out[i] = (av + bv) % Fr
    return poly_trim(out)


def poly_sub(a: Poly, b: Poly, Fr: int) -> Poly:
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        av = a[i] if i < len(a) else 0
        bv = b[i] if i < len(b) else 0
        out[i] = (av - bv) % Fr
    return poly_trim(out)


def poly_scale(a: Poly, k: int, Fr: int) -> Poly:
    return poly_trim([(c * k) % Fr for c in a])


def poly_mul(a: Poly, b: Poly, Fr: int) -> Poly:
    if a == [0] or b == [0]:
        return [0]
    out = [0] * (len(a) + len(b) - 1)
    for i, av in enumerate(a):
        if av == 0:
            continue
        for j, bv in enumerate(b):
            out[i + j] = (out[i + j] + av * bv) % Fr
    return poly_trim(out)


def poly_eval(a: Poly, x: int, Fr: int) -> int:
    """Horner's method."""
    result = 0
    for c in reversed(a):
        result = (result * x + c) % Fr
    return result


def poly_divmod(a: Poly, b: Poly, Fr: int):
    """
    Polynomial long division over Fr: a = q*b + r, deg(r) < deg(b).
    b must not be the zero polynomial.
    """
    a = poly_trim(a)
    b = poly_trim(b)
    if b == [0]:
        raise ZeroDivisionError("poly_divmod: divisor is zero polynomial")
    rem = list(a)
    deg_b = len(b) - 1
    lead_b_inv = pow(b[-1], Fr - 2, Fr)
    q = [0] * max(1, len(a) - len(b) + 1)
    while len(poly_trim(rem)) - 1 >= deg_b and poly_trim(rem) != [0]:
        rem = poly_trim(rem)
        deg_r = len(rem) - 1
        coeff = (rem[-1] * lead_b_inv) % Fr
        shift = deg_r - deg_b
        if shift < 0:
            break
        q[shift] = coeff
        # rem -= coeff * x^shift * b
        sub = [0] * shift + [(coeff * bc) % Fr for bc in b]
        rem = poly_sub(rem, sub, Fr)
    return poly_trim(q), poly_trim(rem)


def lagrange_interpolate(xs: List[int], ys: List[int], Fr: int) -> Poly:
    """
    Standard Lagrange interpolation: returns the unique polynomial of
    degree < len(xs) passing through all (xs[i], ys[i]) points, over Fr.
    """
    assert len(xs) == len(ys)
    result = [0]
    for i in range(len(xs)):
        if ys[i] == 0:
            continue
        # basis_i(x) = prod_{j != i} (x - xs[j]) / (xs[i] - xs[j])
        basis = [1]
        denom = 1
        for j in range(len(xs)):
            if j == i:
                continue
            # multiply basis by (x - xs[j])
            basis = poly_mul(basis, [(-xs[j]) % Fr, 1], Fr)
            denom = (denom * ((xs[i] - xs[j]) % Fr)) % Fr
        inv_denom = pow(denom, Fr - 2, Fr)
        term = poly_scale(basis, (ys[i] * inv_denom) % Fr, Fr)
        result = poly_add(result, term, Fr)
    return poly_trim(result)


if __name__ == "__main__":
    Fr = 21888242871839275222246405745257275088548364400416034343698204186575808495617

    print("=" * 60)
    print("  poly_fr.py — Self Test")
    print("=" * 60)

    # add/sub/mul sanity: (1 + 2x) + (3 + x) = (4 + 3x)
    a = [1, 2]
    b = [3, 1]
    assert poly_add(a, b, Fr) == [4, 3]
    print("[OK] poly_add")

    # (4+3x) - (3+x) = (1+2x)
    assert poly_sub([4, 3], b, Fr) == [1, 2]
    print("[OK] poly_sub")

    # (1+2x)*(3+x) = 3 + x + 6x + 2x^2 = 3 + 7x + 2x^2
    assert poly_mul(a, b, Fr) == [3, 7, 2]
    print("[OK] poly_mul")

    # eval: p(x)=1+2x at x=5 -> 11
    assert poly_eval([1, 2], 5, Fr) == 11
    print("[OK] poly_eval")

    # divmod: (x^2 - 1) / (x - 1) = (x + 1), remainder 0
    q, r = poly_divmod([-1 % Fr, 0, 1], [-1 % Fr, 1], Fr)
    assert q == [1, 1], f"expected [1,1] got {q}"
    assert r == [0], f"expected remainder 0 got {r}"
    print("[OK] poly_divmod exact division")

    # divmod with nonzero remainder: (x^2 + 1) / (x - 1) = (x+1) rem 2
    q2, r2 = poly_divmod([1, 0, 1], [-1 % Fr, 1], Fr)
    # verify q2*b + r2 == a
    b_ = [-1 % Fr, 1]
    reconstructed = poly_add(poly_mul(q2, b_, Fr), r2, Fr)
    assert reconstructed == poly_trim([1, 0, 1]), f"reconstruction failed: {reconstructed}"
    print(f"[OK] poly_divmod with remainder (q={q2}, r={r2}), reconstruction verified")

    # Lagrange interpolation: fit points (1,1),(2,4),(3,9) -> should recover x^2
    xs = [1, 2, 3]
    ys = [1, 4, 9]
    p = lagrange_interpolate(xs, ys, Fr)
    for x, y in zip(xs, ys):
        assert poly_eval(p, x, Fr) == y, f"interpolation failed at x={x}"
    # also check it behaves like x^2 at a held-out point x=4 -> 16
    assert poly_eval(p, 4, Fr) == 16, f"expected 16 at x=4, got {poly_eval(p, 4, Fr)}"
    print(f"[OK] lagrange_interpolate recovers x^2 exactly: p={p}")

    # Larger random cross-check: interpolate random points, verify polydiv consistency
    import random
    random.seed(42)
    xs2 = [i for i in range(1, 8)]
    ys2 = [random.randrange(0, Fr) for _ in xs2]
    p2 = lagrange_interpolate(xs2, ys2, Fr)
    for x, y in zip(xs2, ys2):
        assert poly_eval(p2, x, Fr) == y
    print(f"[OK] lagrange_interpolate on 7 random points, degree {len(p2)-1}")

    print("\nAll poly_fr self-tests passed.")
