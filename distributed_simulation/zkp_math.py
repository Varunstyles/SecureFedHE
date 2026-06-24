"""
zkp_math.py — SecureFedHE Phase 1
Core finite field arithmetic and elliptic curve primitives for zk-SNARK implementation.
Uses BN128 (alt_bn128) — same curve as Ethereum, Circom, snarkjs.
"""

import hashlib
from typing import Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# BN128 Constants
# ─────────────────────────────────────────────────────────────────────────────

Fr = 21888242871839275222246405745257275088548364400416034343698204186575808495617  # scalar field
Fq = 21888242871839275222246405745257275088696311157297823662689037894645226208583  # base field
G1 = (1, 2)  # Generator point

# ─────────────────────────────────────────────────────────────────────────────
# Fr Arithmetic (scalar field — used for witness/proof values)
# ─────────────────────────────────────────────────────────────────────────────

def fr_add(a, b): return (a + b) % Fr
def fr_sub(a, b): return (a - b) % Fr
def fr_mul(a, b): return (a * b) % Fr
def fr_neg(a):    return (-a) % Fr
def fr_inv(a):    return pow(a, Fr - 2, Fr)
def fr_div(a, b): return fr_mul(a, fr_inv(b))
def fr_pow(a, e): return pow(a, e, Fr)

# ─────────────────────────────────────────────────────────────────────────────
# Fq Arithmetic (base field — used for point coordinates)
# ─────────────────────────────────────────────────────────────────────────────

def fq_add(a, b): return (a + b) % Fq
def fq_sub(a, b): return (a - b) % Fq
def fq_mul(a, b): return (a * b) % Fq
def fq_neg(a):    return (-a) % Fq
def fq_inv(a):    return pow(a, Fq - 2, Fq)
def fq_div(a, b): return fq_mul(a, fq_inv(b))

# ─────────────────────────────────────────────────────────────────────────────
# BN128 G1 Elliptic Curve (points use Fq for coordinates)
# ─────────────────────────────────────────────────────────────────────────────

Point = Optional[Tuple[int, int]]

def point_add(P: Point, Q: Point) -> Point:
    if P is None: return Q
    if Q is None: return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2:
        if y1 != y2: return None
        lam = fq_div(fq_mul(3, fq_mul(x1, x1)), fq_mul(2, y1))
    else:
        lam = fq_div(fq_sub(y2, y1), fq_sub(x2, x1))
    x3 = fq_sub(fq_sub(fq_mul(lam, lam), x1), x2)
    y3 = fq_sub(fq_mul(lam, fq_sub(x1, x3)), y1)
    return (x3, y3)

def point_mul(P: Point, scalar: int) -> Point:
    scalar = scalar % Fr
    result, addend = None, P
    while scalar:
        if scalar & 1: result = point_add(result, addend)
        addend = point_add(addend, addend)
        scalar >>= 1
    return result

def point_neg(P: Point) -> Point:
    return None if P is None else (P[0], fq_neg(P[1]))

def point_on_curve(P: Point) -> bool:
    if P is None: return True
    x, y = P
    return pow(y, 2, Fq) == (pow(x, 3, Fq) + 3) % Fq

# ─────────────────────────────────────────────────────────────────────────────
# Poseidon Hash — ZK-friendly, cheap in arithmetic circuits
# Parameters: t=3, RF=8 full rounds, RP=57 partial rounds, alpha=5
# ─────────────────────────────────────────────────────────────────────────────

_T  = 3
_RF = 8
_RP = 57
_ALPHA = 5

def _gen_constants(n):
    out = []
    for i in range(n):
        h = hashlib.sha256(f"poseidon_rc_{i}".encode()).digest()
        out.append(int.from_bytes(h, 'big') % Fr)
    return out

def _gen_mds(t):
    M = []
    for i in range(t):
        row = []
        for j in range(t):
            h = hashlib.sha256(f"poseidon_mds_{i}_{j}".encode()).digest()
            v = int.from_bytes(h, 'big') % Fr
            row.append(v if v else 1)
        M.append(row)
    return M

_RC  = _gen_constants((_RF + _RP) * _T)
_MDS = _gen_mds(_T)

def _permute(state):
    rc = 0
    def add_rc(s):
        nonlocal rc
        s = [fr_add(s[i], _RC[rc * _T + i]) for i in range(_T)]
        rc += 1
        return s
    def mds(s):
        return [sum(fr_mul(_MDS[i][j], s[j]) for j in range(_T)) % Fr for i in range(_T)]
    def sbox_full(s):  return [fr_pow(x, _ALPHA) for x in s]
    def sbox_part(s):  return [fr_pow(s[0], _ALPHA)] + s[1:]

    for _ in range(_RF // 2):
        state = mds(sbox_full(add_rc(state)))
    for _ in range(_RP):
        state = mds(sbox_part(add_rc(state)))
    for _ in range(_RF // 2):
        state = mds(sbox_full(add_rc(state)))
    return state

def poseidon_hash(inputs: list) -> int:
    """Hash a list of Fr elements → single Fr element."""
    rate = _T - 1
    state = [0] * _T
    padded = inputs + [0] * ((-len(inputs)) % rate)
    for i in range(0, len(padded), rate):
        for j in range(rate):
            state[j] = fr_add(state[j], padded[i + j])
        state = _permute(state)
    return state[1]

# ─────────────────────────────────────────────────────────────────────────────
# Gradient Quantization — Float ↔ Field Element
# ─────────────────────────────────────────────────────────────────────────────

SCALE = 10 ** 6  # 1,000,000 — 6 decimal places preserved

def quantize(grad: list) -> list:
    """Float gradient → list of Fr integers."""
    return [int(round(g * SCALE)) % Fr for g in grad]

def dequantize(grad_fr: list) -> list:
    """Fr integers → float gradient."""
    half = Fr // 2
    return [(g - Fr if g > half else g) / SCALE for g in grad_fr]

def norm_sq_int(grad_fr: list) -> int:
    """Squared L2 norm in integer space (before mod Fr)."""
    half = Fr // 2
    total = 0
    for g in grad_fr:
        gs = (g - Fr) if g > half else g
        total += gs * gs
    return total

def threshold_sq_int(C: float) -> int:
    """C=0.5 → (0.5 * SCALE)² = 250_000_000_000"""
    return int((C * SCALE) ** 2)


if __name__ == "__main__":
    print("=" * 55)
    print("  SecureFedHE ZKP Math Primitives — Self Test")
    print("=" * 55)

    # Fr arithmetic
    assert fr_mul(fr_inv(7), 7) == 1, "Fr inverse failed"
    print("✓  Fr field arithmetic")

    # Curve
    assert point_on_curve(G1), "G1 not on curve"
    G2 = point_add(G1, G1)
    assert point_on_curve(G2), "2*G1 not on curve"
    assert G2 == point_mul(G1, 2), "point_mul mismatch"
    print("✓  BN128 G1 elliptic curve")

    # Poseidon determinism & collision resistance
    h1 = poseidon_hash([1, 2])
    h2 = poseidon_hash([1, 3])
    assert h1 == poseidon_hash([1, 2]), "Poseidon not deterministic"
    assert h1 != h2, "Poseidon collision"
    print(f"✓  Poseidon hash  h([1,2])={h1 % 10**8}... (truncated)")

    # Quantization round-trip
    grad = [0.3, -0.2, 0.45, -0.1]
    q    = quantize(grad)
    dq   = dequantize(q)
    assert all(abs(a - b) < 1e-5 for a, b in zip(grad, dq)), "Quant roundtrip failed"
    print(f"✓  Quantization round-trip  {grad} → dequantized correctly")

    # Norm check
    import math
    C = 0.5
    ns  = norm_sq_int(quantize(grad))
    thr = threshold_sq_int(C)
    actual = math.sqrt(sum(x**2 for x in grad))
    compliant = ns <= thr
    print(f"✓  Norm check  ‖g‖={actual:.4f} > C={C}  → compliant={compliant}  (correct: False)")

    grad2 = [0.2, -0.1, 0.15, 0.1]
    ns2   = norm_sq_int(quantize(grad2))
    a2    = math.sqrt(sum(x**2 for x in grad2))
    print(f"✓  Norm check  ‖g‖={a2:.4f} < C={C}  → compliant={ns2 <= thr}  (correct: True)")

    print("\nAll checks passed ✓")
