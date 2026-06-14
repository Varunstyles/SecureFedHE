"""
crypto/validate_he.py
Quick sanity-check script — run this BEFORE he_train.py.

Works in both real TenSEAL mode and simulated mode (Windows).

Checks:
  1. HE backend availability
  2. Context generation
  3. Encrypt → Decrypt roundtrip
  4. Approximation error within bounds
  5. HE aggregation (FedAvg on ciphertexts)
  6. Encryption timing

Usage:
    python -m crypto.validate_he
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(label: str, width: int = 30):
    print(f"  [{label}]{'.' * (width - len(label))}", end=" ", flush=True)


def ok(msg=""):
    print(f"OK  {msg}")


def fail(msg=""):
    print(f"FAIL  {msg}")
    sys.exit(1)


def main():
    print(f"\n{'='*55}")
    print(f"  SecureFedHE · HE Validation")
    print(f"{'='*55}\n")

    # ── 1. Backend ────────────────────────────────────────────────────────
    check("1. HE backend")
    from crypto.he_layer import SIMULATED, HE_AVAILABLE
    if SIMULATED:
        ok("(SIMULATED mode — TenSEAL not available)")
    elif HE_AVAILABLE:
        import tenseal as ts
        ok(f"(TenSEAL {ts.__version__})")
    else:
        fail("No HE backend available")

    # ── 2. Context ─────────────────────────────────────────────────────────
    check("2. Context creation")
    try:
        from crypto.he_layer import create_he_context
        t0 = time.perf_counter()
        ctx = create_he_context(poly_modulus_degree=8192)
        ok(f"({time.perf_counter()-t0:.2f}s)")
    except Exception as e:
        fail(str(e))

    # ── 3. Encrypt → Decrypt ───────────────────────────────────────────────
    check("3. Encrypt -> Decrypt")
    try:
        from crypto.he_layer import encrypt_layer, decrypt_layer
        original = np.random.randn(10, 256).astype(np.float32)
        enc = encrypt_layer(original, ctx)
        recovered = decrypt_layer(enc, original.shape)
        ok()
    except Exception as e:
        fail(str(e))

    # ── 4. Approximation Error ─────────────────────────────────────────────
    check("4. Approximation error")
    diff = np.abs(original - recovered)
    max_err = diff.max()
    mean_err = diff.mean()
    threshold = 1e-3
    if max_err < threshold:
        ok(f"(max={max_err:.2e}, mean={mean_err:.2e})")
    else:
        fail(f"Error too large: max={max_err:.2e} (threshold={threshold:.0e})")

    # ── 5. HE Aggregation ──────────────────────────────────────────────────
    check("5. HE aggregation")
    try:
        client_weights = [0.4, 0.35, 0.25]
        arrs = [np.random.randn(10, 256).astype(np.float32) for _ in range(3)]
        encs = [encrypt_layer(a, ctx) for a in arrs]

        # Expected: weighted average (plaintext)
        expected = sum(a * w for a, w in zip(arrs, client_weights))

        # HE aggregation
        result_enc = encs[0] * client_weights[0]
        for e, w in zip(encs[1:], client_weights[1:]):
            result_enc = result_enc + (e * w)
        result = np.array(result_enc.decrypt(), dtype=np.float32).reshape(10, 256)

        agg_diff = np.abs(expected - result).max()
        if agg_diff < 1e-3:
            ok(f"(max_diff={agg_diff:.2e})")
        else:
            fail(f"Aggregation error too large: {agg_diff:.2e}")
    except Exception as e:
        fail(str(e))

    # ── 6. Encryption Timing ───────────────────────────────────────────────
    check("6. Encryption time")
    try:
        arr = np.random.randn(10, 256).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(5):
            enc = encrypt_layer(arr, ctx)
            _ = enc.decrypt()
        avg_time = (time.perf_counter() - t0) / 5
        ok(f"({avg_time:.3f}s per encrypt+decrypt)")
    except Exception as e:
        fail(str(e))

    mode_str = "SIMULATED" if SIMULATED else "TenSEAL"
    print(f"\n{'='*55}")
    print(f"  [OK]  All checks passed ({mode_str} mode).")
    print(f"\n  Run Ring 2 with:")
    print(f"    python -m baseline.he_train")
    print(f"    python -m baseline.he_train --rounds 5 --clients 3  # quick test")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
