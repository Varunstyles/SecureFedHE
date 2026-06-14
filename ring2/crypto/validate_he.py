"""
crypto/validate_he.py
Quick sanity-check script — run this BEFORE he_train.py.

Checks:
  1. TenSEAL imports correctly
  2. CKKS context generates without error
  3. Encrypt → decrypt a dummy fc2 weight array
  4. Approximation error is within acceptable bounds
  5. HE addition (FedAvg on ciphertexts) works correctly
  6. Encryption time is reasonable

Expected output (on a typical laptop):
  [1] TenSEAL import ........... OK  (version x.x.x)
  [2] Context creation ......... OK  (1.23s)
  [3] Encrypt → Decrypt ........ OK
  [4] Approximation error ...... OK  (max=3.2e-06, mean=1.1e-06)
  [5] HE aggregation ........... OK  (max_diff=4.1e-06)
  [6] Encryption time .......... OK  (0.034s per round)

  ✓  All checks passed. Safe to run he_train.py

Usage:
    python -m crypto.validate_he
"""

import sys
import time
import numpy as np


def check(label: str, width: int = 30):
    print(f"  [{label}]{'.' * (width - len(label))}", end=" ", flush=True)


def ok(msg=""):
    print(f"OK  {msg}")


def fail(msg=""):
    print(f"FAIL  {msg}")
    sys.exit(1)


def main():
    print(f"\n{'='*55}")
    print(f"  SecureFedHE · Ring 2 HE Validation")
    print(f"{'='*55}\n")

    # ── 1. Import ──────────────────────────────────────────────────────────
    check("1. TenSEAL import")
    try:
        import tenseal as ts
        ok(f"(version {ts.__version__})")
    except ImportError:
        fail("Not installed. Run: pip install tenseal")

    # ── 2. Context ─────────────────────────────────────────────────────────
    check("2. Context creation")
    try:
        t0 = time.perf_counter()
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        ctx.generate_galois_keys()
        ctx.global_scale = 2 ** 40
        ok(f"({time.perf_counter()-t0:.2f}s)")
    except Exception as e:
        fail(str(e))

    # ── 3. Encrypt → Decrypt ───────────────────────────────────────────────
    check("3. Encrypt → Decrypt")
    try:
        # Simulate fc2 weight shape: (10, 256) = 2560 floats
        original = np.random.randn(10, 256).astype(np.float32)
        enc = ts.ckks_vector(ctx, original.flatten().tolist())
        recovered = np.array(enc.decrypt(), dtype=np.float32).reshape(10, 256)
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
        # Simulate 3 clients each with a different fc2 update
        client_weights = [0.4, 0.35, 0.25]
        arrs = [np.random.randn(10, 256).astype(np.float32) for _ in range(3)]
        encs = [ts.ckks_vector(ctx, a.flatten().tolist()) for a in arrs]

        # Expected: weighted average
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
            enc = ts.ckks_vector(ctx, arr.flatten().tolist())
            _ = enc.decrypt()
        avg_time = (time.perf_counter() - t0) / 5
        ok(f"({avg_time:.3f}s per encrypt+decrypt)")
    except Exception as e:
        fail(str(e))

    print(f"\n{'='*55}")
    print(f"  ✓  All checks passed. Safe to run he_train.py")
    print(f"\n  Run Ring 2 with:")
    print(f"    python -m baseline.he_train")
    print(f"    python -m baseline.he_train --rounds 5 --clients 3  # quick test")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
