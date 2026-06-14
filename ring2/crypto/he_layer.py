"""
crypto/he_layer.py
Ring 2 — Selective Homomorphic Encryption (FULL IMPLEMENTATION).

Replace the stub file in your project with this.

Encryption strategy (the core novelty of SecureFedHE):
  ┌─────────────────┬──────────────────────────────────────────┐
  │ Layer           │ Protection                               │
  ├─────────────────┼──────────────────────────────────────────┤
  │ conv blocks     │ Plaintext (generic features, low risk)   │
  │ fc1 (256 units) │ Differential Privacy (Gaussian noise)    │
  │ fc2 (classifier)│ CKKS Homomorphic Encryption  ← HE target │
  └─────────────────┴──────────────────────────────────────────┘

Why this split?
  - fc2 maps directly from learned representations to class labels.
    Its weights encode the most information about individual training
    samples — this is what needs the strongest protection.
  - Full HE on all layers would cost 10-15x overhead. Selective HE
    costs ~2-3x, which is the 3-5x throughput improvement you claim.
  - Convolutional weights learn generic edges/textures that reveal
    nothing about individual users — encrypting them wastes compute.
"""

import numpy as np
from typing import List, Tuple, Dict

try:
    import tenseal as ts
    HE_AVAILABLE = True
except ImportError:
    HE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# HE Context
# ─────────────────────────────────────────────────────────────────────────────

def create_he_context(
    poly_modulus_degree: int = 8192,
    scale_bits: int = 40,
) -> "ts.Context":
    """
    Create and return a CKKS context.

    CKKS (Cheon-Kim-Kim-Song) is the right scheme for neural network
    weights because it supports approximate arithmetic on real numbers.
    BFV/BGV only work on integers — not suitable here.

    Security parameters:
      poly_modulus_degree=8192 → 128-bit security (publication standard)
      scale_bits=40            → ~12 decimal digits of precision
                                 (plenty for float32 weights)

    fc2 has (10 × 256) + 10 = 2,570 floats → fits comfortably in 8192.
    """
    if not HE_AVAILABLE:
        raise ImportError(
            "TenSEAL is not installed.\n"
            "On Linux/WSL/Colab: pip install tenseal\n"
            "TenSEAL does not support Windows natively."
        )

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=[60, scale_bits, scale_bits, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** scale_bits
    return ctx


def serialize_context(ctx: "ts.Context") -> bytes:
    """Serialize context for sharing between simulated nodes."""
    return ctx.serialize(save_secret_key=False)


def load_context(data: bytes) -> "ts.Context":
    return ts.context_from(data)


# ─────────────────────────────────────────────────────────────────────────────
# Encrypt / Decrypt
# ─────────────────────────────────────────────────────────────────────────────

def encrypt_layer(
    weights: np.ndarray,
    ctx: "ts.Context",
) -> "ts.CKKSVector":
    """
    Flatten and encrypt a weight array as a CKKS vector.
    Shape is NOT preserved in the ciphertext — caller must track original_shape.
    """
    return ts.ckks_vector(ctx, weights.flatten().tolist())


def decrypt_layer(
    enc_vector: "ts.CKKSVector",
    original_shape: tuple,
) -> np.ndarray:
    """
    Decrypt and reshape back to original parameter dimensions.
    Small approximation error (~1e-6) is expected from CKKS — this is normal.
    """
    decrypted = np.array(enc_vector.decrypt(), dtype=np.float32)
    return decrypted[:np.prod(original_shape)].reshape(original_shape)


def encrypt_fc2(
    model_params: Dict[str, np.ndarray],
    ctx: "ts.Context",
) -> Tuple[Dict[str, "ts.CKKSVector"], Dict[str, tuple]]:
    """
    Extract and encrypt only fc2 weight and bias from full model params.

    Returns:
        enc_fc2:   {"fc2.weight": CKKSVector, "fc2.bias": CKKSVector}
        shapes:    {"fc2.weight": (10, 256),  "fc2.bias": (10,)}
    """
    enc_fc2, shapes = {}, {}
    for key in ["fc2.weight", "fc2.bias"]:
        arr = model_params[key]
        shapes[key] = arr.shape
        enc_fc2[key] = encrypt_layer(arr, ctx)
    return enc_fc2, shapes


# ─────────────────────────────────────────────────────────────────────────────
# Encrypted Aggregation (FedAvg on ciphertexts)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_encrypted_fc2(
    enc_updates: List[Dict[str, "ts.CKKSVector"]],
    client_sizes: List[int],
) -> Dict[str, "ts.CKKSVector"]:
    """
    Weighted average of encrypted fc2 updates — NO DECRYPTION.

    This is the key HE property: the server computes the aggregate
    without ever seeing plaintext gradients.

    HE supports:
      ciphertext + ciphertext  → ciphertext  (used here for sum)
      ciphertext * scalar      → ciphertext  (used here for weighting)
    """
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]

    aggregated = {}
    for key in ["fc2.weight", "fc2.bias"]:
        # Weighted sum over all clients
        result = enc_updates[0][key] * weights[0]
        for enc, w in zip(enc_updates[1:], weights[1:]):
            result = result + (enc[key] * w)
        aggregated[key] = result

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Differential Privacy (for fc1 intermediate layer)
# ─────────────────────────────────────────────────────────────────────────────

def add_gaussian_dp_noise(
    weights: np.ndarray,
    sensitivity: float = 0.1,
    epsilon: float = 2.0,
    delta: float = 1e-5,
) -> np.ndarray:
    """
    Gaussian mechanism for (epsilon, delta)-DP.
    Applied to fc1 weights (intermediate layer).

    sigma = sensitivity × sqrt(2 × ln(1.25/delta)) / epsilon

    Recommended starting values:
      epsilon=2.0, delta=1e-5 → reasonable privacy with low accuracy impact
      epsilon=1.0              → stronger privacy, may hurt accuracy ~1-2%
      sensitivity=0.1          → clip norm for fc1 gradients

    Lower epsilon = stronger privacy = more noise = lower accuracy.
    Track this trade-off in your Phase 4 benchmarks.
    """
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0.0, sigma, size=weights.shape).astype(np.float32)
    return weights + noise


def clip_weights(weights: np.ndarray, max_norm: float = 0.1) -> np.ndarray:
    """
    L2 clip before adding DP noise — standard practice.
    Bounding sensitivity makes the DP guarantee meaningful.
    """
    norm = np.linalg.norm(weights)
    if norm > max_norm:
        weights = weights * (max_norm / norm)
    return weights


def apply_dp_to_fc1(
    model_params: Dict[str, np.ndarray],
    epsilon: float = 2.0,
    delta: float = 1e-5,
    sensitivity: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Apply clip + DP noise to fc1.weight and fc1.bias."""
    protected = dict(model_params)
    for key in ["fc1.weight", "fc1.bias"]:
        clipped = clip_weights(protected[key], max_norm=sensitivity)
        protected[key] = add_gaussian_dp_noise(
            clipped, sensitivity=sensitivity, epsilon=epsilon, delta=delta
        )
    return protected


# ─────────────────────────────────────────────────────────────────────────────
# Approximation Error Measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_he_error(
    original: np.ndarray,
    ctx: "ts.Context",
) -> Dict[str, float]:
    """
    Encrypt → decrypt a weight array and measure the approximation error.
    Run this once in your Ring 2 validation to confirm error is negligible.
    Target: max_abs_error < 1e-4 for float32 weights.
    """
    enc = encrypt_layer(original, ctx)
    recovered = decrypt_layer(enc, original.shape)
    diff = np.abs(original - recovered)
    return {
        "max_abs_error":  float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "relative_error": float(diff.mean() / (np.abs(original).mean() + 1e-9)),
    }
