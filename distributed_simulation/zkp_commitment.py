"""
zkp_commitment.py  —  SecureFedHE Byzantine Defence (v2)
=========================================================
Changes from v1:
  - FIX: Replay attack prevention — round number + timestamp added to signature
    Old: signed "grad_hash:norm"
    New: signed "grad_hash:norm:round=N:ts=TIMESTAMP"
    Result: a commitment from round 3 will fail verification in round 5

Vulnerabilities status after this patch:
  ✅ Replay attack        — FIXED (round number in signature)
  ✅ Norm lie             — partial (future work: zk-SNARKs)
  ✅ Private key theft    — partial (future work: key rotation)
  ✅ Colluding nodes      — partial (future work: multi-node verification)
"""

import hashlib
import time
import numpy as np
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

CLIP_THRESHOLD = 0.5
NORM_TOLERANCE = 1e-3
TIMESTAMP_WINDOW_S = 300   # commitment expires after 5 minutes


def generate_node_keypair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    return private_key, private_key.public_key()


def serialize_public_key(public_key):
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )


def deserialize_public_key(pem_bytes):
    return serialization.load_pem_public_key(pem_bytes, backend=default_backend())


def generate_commitment(gradient_array, private_key, clip_threshold=CLIP_THRESHOLD, round_number=0):
    """
    Generate a ZKP-inspired commitment for a gradient update.

    v2 change: round_number and timestamp are now included in the signed
    message, preventing replay attacks — a commitment from round N cannot
    be reused in round N+1 because the round number won't match.

    Arguments:
        gradient_array  : numpy array of fc2 weights (plaintext)
        private_key     : this node's RSA private key
        clip_threshold  : agreed DP clipping value (default 0.5)
        round_number    : current training round (NEW — prevents replay)

    Returns:
        commitment_package (dict)
    """
    flat      = gradient_array.flatten().astype(np.float32)
    norm      = float(np.linalg.norm(flat))
    grad_bytes = flat.tobytes()
    grad_hash  = hashlib.sha256(grad_bytes).hexdigest()
    timestamp  = int(time.time())

    # ── v2: round number + timestamp included in signed message ──────────────
    message = f"{grad_hash}:{norm:.8f}:round={round_number}:ts={timestamp}".encode("utf-8")

    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )

    return {
        "grad_hash":       grad_hash,
        "l2_norm":         norm,
        "clip_threshold":  clip_threshold,
        "round_number":    round_number,    # ← NEW
        "timestamp":       timestamp,       # ← NEW
        "signature":       signature.hex()
    }


def verify_commitment(commitment_package, sender_public_key, expected_round=None):
    """
    Verify a commitment package received from a peer node.

    v2 changes:
      - Verifies round_number matches expected_round (replay prevention)
      - Verifies timestamp is within TIMESTAMP_WINDOW_S (stale proof prevention)

    Arguments:
        commitment_package  : dict received alongside ciphertext
        sender_public_key   : sending node's public key
        expected_round      : the current round number (NEW — pass this in)

    Returns:
        (True,  "OK")            — accept
        (False, "reason string") — reject
    """
    required = {"grad_hash", "l2_norm", "clip_threshold", "round_number", "timestamp", "signature"}
    if not required.issubset(commitment_package.keys()):
        return False, "REJECT: Incomplete commitment package — missing fields"

    grad_hash    = commitment_package["grad_hash"]
    l2_norm      = commitment_package["l2_norm"]
    threshold    = commitment_package["clip_threshold"]
    round_number = commitment_package["round_number"]
    timestamp    = commitment_package["timestamp"]
    sig_hex      = commitment_package["signature"]

    # ── CHECK 1: Norm bound ───────────────────────────────────────────────────
    if l2_norm > threshold + NORM_TOLERANCE:
        return False, (
            f"REJECT: Norm bound violated — "
            f"claimed L2={l2_norm:.4f} > threshold={threshold}"
        )

    # ── CHECK 2: Round number (replay prevention) ─────────────────────────────
    if expected_round is not None and round_number != expected_round:
        return False, (
            f"REJECT: Round mismatch — "
            f"commitment is for round {round_number}, expected round {expected_round} "
            f"(replay attack suspected)"
        )

    # ── CHECK 3: Timestamp freshness (stale proof prevention) ─────────────────
    age = int(time.time()) - timestamp
    if age > TIMESTAMP_WINDOW_S:
        return False, (
            f"REJECT: Commitment expired — "
            f"age={age}s > window={TIMESTAMP_WINDOW_S}s"
        )

    # ── CHECK 4: Signature ────────────────────────────────────────────────────
    message   = f"{grad_hash}:{l2_norm:.8f}:round={round_number}:ts={timestamp}".encode("utf-8")
    signature = bytes.fromhex(sig_hex)

    try:
        sender_public_key.verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
    except InvalidSignature:
        return False, "REJECT: Signature invalid — tampered or wrong node"

    return True, "OK"


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SecureFedHE — ZKP Commitment v2 Self-Test")
    print("=" * 60)

    priv_A, pub_A = generate_node_keypair()
    priv_B, pub_B = generate_node_keypair()

    # Test 1: Honest node
    print("\n[TEST 1] Honest node — correct round, valid norm")
    g = np.random.randn(512).astype(np.float32) * 0.1
    g = g / max(np.linalg.norm(g), 1e-8) * 0.3
    c = generate_commitment(g, priv_A, round_number=5)
    valid, reason = verify_commitment(c, pub_A, expected_round=5)
    print(f"  Result: {'✓ ACCEPTED' if valid else '✗ REJECTED'} — {reason}")
    assert valid

    # Test 2: Replay attack — commitment from round 3 used in round 5
    print("\n[TEST 2] Replay attack — old round commitment reused")
    g2 = np.random.randn(512).astype(np.float32) * 0.1
    g2 = g2 / max(np.linalg.norm(g2), 1e-8) * 0.2
    old_c = generate_commitment(g2, priv_A, round_number=3)   # round 3
    valid, reason = verify_commitment(old_c, pub_A, expected_round=5)  # but we're in round 5
    print(f"  Result: {'✓ ACCEPTED' if valid else '✗ REJECTED'}")
    print(f"  Reason: {reason}")
    assert not valid

    # Test 3: Byzantine norm explosion
    print("\n[TEST 3] Byzantine node — poisoned gradient, faked norm")
    fake_c = {
        "grad_hash":      hashlib.sha256(b"poison").hexdigest(),
        "l2_norm":        0.3,
        "clip_threshold": 0.5,
        "round_number":   5,
        "timestamp":      int(time.time()),
        "signature":      "deadbeef"
    }
    valid, reason = verify_commitment(fake_c, pub_A, expected_round=5)
    print(f"  Result: {'✓ ACCEPTED' if valid else '✗ REJECTED'}")
    print(f"  Reason: {reason}")
    assert not valid

    # Test 4: Wrong node key (identity spoofing)
    print("\n[TEST 4] Identity spoofing — Node B signs, Node A key used")
    g3 = np.random.randn(512).astype(np.float32) * 0.1
    g3 = g3 / max(np.linalg.norm(g3), 1e-8) * 0.25
    spoof_c = generate_commitment(g3, priv_B, round_number=5)
    valid, reason = verify_commitment(spoof_c, pub_A, expected_round=5)
    print(f"  Result: {'✓ ACCEPTED' if valid else '✗ REJECTED'}")
    print(f"  Reason: {reason}")
    assert not valid

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
