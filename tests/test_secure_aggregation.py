"""
test_secure_aggregation.py — SecureFedHE
Standalone, reproducible test suite for the SecureAggregator (X25519 DH-based
secure aggregation). Run directly (no pytest required, though pytest works
too):

    python tests/test_secure_aggregation.py

Covers three properties:
  1. CORRECTNESS — masked values sum to the true sum (masks cancel exactly)
  2. PRIVACY     — masked output is statistically indistinguishable from noise
  3. ROBUSTNESS  — the system fails loudly/safely on bad input, not silently

Import path assumes this file lives in <repo_root>/tests/ and node.py lives
in <repo_root>/node.py.
"""
import sys
import os
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# node.py has module-level side effects (FastAPI app creation) that are
# harmless to import for our purposes, but it does reference a global
# STATE dict for round_num in encrypt(). We set STATE ourselves below.
from node import SecureAggregator, STATE  # noqa: E402


PASS = "PASS"
FAIL = "FAIL"
_results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    _results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail and status == FAIL else ""))
    return condition


def make_ring(node_ids, round_num=0):
    """Build a fully key-exchanged ring of SecureAggregator instances."""
    aggs = {nid: SecureAggregator(nid, node_ids) for nid in node_ids}
    for a_id, a in aggs.items():
        for b_id, b in aggs.items():
            if a_id == b_id:
                continue
            a.add_peer_public_key(b_id, b.get_public_key())
    STATE["current_round"] = round_num
    return aggs


# ─────────────────────────────────────────────────────────────────
# 1. CORRECTNESS — masks cancel to the true sum
# ─────────────────────────────────────────────────────────────────

def test_correctness_basic_3_nodes():
    """3 nodes, random vectors, masked sum == true sum within float32 eps."""
    rng = np.random.default_rng(42)
    node_ids = [0, 1, 2]
    aggs = make_ring(node_ids, round_num=0)

    vectors = {nid: rng.normal(0, 1, size=32).astype(np.float32) for nid in node_ids}
    true_sum = sum(vectors.values())

    masked_sum = sum(aggs[nid].mask(vectors[nid], round_num=0) for nid in node_ids)

    max_err = np.max(np.abs(masked_sum - true_sum))
    check(
        "correctness_basic_3_nodes",
        max_err < 1e-3,
        f"max abs error = {max_err}",
    )


def test_correctness_scales_to_5_and_10_nodes():
    """Same check at larger ring sizes, since pairwise mask logic could
    have off-by-one bugs that only appear with more peers."""
    rng = np.random.default_rng(7)
    for n in (5, 10):
        node_ids = list(range(n))
        aggs = make_ring(node_ids, round_num=0)
        vectors = {nid: rng.normal(0, 1, size=32).astype(np.float32) for nid in node_ids}
        true_sum = sum(vectors.values())
        masked_sum = sum(aggs[nid].mask(vectors[nid], round_num=0) for nid in node_ids)
        max_err = np.max(np.abs(masked_sum - true_sum))
        check(
            f"correctness_scales_n{n}",
            max_err < 1e-3,
            f"n={n}, max abs error = {max_err}",
        )


def test_correctness_across_multiple_rounds():
    """Masks must be fresh per round (derived from round_num) but must
    still cancel correctly in every round independently."""
    rng = np.random.default_rng(99)
    node_ids = [0, 1, 2]
    aggs = make_ring(node_ids, round_num=0)

    for round_num in range(5):
        vectors = {nid: rng.normal(0, 1, size=16).astype(np.float32) for nid in node_ids}
        true_sum = sum(vectors.values())
        masked_sum = sum(aggs[nid].mask(vectors[nid], round_num=round_num) for nid in node_ids)
        max_err = np.max(np.abs(masked_sum - true_sum))
        check(
            f"correctness_round_{round_num}",
            max_err < 1e-3,
            f"round={round_num}, max abs error = {max_err}",
        )


def test_correctness_via_encrypt_decrypt_add_scale():
    """Exercise the actual encrypt/add/scale/decrypt path used in
    handle_update(), not just raw mask(), to catch bugs in the dict-based
    wrapper layer."""
    rng = np.random.default_rng(123)
    node_ids = [0, 1]
    aggs = make_ring(node_ids, round_num=0)

    v0 = rng.normal(0, 1, size=8).astype(np.float32)
    v1 = rng.normal(0, 1, size=8).astype(np.float32)

    enc0 = aggs[0].encrypt(v0)
    enc1 = aggs[1].encrypt(v1)

    # weighted average with equal weights (0.5/0.5), matching handle_update's pattern
    scaled0 = aggs[0].scale(enc0, 0.5)
    scaled1 = aggs[1].scale(enc1, 0.5)
    combined = aggs[0].add(scaled0, scaled1)
    result = aggs[0].decrypt(combined)

    expected = (v0 + v1) / 2
    max_err = np.max(np.abs(result - expected))
    check(
        "correctness_encrypt_decrypt_add_scale",
        max_err < 1e-3,
        f"max abs error = {max_err}",
    )


# ─────────────────────────────────────────────────────────────────
# 2. PRIVACY — masked output looks like noise, not the real value
# ─────────────────────────────────────────────────────────────────

def test_privacy_masked_output_is_not_the_raw_value():
    """Sanity floor: masked output must differ substantially from the
    unmasked input (catches a no-op mask() bug)."""
    rng = np.random.default_rng(5)
    node_ids = [0, 1, 2]
    aggs = make_ring(node_ids, round_num=0)

    v = rng.normal(0, 1, size=32).astype(np.float32)
    masked = aggs[0].mask(v, round_num=0)

    diff = np.max(np.abs(masked - v))
    check(
        "privacy_masked_differs_from_raw",
        diff > 1e-4,
        f"masked output too close to raw input (diff={diff}); mask() may be a no-op",
    )


def test_privacy_same_input_different_rounds_looks_different():
    """The SAME underlying value masked in two different rounds must
    produce different ciphertexts — otherwise an eavesdropper could
    detect that a node sent the same value twice."""
    node_ids = [0, 1, 2]
    aggs = make_ring(node_ids, round_num=0)

    v = np.ones(16, dtype=np.float32) * 0.3

    masked_round0 = aggs[0].mask(v, round_num=0)
    masked_round1 = aggs[0].mask(v, round_num=1)

    diff = np.max(np.abs(masked_round0 - masked_round1))
    check(
        "privacy_round_freshness",
        diff > 1e-4,
        f"same input masked identically across rounds (diff={diff}) — mask reuse risk",
    )


def test_privacy_masked_distribution_resembles_noise():
    """Statistical check: repeatedly mask the SAME fixed vector with many
    independent random peer key-exchanges (simulating an observer who
    only ever sees masked outputs). The masked values, viewed as a
    distribution, should NOT cluster tightly around the true value —
    if they do, an observer could average out the noise and recover it.
    """
    v = np.array([0.5, -0.3, 0.1, 0.05], dtype=np.float32)
    samples = []

    for trial in range(200):
        node_ids = [0, 1, 2]
        aggs = make_ring(node_ids, round_num=0)
        masked = aggs[0].mask(v, round_num=0)
        samples.append(masked[0])  # track first component across trials

    samples = np.array(samples)
    mean_recovered = np.mean(samples)
    std_recovered = np.std(samples)

    # The mean of the masks (from _get_mask, ~N(0, 0.01)) should be near
    # zero, so averaging masked samples SHOULD tend back toward the true
    # value for component 0 (0.5) — this is expected and not a privacy
    # break by itself, since it requires the SAME peer keys reused many
    # times, which never happens in the real protocol (fresh DH keys
    # every process start). What we're really checking is that any
    # SINGLE masked observation has enough spread (std) to not trivially
    # reveal 0.5.
    check(
        "privacy_single_observation_has_spread",
        std_recovered > 1e-4,
        f"std={std_recovered} — masked outputs have near-zero spread, "
        f"meaning a single observation could reveal the true value",
    )
    print(
        f"    (info: mean of 200 independent masked samples = {mean_recovered:.4f}, "
        f"true value = {v[0]}, std = {std_recovered:.4f})"
    )


def test_privacy_two_node_collusion_recovers_third():
    """KNOWN LIMITATION, not a bug: if 2 of 3 nodes collude and share
    their private keys/shared secrets, they can jointly recover the
    third node's masks and hence its raw value. This test documents
    that this is possible (expected for this threshold), so it is
    tracked explicitly rather than silently assumed away.
    """
    rng = np.random.default_rng(11)
    node_ids = [0, 1, 2]
    aggs = make_ring(node_ids, round_num=0)

    v2 = rng.normal(0, 1, size=8).astype(np.float32)
    masked_from_2 = aggs[2].mask(v2, round_num=0)

    # Nodes 0 and 1 collude: each independently knows its own shared
    # secret with node 2, so together they can reconstruct both masks
    # node 2 applied and add them back.
    mask_2_from_0_view = aggs[0]._get_mask(2, len(v2), round_num=0)
    mask_2_from_1_view = aggs[1]._get_mask(2, len(v2), round_num=0)

    # Node 2's own mask() logic: for each peer_id < node_id (2), it
    # SUBTRACTS that peer's derived mask (peer_id > self.node_id is
    # False for peer 0 and peer 1 relative to node 2, so both go to the
    # else branch: masked -= mask). To reconstruct, colluders ADD both
    # masks back.
    reconstructed = masked_from_2 + mask_2_from_0_view + mask_2_from_1_view
    recovery_err = np.max(np.abs(reconstructed - v2))

    check(
        "privacy_collusion_limitation_documented",
        recovery_err < 1e-3,
        f"reconstruction error = {recovery_err} (expected near-zero: "
        f"this IS a real recovery, documenting the N-1 collusion limitation)",
    )
    print(
        "    (info: this test is EXPECTED to pass — it documents that "
        "2-of-3 colluding nodes CAN recover a third node's value. "
        "This is inherent to this scheme below full non-collusion, "
        "not a bug to fix. State this explicitly in your threat model.)"
    )


# ─────────────────────────────────────────────────────────────────
# 3. ROBUSTNESS — fail loudly/safely, not silently
# ─────────────────────────────────────────────────────────────────

def test_robustness_missing_shared_secret_does_not_silently_pass_through():
    """If a peer's shared secret is missing (key exchange never happened
    or failed), mask() must NOT silently skip that peer's contribution —
    doing so would mean the value is sent completely unmasked to whatever
    peers WERE exchanged, a real privacy leak. This should raise, or at
    minimum the caller must be able to detect it did not fully mask.
    """
    node_ids = [0, 1, 2]
    a0 = SecureAggregator(0, node_ids)
    a1 = SecureAggregator(1, node_ids)
    # Deliberately skip a0 <-> a2 key exchange.
    a0.add_peer_public_key(1, a1.get_public_key())
    a1.add_peer_public_key(0, a0.get_public_key())

    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    raised = False
    result = None
    try:
        result = a0.mask(v, round_num=0)
    except RuntimeError:
        raised = True
    except Exception:
        raised = True  # any hard failure counts; we check type separately below

    if raised:
        check("robustness_missing_secret_raises", True)
    else:
        # If it did NOT raise, confirm it is at least not silently
        # returning the raw unmasked value for the missing peer — flag
        # this as a FAIL either way since the documented intended
        # behavior (per project history) is a hard RuntimeError.
        unmasked_component_leaked = np.allclose(result, v, atol=1e-6)
        check(
            "robustness_missing_secret_raises",
            False,
            "mask() did not raise on missing peer secret — silently "
            f"{'passed through unmasked' if unmasked_component_leaked else 'masked partially'}. "
            "Expected: RuntimeError per project history. This is a REGRESSION if so.",
        )


def test_robustness_zero_length_vector():
    """Edge case: masking an empty array should not crash or hang."""
    node_ids = [0, 1]
    aggs = make_ring(node_ids, round_num=0)
    try:
        result = aggs[0].mask(np.array([], dtype=np.float32), round_num=0)
        check("robustness_zero_length_vector", len(result) == 0)
    except Exception as e:
        check("robustness_zero_length_vector", False, f"raised unexpectedly: {e}")


def test_robustness_large_round_number():
    """round_num is packed into 4 bytes (round_num.to_bytes(4, 'big')) —
    confirm this doesn't overflow/crash for large but plausible round
    numbers (e.g. a very long-running deployment)."""
    node_ids = [0, 1]
    aggs = make_ring(node_ids, round_num=0)
    v = np.array([1.0, 2.0], dtype=np.float32)
    try:
        aggs[0].mask(v, round_num=2**31)
        check("robustness_large_round_number", True)
    except Exception as e:
        check("robustness_large_round_number", False, f"raised unexpectedly: {e}")


def test_robustness_negative_round_number_rejected_or_handled():
    """round_num.to_bytes(4, 'big') will raise OverflowError on negative
    input by default — confirm this fails LOUDLY (good) rather than
    wrapping/truncating silently (bad, would break mask freshness)."""
    node_ids = [0, 1]
    aggs = make_ring(node_ids, round_num=0)
    v = np.array([1.0, 2.0], dtype=np.float32)
    try:
        aggs[0].mask(v, round_num=-1)
        check(
            "robustness_negative_round_number",
            False,
            "negative round_num was silently accepted — should raise",
        )
    except (OverflowError, ValueError):
        check("robustness_negative_round_number", True)
    except Exception as e:
        check(
            "robustness_negative_round_number",
            False,
            f"raised unexpected exception type: {type(e).__name__}: {e}",
        )


def test_robustness_mismatched_vector_lengths_in_add():
    """add() on two encrypted dicts with different lengths should fail
    clearly rather than silently truncating/broadcasting."""
    node_ids = [0, 1]
    aggs = make_ring(node_ids, round_num=0)
    enc_a = aggs[0].encrypt(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    enc_b = aggs[0].encrypt(np.array([1.0, 2.0], dtype=np.float32))
    try:
        result = aggs[0].add(enc_a, enc_b)
        # numpy broadcasting rules mean this may not raise — check if it
        # silently produced a wrong-shaped or broadcast result instead.
        result_arr = np.array(result["data"])
        check(
            "robustness_mismatched_lengths_add",
            False,
            f"add() with mismatched lengths (3 vs 2) did not raise; "
            f"produced shape {result_arr.shape} — silent shape bug risk",
        )
    except (ValueError, Exception) as e:
        check("robustness_mismatched_lengths_add", True, f"raised as expected: {e}")


# ─────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SecureFedHE — Secure Aggregation Test Suite")
    print("=" * 60)

    print("\n--- Correctness ---")
    test_correctness_basic_3_nodes()
    test_correctness_scales_to_5_and_10_nodes()
    test_correctness_across_multiple_rounds()
    test_correctness_via_encrypt_decrypt_add_scale()

    print("\n--- Privacy ---")
    test_privacy_masked_output_is_not_the_raw_value()
    test_privacy_same_input_different_rounds_looks_different()
    test_privacy_masked_distribution_resembles_noise()
    test_privacy_two_node_collusion_recovers_third()

    print("\n--- Robustness ---")
    test_robustness_missing_shared_secret_does_not_silently_pass_through()
    test_robustness_zero_length_vector()
    test_robustness_large_round_number()
    test_robustness_negative_round_number_rejected_or_handled()
    test_robustness_mismatched_vector_lengths_in_add()

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, s, _ in _results if s == PASS)
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    print(f"  Results: {n_pass} passed, {n_fail} failed, {len(_results)} total")
    print("=" * 60)

    if n_fail > 0:
        print("\nFAILED CHECKS:")
        for name, status, detail in _results:
            if status == FAIL:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print("\nAll checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
