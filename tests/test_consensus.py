"""
test_consensus.py — SecureFedHE-Consensus (Stage 1)
Standalone test suite for consensus_models.py and consensus_engine.py.
Does NOT require node.py, FastAPI, or a running ring — pure logic
tests using freshly generated RSA keys (not your real certs), so this
is safe to run anywhere without touching certs/.

Run:
    python tests/test_consensus.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402

from consensus_models import (  # noqa: E402
    ConsensusVote,
    QuorumCertificate,
    RejectionReason,
    canonical_json,
    sha256_hex,
)
from consensus_engine import (  # noqa: E402
    sign_payload,
    verify_signature,
    make_vote,
    verify_vote,
    compute_update_hash,
    QuorumTracker,
    evaluate_update_for_vote,
)


PASS = "PASS"
FAIL = "FAIL"
_results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    _results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail and status == FAIL else ""))
    return condition


def gen_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


# ─────────────────────────────────────────────────────────────────
# Signing primitives
# ─────────────────────────────────────────────────────────────────

def test_canonical_json_is_deterministic():
    a = {"b": 1, "a": 2, "c": [1, 2, 3]}
    b = {"c": [1, 2, 3], "a": 2, "b": 1}
    check("canonical_json_deterministic", canonical_json(a) == canonical_json(b))


def test_sign_and_verify_roundtrip():
    key = gen_key()
    payload = {"round_id": 1, "update_hash": "abc123", "decision": True}
    sig = sign_payload(key, payload)
    ok = verify_signature(key.public_key(), payload, sig)
    check("sign_verify_roundtrip", ok)


def test_verify_rejects_tampered_payload():
    key = gen_key()
    payload = {"round_id": 1, "update_hash": "abc123", "decision": True}
    sig = sign_payload(key, payload)
    tampered = {"round_id": 1, "update_hash": "abc123", "decision": False}
    ok = verify_signature(key.public_key(), tampered, sig)
    check("verify_rejects_tampered_payload", not ok)


def test_verify_rejects_wrong_key():
    key_a = gen_key()
    key_b = gen_key()
    payload = {"round_id": 1, "update_hash": "abc123"}
    sig = sign_payload(key_a, payload)
    ok = verify_signature(key_b.public_key(), payload, sig)
    check("verify_rejects_wrong_key", not ok)


def test_verify_rejects_garbage_signature():
    key = gen_key()
    payload = {"round_id": 1}
    ok = verify_signature(key.public_key(), payload, "not_a_real_hex_signature")
    check("verify_rejects_garbage_signature", not ok)


def test_update_hash_is_deterministic_and_binds_both_parts():
    proof = {"A": [1, 2], "B": [[3, 4], [5, 6]], "public_inputs": [1, 250000000000, 0]}
    enc_summary = {"length": 32, "digest": "deadbeef"}
    h1 = compute_update_hash(proof, enc_summary)
    h2 = compute_update_hash(proof, enc_summary)
    check("update_hash_deterministic", h1 == h2)

    different_enc = {"length": 32, "digest": "cafebabe"}
    h3 = compute_update_hash(proof, different_enc)
    check("update_hash_binds_enc_summary", h1 != h3, "changing enc_summary did not change hash")

    different_proof = {"A": [9, 9], "B": [[3, 4], [5, 6]], "public_inputs": [1, 250000000000, 0]}
    h4 = compute_update_hash(different_proof, enc_summary)
    check("update_hash_binds_proof", h1 != h4, "changing proof did not change hash")


# ─────────────────────────────────────────────────────────────────
# Vote creation/verification
# ─────────────────────────────────────────────────────────────────

def test_make_vote_and_verify():
    key = gen_key()
    vote = make_vote(key, round_id=5, update_hash="xyz", voter_node_id=1,
                      decision=True, reason_code=RejectionReason.OK)
    ok = verify_vote(key.public_key(), vote)
    check("make_vote_verifies", ok)


def test_vote_serialization_roundtrip():
    key = gen_key()
    vote = make_vote(key, round_id=5, update_hash="xyz", voter_node_id=1,
                      decision=False, reason_code=RejectionReason.ZKP_FAILED)
    d = vote.to_dict()
    vote2 = ConsensusVote.from_dict(d)
    ok = verify_vote(key.public_key(), vote2)
    check("vote_json_roundtrip_still_verifies", ok)
    check("vote_json_roundtrip_preserves_decision", vote2.decision == vote.decision)


def test_tampered_vote_fails_verification():
    key = gen_key()
    vote = make_vote(key, round_id=5, update_hash="xyz", voter_node_id=1,
                      decision=True, reason_code=RejectionReason.OK)
    vote.decision = False  # tamper after signing, before sending
    ok = verify_vote(key.public_key(), vote)
    check("tampered_vote_fails_verification", not ok)


# ─────────────────────────────────────────────────────────────────
# Hard-rejection decision logic
# ─────────────────────────────────────────────────────────────────

def test_evaluate_accepts_valid_update():
    decision, reason = evaluate_update_for_vote(
        zkp_ok=True, expected_round=3, received_round=3,
        expected_hash="h1", received_hash="h1",
    )
    check("evaluate_accepts_valid", decision and reason == RejectionReason.OK)


def test_evaluate_rejects_wrong_round_even_if_zkp_ok():
    decision, reason = evaluate_update_for_vote(
        zkp_ok=True, expected_round=3, received_round=2,
        expected_hash="h1", received_hash="h1",
    )
    check(
        "evaluate_rejects_wrong_round",
        not decision and reason == RejectionReason.WRONG_ROUND,
        f"got decision={decision}, reason={reason}",
    )


def test_evaluate_rejects_hash_mismatch_even_if_zkp_ok():
    decision, reason = evaluate_update_for_vote(
        zkp_ok=True, expected_round=3, received_round=3,
        expected_hash="h1", received_hash="h2",
    )
    check(
        "evaluate_rejects_hash_mismatch",
        not decision and reason == RejectionReason.HASH_MISMATCH,
        f"got decision={decision}, reason={reason}",
    )


def test_evaluate_rejects_failed_zkp():
    decision, reason = evaluate_update_for_vote(
        zkp_ok=False, expected_round=3, received_round=3,
        expected_hash="h1", received_hash="h1",
    )
    check(
        "evaluate_rejects_failed_zkp",
        not decision and reason == RejectionReason.ZKP_FAILED,
        f"got decision={decision}, reason={reason}",
    )


def test_evaluate_checks_round_before_hash_before_zkp():
    """Order matters: a wrong-round message should be rejected as
    WRONG_ROUND, not accidentally reported as HASH_MISMATCH or
    ZKP_FAILED, so operators can tell replay/staleness apart from
    proof forgery in the logs/dashboard."""
    decision, reason = evaluate_update_for_vote(
        zkp_ok=False, expected_round=3, received_round=99,
        expected_hash="h1", received_hash="wrong",
    )
    check(
        "evaluate_checks_round_first",
        reason == RejectionReason.WRONG_ROUND,
        f"expected WRONG_ROUND to take priority, got {reason}",
    )


# ─────────────────────────────────────────────────────────────────
# Quorum tracking
# ─────────────────────────────────────────────────────────────────

def test_quorum_not_satisfied_below_threshold():
    key1, key2 = gen_key(), gen_key()
    tracker = QuorumTracker(round_id=1, update_hash="h", quorum_required=2)
    v1 = make_vote(key1, 1, "h", voter_node_id=1, decision=True, reason_code=RejectionReason.OK)
    tracker.add_vote(v1)
    check("quorum_not_satisfied_with_1_of_2", not tracker.is_satisfied)


def test_quorum_satisfied_at_threshold():
    key1, key2 = gen_key(), gen_key()
    tracker = QuorumTracker(round_id=1, update_hash="h", quorum_required=2)
    v1 = make_vote(key1, 1, "h", voter_node_id=1, decision=True, reason_code=RejectionReason.OK)
    v2 = make_vote(key2, 1, "h", voter_node_id=2, decision=True, reason_code=RejectionReason.OK)
    tracker.add_vote(v1)
    tracker.add_vote(v2)
    check("quorum_satisfied_at_2_of_2", tracker.is_satisfied)


def test_quorum_rejects_mismatched_round_or_hash_votes():
    key1 = gen_key()
    tracker = QuorumTracker(round_id=1, update_hash="h", quorum_required=1)
    wrong_round_vote = make_vote(key1, round_id=99, update_hash="h",
                                  voter_node_id=1, decision=True, reason_code=RejectionReason.OK)
    accepted = tracker.add_vote(wrong_round_vote)
    check(
        "quorum_rejects_wrong_round_vote",
        not accepted and not tracker.is_satisfied,
        "a vote for a different round_id was incorrectly accepted into this tracker",
    )


def test_quorum_ignores_duplicate_vote_from_same_node():
    """A node re-sending its vote (network retry) must not let it be
    counted twice, and a node should not be able to flip its vote."""
    key1 = gen_key()
    tracker = QuorumTracker(round_id=1, update_hash="h", quorum_required=2)
    v1 = make_vote(key1, 1, "h", voter_node_id=1, decision=True, reason_code=RejectionReason.OK)
    first = tracker.add_vote(v1)
    v1_flip = make_vote(key1, 1, "h", voter_node_id=1, decision=False, reason_code=RejectionReason.ZKP_FAILED)
    second = tracker.add_vote(v1_flip)
    cert = tracker.certificate()
    check(
        "quorum_ignores_duplicate_vote",
        first and not second and cert.accepted_by == [1] and cert.rejected_by == [],
        f"first={first}, second={second}, accepted={cert.accepted_by}, rejected={cert.rejected_by}",
    )


def test_quorum_certificate_reflects_mixed_votes():
    key1, key2, key3 = gen_key(), gen_key(), gen_key()
    tracker = QuorumTracker(round_id=7, update_hash="deadbeef", quorum_required=2)
    tracker.add_vote(make_vote(key1, 7, "deadbeef", 0, True, RejectionReason.OK))
    tracker.add_vote(make_vote(key2, 7, "deadbeef", 1, True, RejectionReason.OK))
    tracker.add_vote(make_vote(key3, 7, "deadbeef", 2, False, RejectionReason.ZKP_FAILED))
    cert = tracker.certificate()
    check(
        "quorum_certificate_mixed_votes",
        set(cert.accepted_by) == {0, 1} and set(cert.rejected_by) == {2} and cert.is_satisfied,
        f"accepted={cert.accepted_by}, rejected={cert.rejected_by}, satisfied={cert.is_satisfied}",
    )


def test_certificate_serialization_roundtrip():
    key1 = gen_key()
    tracker = QuorumTracker(round_id=1, update_hash="h", quorum_required=1)
    tracker.add_vote(make_vote(key1, 1, "h", 0, True, RejectionReason.OK))
    cert = tracker.certificate()
    d = cert.to_dict()
    cert2 = QuorumCertificate.from_dict(d)
    check(
        "certificate_json_roundtrip",
        cert2.is_satisfied == cert.is_satisfied and cert2.accepted_by == cert.accepted_by,
    )


# ─────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SecureFedHE-Consensus — Stage 1 Test Suite")
    print("=" * 60)

    print("\n--- Signing primitives ---")
    test_canonical_json_is_deterministic()
    test_sign_and_verify_roundtrip()
    test_verify_rejects_tampered_payload()
    test_verify_rejects_wrong_key()
    test_verify_rejects_garbage_signature()
    test_update_hash_is_deterministic_and_binds_both_parts()

    print("\n--- Vote creation/verification ---")
    test_make_vote_and_verify()
    test_vote_serialization_roundtrip()
    test_tampered_vote_fails_verification()

    print("\n--- Hard-rejection decision logic ---")
    test_evaluate_accepts_valid_update()
    test_evaluate_rejects_wrong_round_even_if_zkp_ok()
    test_evaluate_rejects_hash_mismatch_even_if_zkp_ok()
    test_evaluate_rejects_failed_zkp()
    test_evaluate_checks_round_before_hash_before_zkp()

    print("\n--- Quorum tracking ---")
    test_quorum_not_satisfied_below_threshold()
    test_quorum_satisfied_at_threshold()
    test_quorum_rejects_mismatched_round_or_hash_votes()
    test_quorum_ignores_duplicate_vote_from_same_node()
    test_quorum_certificate_reflects_mixed_votes()
    test_certificate_serialization_roundtrip()

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
