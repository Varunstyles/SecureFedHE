"""
consensus_engine.py — SecureFedHE-Consensus (Stage 1)
Signing, vote creation/verification, and quorum-detection logic for
update-level consensus. Uses each node's existing mTLS client RSA key
for signing — no new key material is introduced.

This module is intentionally free of FastAPI/network code: node.py
wires it into HTTP endpoints. Keeping this module network-agnostic
makes it independently unit-testable (see tests/test_consensus.py).
"""
from __future__ import annotations

from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature

from consensus_models import (
    ConsensusVote,
    QuorumCertificate,
    RejectionReason,
    canonical_json,
    sha256_hex,
)


def load_private_key(key_path: str) -> rsa.RSAPrivateKey:
    with open(key_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def load_public_key_from_cert(cert_path: str):
    from cryptography import x509
    with open(cert_path, "rb") as f:
        cert = x509.load_pem_x509_certificate(f.read())
    return cert.public_key()


def sign_payload(private_key: rsa.RSAPrivateKey, payload: dict) -> str:
    data = canonical_json(payload).encode("utf-8")
    signature = private_key.sign(
        data,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    return signature.hex()


def verify_signature(public_key, payload: dict, signature_hex: str) -> bool:
    try:
        signature = bytes.fromhex(signature_hex)
        data = canonical_json(payload).encode("utf-8")
        public_key.verify(
            signature,
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def compute_update_hash(proof_json: dict, enc_payload_summary: dict) -> str:
    """Deterministic hash binding the ZKP proof and a summary of the
    encrypted payload together, so a vote genuinely commits to THIS
    specific update and can't be replayed against a different one.

    enc_payload_summary should be small/stable (e.g. lengths and a
    hash of the ciphertext blobs) — NOT the raw ciphertext itself,
    to keep hashing cheap and avoid coupling this module to the
    encryption format.
    """
    combined = canonical_json({"proof": proof_json, "enc_summary": enc_payload_summary})
    return sha256_hex(combined.encode("utf-8"))


def make_vote(
    private_key: rsa.RSAPrivateKey,
    round_id: int,
    update_hash: str,
    voter_node_id: int,
    decision: bool,
    reason_code: str,
) -> ConsensusVote:
    vote = ConsensusVote(
        round_id=round_id,
        update_hash=update_hash,
        voter_node_id=voter_node_id,
        decision=decision,
        reason_code=reason_code,
        signature="",  # filled in below
    )
    vote.signature = sign_payload(private_key, vote.signing_payload())
    return vote


def verify_vote(public_key, vote: ConsensusVote) -> bool:
    return verify_signature(public_key, vote.signing_payload(), vote.signature)


class QuorumTracker:
    """Accumulates votes for a single (round_id, update_hash) pair and
    reports whether quorum has been reached. One instance per pending
    update; the caller (node.py) is responsible for keying these by
    (round_id, update_hash) and discarding stale ones.
    """

    def __init__(self, round_id: int, update_hash: str, quorum_required: int):
        self.round_id = round_id
        self.update_hash = update_hash
        self.quorum_required = quorum_required
        self._votes: dict[int, ConsensusVote] = {}  # voter_node_id -> vote

    def add_vote(self, vote: ConsensusVote) -> bool:
        """Returns True if the vote was newly recorded, False if it was
        a duplicate/stale vote from a node that already voted (first
        vote wins — a node should not be able to flip its vote to
        manipulate quorum formation after the fact)."""
        if vote.round_id != self.round_id or vote.update_hash != self.update_hash:
            return False
        if vote.voter_node_id in self._votes:
            return False
        self._votes[vote.voter_node_id] = vote
        return True

    def certificate(self) -> QuorumCertificate:
        accepted = [v.voter_node_id for v in self._votes.values() if v.decision]
        rejected = [v.voter_node_id for v in self._votes.values() if not v.decision]
        return QuorumCertificate(
            round_id=self.round_id,
            update_hash=self.update_hash,
            accepted_by=accepted,
            rejected_by=rejected,
            votes=[v.to_dict() for v in self._votes.values()],
            quorum_required=self.quorum_required,
        )

    @property
    def is_satisfied(self) -> bool:
        return self.certificate().is_satisfied


def evaluate_update_for_vote(
    zkp_ok: bool,
    expected_round: int,
    received_round: int,
    expected_hash: Optional[str],
    received_hash: str,
) -> tuple[bool, str]:
    """Hard-rejection logic (Stage 1 scope: no soft trust score yet —
    that is Stage 4). Returns (decision, reason_code).

    Order matters: check identity/round/hash consistency BEFORE trusting
    the ZKP result, since a stale or replayed message could carry a
    valid-looking proof for the wrong context.
    """
    if received_round != expected_round:
        return False, RejectionReason.WRONG_ROUND
    if expected_hash is not None and received_hash != expected_hash:
        return False, RejectionReason.HASH_MISMATCH
    if not zkp_ok:
        return False, RejectionReason.ZKP_FAILED
    return True, RejectionReason.OK
