"""
consensus_models.py — SecureFedHE-Consensus (Stage 1)
Typed data structures for update-level quorum consensus: proposals,
votes, and quorum certificates. Deliberately dependency-light (stdlib
only + dataclasses) so it can be imported by both node.py and the
dashboard without pulling in heavy ML/crypto libraries.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


def canonical_json(obj: dict) -> str:
    """Deterministic JSON serialization for signing: sorted keys, no
    whitespace ambiguity. Both signer and verifier must produce byte-
    identical output for the same logical content."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class UpdateProposal:
    """Broadcast by the node submitting an update, so peers know what
    they are being asked to vote on before the full payload arrives
    (or alongside it, for this ring topology)."""
    round_id: int
    origin_node_id: int
    sender_node_id: int
    update_hash: str          # sha256 of the ZKP proof JSON + enc payload hash
    zkp_public_inputs: list   # proof.public_inputs, so peers can sanity-check
    timestamp: float = field(default_factory=time.time)

    def signing_payload(self) -> dict:
        return {
            "round_id": self.round_id,
            "origin_node_id": self.origin_node_id,
            "sender_node_id": self.sender_node_id,
            "update_hash": self.update_hash,
            "zkp_public_inputs": self.zkp_public_inputs,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UpdateProposal":
        return cls(
            round_id=d["round_id"],
            origin_node_id=d["origin_node_id"],
            sender_node_id=d["sender_node_id"],
            update_hash=d["update_hash"],
            zkp_public_inputs=d["zkp_public_inputs"],
            timestamp=d.get("timestamp", time.time()),
        )


@dataclass
class ConsensusVote:
    """A single node's signed ACCEPT/REJECT decision on a proposal."""
    round_id: int
    update_hash: str
    voter_node_id: int
    decision: bool            # True = ACCEPT, False = REJECT
    reason_code: str          # e.g. "zkp_ok", "zkp_failed", "hash_mismatch"
    signature: str            # hex-encoded signature over signing_payload()
    timestamp: float = field(default_factory=time.time)

    def signing_payload(self) -> dict:
        return {
            "round_id": self.round_id,
            "update_hash": self.update_hash,
            "voter_node_id": self.voter_node_id,
            "decision": self.decision,
            "reason_code": self.reason_code,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConsensusVote":
        return cls(
            round_id=d["round_id"],
            update_hash=d["update_hash"],
            voter_node_id=d["voter_node_id"],
            decision=d["decision"],
            reason_code=d["reason_code"],
            signature=d["signature"],
            timestamp=d.get("timestamp", time.time()),
        )


@dataclass
class QuorumCertificate:
    """Formed once enough ACCEPT votes are collected for a given
    update. This is the artifact that gates aggregation: an update
    without a valid certificate must not be aggregated."""
    round_id: int
    update_hash: str
    accepted_by: list          # list[int] of node_ids that voted ACCEPT
    rejected_by: list          # list[int] of node_ids that voted REJECT
    votes: list                # list[dict] — the raw ConsensusVote.to_dict()s
    quorum_required: int
    formed_at: float = field(default_factory=time.time)

    @property
    def is_satisfied(self) -> bool:
        return len(self.accepted_by) >= self.quorum_required

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QuorumCertificate":
        return cls(
            round_id=d["round_id"],
            update_hash=d["update_hash"],
            accepted_by=list(d["accepted_by"]),
            rejected_by=list(d["rejected_by"]),
            votes=list(d["votes"]),
            quorum_required=d["quorum_required"],
            formed_at=d.get("formed_at", time.time()),
        )


class RejectionReason:
    """Reason codes for REJECT votes — keep these stable strings since
    they get logged and shown on the dashboard."""
    ZKP_FAILED = "zkp_failed"
    HASH_MISMATCH = "hash_mismatch"
    BAD_SIGNATURE = "bad_signature"
    WRONG_ROUND = "wrong_round"
    UNKNOWN_SENDER = "unknown_sender"
    TIMEOUT = "timeout"
    OK = "zkp_ok"
    PREDICTION_DISAGREEMENT = "prediction_disagreement"
    PER_CLASS_AGREEMENT_GAP = "per_class_agreement_gap"
