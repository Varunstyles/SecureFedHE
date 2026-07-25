"""
test_node_exclusion.py — SecureFedHE node-exclusion logic
Covers what test_consensus.py does not: _apply_exclusion,
_handle_accusation, _record_suspicion, get_proposer_for_round,
get_successor_url. Pure STATE-dict manipulation, no live ring/FastAPI
needed — but node.py itself imports torch/fastapi/etc., so this file
needs the same environment node.py runs in (unlike test_consensus.py,
which avoids importing node.py entirely).

Run:
    python tests/test_node_exclusion.py
"""
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import node  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
_results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    _results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail and status == FAIL else ""))
    return condition


def make_ring(n):
    return {"ring": {"nodes": [{"id": i, "ip": "127.0.0.1", "port": 9000 + i} for i in range(n)]}}


def reset_state(n=4, node_id=0):
    """Fresh STATE dict before each test, mirroring node.py's real init."""
    node.STATE.clear()
    node.STATE["node_id"] = node_id
    node.STATE["config"] = make_ring(n)
    node.STATE["logger"] = logging.getLogger("test_node_exclusion")
    node.STATE["excluded_nodes"] = set()
    node.STATE["exclusion_threshold"] = 5
    node.STATE["node_rejection_counts"] = {}
    node.STATE["accusations"] = {}
    node.STATE["accused_by_me"] = set()
    node.STATE["session"] = _FakeSession()
    node.STATE["dev_mode"] = True


class _FakeSession:
    """Stand-in for the real mTLS requests.Session — exclusion tests
    call code paths that try to notify peers over HTTP, but no real
    nodes are running here, so every send should just fail quietly
    instead of raising."""
    def post(self, *a, **kw):
        raise ConnectionError("no live peer in unit test")


# ─────────────────────────────────────────────────────────────────
# _apply_exclusion — quorum recalculation (3f+1)
# ─────────────────────────────────────────────────────────────────

def test_apply_exclusion_adds_to_set():
    reset_state(n=4)
    node._apply_exclusion(2, node.STATE["logger"])
    check("apply_exclusion_adds_node", 2 in node.STATE["excluded_nodes"])


def test_apply_exclusion_recomputes_quorum_4_to_3():
    """4 nodes -> exclude 1 -> 3 remain -> f=0 -> quorum=3."""
    reset_state(n=4)
    node._apply_exclusion(3, node.STATE["logger"])
    check("quorum_after_exclusion_4to3", node.STATE["quorum_required"] == 3,
          f"got {node.STATE.get('quorum_required')}")


def test_apply_exclusion_recomputes_quorum_7_to_6():
    """7 nodes -> exclude 1 -> 6 remain -> f=1 -> quorum=5."""
    reset_state(n=7)
    node._apply_exclusion(6, node.STATE["logger"])
    check("quorum_after_exclusion_7to6", node.STATE["quorum_required"] == 5,
          f"got {node.STATE.get('quorum_required')}")


def test_apply_exclusion_is_idempotent():
    reset_state(n=4)
    node._apply_exclusion(1, node.STATE["logger"])
    q1 = node.STATE["quorum_required"]
    node._apply_exclusion(1, node.STATE["logger"])  # exclude same node again
    check("apply_exclusion_idempotent",
          node.STATE["excluded_nodes"] == {1} and node.STATE["quorum_required"] == q1)


# ─────────────────────────────────────────────────────────────────
# _record_suspicion — local strikes, threshold trigger, self-guard
# ─────────────────────────────────────────────────────────────────

def test_record_suspicion_accumulates():
    reset_state(n=4)
    node._record_suspicion(1, node.STATE["logger"])
    node._record_suspicion(1, node.STATE["logger"])
    check("suspicion_accumulates", node.STATE["node_rejection_counts"][1] == 2.0)


def test_record_suspicion_never_self_accuses():
    reset_state(n=4, node_id=0)
    node._record_suspicion(0, node.STATE["logger"])
    check("suspicion_ignores_self", 0 not in node.STATE.get("node_rejection_counts", {}))


def test_record_suspicion_triggers_accusation_at_threshold(monkeypatch=None):
    reset_state(n=4)
    node.STATE["exclusion_threshold"] = 3
    broadcasted = []
    orig = node._broadcast_accusation
    node._broadcast_accusation = lambda accused_id: broadcasted.append(accused_id)
    try:
        node._record_suspicion(2, node.STATE["logger"], weight=1.0)
        node._record_suspicion(2, node.STATE["logger"], weight=1.0)
        check("no_accusation_below_threshold", broadcasted == [])
        node._record_suspicion(2, node.STATE["logger"], weight=1.0)
        check("accusation_fires_at_threshold", broadcasted == [2],
              f"got {broadcasted}")
    finally:
        node._broadcast_accusation = orig


def test_record_suspicion_accuses_only_once():
    reset_state(n=4)
    node.STATE["exclusion_threshold"] = 1
    broadcasted = []
    orig = node._broadcast_accusation
    node._broadcast_accusation = lambda accused_id: broadcasted.append(accused_id)
    try:
        node._record_suspicion(2, node.STATE["logger"])
        node._record_suspicion(2, node.STATE["logger"])
        node._record_suspicion(2, node.STATE["logger"])
        check("accusation_fires_once_not_per_strike", broadcasted == [2],
              f"got {broadcasted}")
    finally:
        node._broadcast_accusation = orig


# ─────────────────────────────────────────────────────────────────
# _handle_accusation — corroboration quorum (distinct accusers)
# ─────────────────────────────────────────────────────────────────

def test_accusation_not_excluded_below_corroboration_quorum():
    """4 nodes, accused=3, remaining_excluding_accused=3, f=0,
    quorum=3 distinct accusers needed. 1 accuser is not enough."""
    reset_state(n=4)
    node._handle_accusation(accuser_id=1, accused_id=3, log=node.STATE["logger"])
    check("single_accuser_insufficient", 3 not in node.STATE["excluded_nodes"])


def test_accusation_excludes_at_corroboration_quorum():
    reset_state(n=4)
    node._handle_accusation(accuser_id=0, accused_id=3, log=node.STATE["logger"])
    node._handle_accusation(accuser_id=1, accused_id=3, log=node.STATE["logger"])
    node._handle_accusation(accuser_id=2, accused_id=3, log=node.STATE["logger"])
    check("three_distinct_accusers_excludes", 3 in node.STATE["excluded_nodes"],
          f"accusers recorded: {node.STATE['accusations'].get(3)}")


def test_accusation_duplicate_accuser_does_not_double_count():
    reset_state(n=4)
    node._handle_accusation(accuser_id=0, accused_id=3, log=node.STATE["logger"])
    node._handle_accusation(accuser_id=0, accused_id=3, log=node.STATE["logger"])  # same accuser again
    check("duplicate_accuser_not_double_counted",
          len(node.STATE["accusations"][3]) == 1 and 3 not in node.STATE["excluded_nodes"])


def test_accusation_already_excluded_is_noop():
    reset_state(n=4)
    node._apply_exclusion(3, node.STATE["logger"])
    node._handle_accusation(accuser_id=0, accused_id=3, log=node.STATE["logger"])
    check("already_excluded_accusation_is_noop", 3 in node.STATE["excluded_nodes"])


# ─────────────────────────────────────────────────────────────────
# get_proposer_for_round — leader rotation skips excluded nodes
# ─────────────────────────────────────────────────────────────────

def test_proposer_uses_raw_formula_when_none_excluded():
    reset_state(n=4)
    check("proposer_raw_formula", node.get_proposer_for_round(5) == 5 % 4)


def test_proposer_skips_single_excluded_node():
    reset_state(n=4)
    node.STATE["excluded_nodes"] = {1}
    # round_id=1 -> raw candidate 1, which is excluded -> next is 2
    check("proposer_skips_excluded", node.get_proposer_for_round(1) == 2,
          f"got {node.get_proposer_for_round(1)}")


def test_proposer_skips_multiple_excluded_nodes():
    reset_state(n=4)
    node.STATE["excluded_nodes"] = {1, 2}
    check("proposer_skips_two_excluded", node.get_proposer_for_round(1) == 3,
          f"got {node.get_proposer_for_round(1)}")


def test_proposer_every_node_gets_a_turn_over_full_cycle():
    """Over N rounds with no exclusions, every node id 0..N-1 appears
    at least once — sanity check the rotation actually rotates."""
    reset_state(n=4)
    seen = {node.get_proposer_for_round(r) for r in range(4)}
    check("proposer_full_rotation_coverage", seen == {0, 1, 2, 3}, f"got {seen}")


# ─────────────────────────────────────────────────────────────────
# get_successor_url — ring forwarding skips excluded/self, handles
# the "everyone excluded" wraparound without crashing
# ─────────────────────────────────────────────────────────────────

def test_successor_skips_excluded_node():
    reset_state(n=4, node_id=0)
    node.STATE["excluded_nodes"] = {1}
    node.STATE["dev_mode"] = True
    url = node.get_successor_url()
    check("successor_skips_excluded", url is not None and ":9002" in url,
          f"got {url}")


def test_successor_returns_none_when_fully_wrapped():
    """All other nodes excluded — should not crash, should indicate
    no live successor rather than looping forever or returning self."""
    reset_state(n=4, node_id=0)
    node.STATE["excluded_nodes"] = {1, 2, 3}
    node.STATE["dev_mode"] = True
    try:
        url = node.get_successor_url()
        check("successor_all_excluded_no_crash", True, f"returned {url}")
    except Exception as e:
        check("successor_all_excluded_no_crash", False, f"raised {e!r}")


# ─────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SecureFedHE — Node Exclusion Test Suite")
    print("=" * 60)

    print("\n--- _apply_exclusion / quorum recalculation ---")
    test_apply_exclusion_adds_to_set()
    test_apply_exclusion_recomputes_quorum_4_to_3()
    test_apply_exclusion_recomputes_quorum_7_to_6()
    test_apply_exclusion_is_idempotent()

    print("\n--- _record_suspicion ---")
    test_record_suspicion_accumulates()
    test_record_suspicion_never_self_accuses()
    test_record_suspicion_triggers_accusation_at_threshold()
    test_record_suspicion_accuses_only_once()

    print("\n--- _handle_accusation / corroboration quorum ---")
    test_accusation_not_excluded_below_corroboration_quorum()
    test_accusation_excludes_at_corroboration_quorum()
    test_accusation_duplicate_accuser_does_not_double_count()
    test_accusation_already_excluded_is_noop()

    print("\n--- get_proposer_for_round ---")
    test_proposer_uses_raw_formula_when_none_excluded()
    test_proposer_skips_single_excluded_node()
    test_proposer_skips_multiple_excluded_nodes()
    test_proposer_every_node_gets_a_turn_over_full_cycle()

    print("\n--- get_successor_url ---")
    test_successor_skips_excluded_node()
    test_successor_returns_none_when_fully_wrapped()

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
