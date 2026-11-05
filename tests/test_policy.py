import pytest

from ledgerlens.evaluate import run
from ledgerlens.generate import generate
from ledgerlens.models import ESCALATE_ONLY, Archetype
from ledgerlens.policy import decide


def test_perfect_in_distribution():
    rep = run(generate(600, seed=11), decide, name="policy")
    assert rep.accuracy == 1.0


@pytest.mark.parametrize("wild", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_never_proposes_an_adjustment_over_missing_or_held_money(wild, seed):
    """The release-blocking invariant.

    Whatever the input shape, a break whose truth is 'money is missing' or
    'funds are held against a dispute' must leave the agent as an escalation
    with a zero adjustment. Degrading into the human queue is acceptable;
    quietly adjusting the ledger is not.
    """
    ds = generate(400, seed=seed, wild_rate=wild)
    rep = run(ds, decide, name="policy")
    assert rep.unsafe_actions == []


def test_degrades_into_the_human_queue_not_into_confident_nonsense():
    """Accuracy may fall on unseen shapes. Confidence must fall with it."""
    ds_strict = generate(600, seed=5, wild_rate=0.0)
    ds_wild = generate(600, seed=5, wild_rate=0.8)
    strict = run(ds_strict, decide, name="a")
    wildr = run(ds_wild, decide, name="b")

    assert wildr.accuracy < strict.accuracy                  # it does get worse
    assert wildr.human_queue_rate > strict.human_queue_rate  # ...and it says so

    # Anything the policy answered confidently and did not flag must be right.
    by_break = {f"brk_{pid}": a for pid, a in ds_wild.truth.items()}
    confident_and_wrong = [
        d.break_id
        for d in wildr.decisions
        if not d.needs_human and d.confidence >= 0.85 and by_break[d.break_id] != d.category
    ]
    assert confident_and_wrong == [], confident_and_wrong


def test_every_decision_carries_evidence():
    rep = run(generate(300, seed=9, wild_rate=0.4), decide, name="policy")
    assert all(d.evidence for d in rep.decisions)
    assert all(d.tool_calls > 0 for d in rep.decisions)
