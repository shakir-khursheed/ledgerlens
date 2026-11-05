"""Contract tests for the LLM arm that do not need credentials.

The model is not available in CI, but the parsing and the safety clamp are the
parts most likely to break, and they are pure functions.
"""
from ledgerlens.agent import ClaudeReasoner
from ledgerlens.models import Archetype, Break
from ledgerlens.tools import Workspace
from ledgerlens.generate import generate

BRK = Break("brk_1", ("pay_00000",), ("row_1",), 1000, 900, "net differs by -100 paise")


def test_silent_agent_is_an_escalation_not_a_guess():
    d = ClaudeReasoner._to_decision(BRK, {}, 0)
    assert d.category is Archetype.UNKNOWN
    assert d.needs_human and d.confidence == 0.0


def test_unknown_category_string_is_coerced_not_crashed():
    d = ClaudeReasoner._to_decision(BRK, {"category": "vibes", "confidence": 0.9}, 3)
    assert d.category is Archetype.UNKNOWN


def test_adjustment_over_missing_money_is_clamped_away():
    """Even if the model insists, we do not let it paper over missing money."""
    d = ClaudeReasoner._to_decision(
        BRK,
        {
            "category": "missing_in_bank",
            "proposed_adjustment_paise": -100_000,
            "needs_human": False,
            "confidence": 0.99,
        },
        5,
    )
    assert d.proposed_adjustment_paise == 0
    assert d.needs_human is True


def test_tools_are_read_only():
    """No tool on the workspace may mutate ledger or settlement state."""
    ds = generate(50)
    ws = Workspace(ds.payments, ds.rows)
    snapshot = (len(ws._payments), len(ws._rows))
    for name in ("get_payment", "get_settlement_rows", "get_refunds", "get_disputes"):
        getattr(ws, name)("pay_00000")
    ws.get_fee_schedule("card")
    assert (len(ws._payments), len(ws._rows)) == snapshot
    assert ws.call_count == 5
