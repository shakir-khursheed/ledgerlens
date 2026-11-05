from ledgerlens.generate import generate
from ledgerlens.matcher import match
from ledgerlens.models import Archetype


def test_matcher_has_no_false_positives_or_negatives():
    """Every clean payment auto-matches; no broken payment slips through.

    This is the invariant the whole system rests on: if the deterministic pass
    silently closes a real break, no amount of downstream reasoning helps.
    """
    for seed in range(5):
        ds = generate(500, seed=seed)
        m = match(ds.payments, ds.rows)
        leaked = [b for b in m.breaks if ds.truth[b.payment_ids[0]] is Archetype.CLEAN]
        swallowed = [p for p in m.auto_matched if ds.truth[p] is not Archetype.CLEAN]
        assert leaked == [], f"clean rows sent to the agent: {leaked[:3]}"
        assert swallowed == [], f"real breaks auto-closed: {swallowed[:3]}"


def test_auto_match_rate_is_worth_having():
    m = match(*(lambda d: (d.payments, d.rows))(generate(800)))
    assert m.auto_match_rate > 0.45


def test_missing_rows_become_breaks_with_zero_observed():
    ds = generate(400, seed=3)
    m = match(ds.payments, ds.rows)
    missing = [
        b for b in m.breaks
        if ds.truth[b.payment_ids[0]] is Archetype.MISSING_IN_BANK
    ]
    assert missing
    assert all(b.observed_net_paise == 0 and b.settlement_row_ids == () for b in missing)
