import pytest

from ledgerlens.feeschedule import expected_net, fee_for, mul_bp
from ledgerlens.models import Method


def test_half_up_not_bankers():
    # Python's round() gives 0 here; settlement files do not.
    assert mul_bp(10_000, 50) == 50          # exact
    assert mul_bp(5_000, 1) == 1             # 0.5 paise rounds up
    assert mul_bp(4_999, 1) == 0             # 0.4999 rounds down
    assert mul_bp(-5_000, 1) == -1           # symmetric about zero


def test_fee_and_gst_are_integers():
    fee, gst = fee_for(123_457, Method.CARD)
    assert isinstance(fee, int) and isinstance(gst, int)
    assert gst == mul_bp(fee, 1800)


def test_net_is_gross_minus_fee_and_tax():
    for amount in (4_900, 100_000, 2_500_000, 1):
        for m in Method:
            fee, gst = fee_for(amount, m)
            assert expected_net(amount, m) == amount - fee - gst


def test_upi_is_zero_rated():
    assert fee_for(500_000, Method.UPI) == (0, 0)
    assert expected_net(500_000, Method.UPI) == 500_000


@pytest.mark.parametrize("amount", [1, 99, 100_000, 9_999_999])
def test_legacy_card_is_never_cheaper(amount):
    assert expected_net(amount, Method.CARD, legacy=True) <= expected_net(amount, Method.CARD)
