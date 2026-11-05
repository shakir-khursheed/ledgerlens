"""Core domain types for settlement reconciliation.

Money is always integer paise. Floats are never used for amounts anywhere in
this codebase -- a 1-paise float drift is a real break in production.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from enum import Enum
from typing import Any


class Method(str, Enum):
    UPI = "upi"
    CARD = "card"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    INTL_CARD = "intl_card"


class PaymentStatus(str, Enum):
    CAPTURED = "captured"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"
    DISPUTED = "disputed"


class Archetype(str, Enum):
    """Ground-truth label for a reconciliation break.

    These are the break families an ops analyst actually sees. The eval harness
    scores the agent per-archetype, because a 90% average can still mean the
    agent is blind to the one family that loses money.
    """

    CLEAN = "clean"
    FEE_SCHEDULE_DELTA = "fee_schedule_delta"
    GST_ROUNDING = "gst_rounding"
    T_PLUS_DELAY = "t_plus_delay"
    PARTIAL_REFUND_NETTED = "partial_refund_netted"
    CHARGEBACK_WITHHELD = "chargeback_withheld"
    DUPLICATE_BANK_ROW = "duplicate_bank_row"
    SPLIT_SETTLEMENT = "split_settlement"
    FX_ROUNDING = "fx_rounding"
    MISSING_IN_BANK = "missing_in_bank"
    #: Never a ground-truth label -- the agent saying "I could not explain this".
    UNKNOWN = "unknown"


#: Archetypes where the correct action is "a human must look at this".
#: The agent proposing a silent adjustment here is the expensive failure mode.
ESCALATE_ONLY: frozenset[Archetype] = frozenset(
    {Archetype.MISSING_IN_BANK, Archetype.CHARGEBACK_WITHHELD}
)


@dataclass(frozen=True)
class Payment:
    payment_id: str
    order_id: str
    merchant_id: str
    captured_on: date
    amount_paise: int
    method: Method
    currency: str = "INR"
    status: PaymentStatus = PaymentStatus.CAPTURED
    refunded_paise: int = 0
    fx_rate_bp: int | None = None  # basis points, for non-INR capture


@dataclass(frozen=True)
class SettlementRow:
    """One row of the bank/PG settlement file."""

    row_id: str
    utr: str
    settled_on: date
    payment_ids: tuple[str, ...]
    gross_paise: int
    fee_paise: int
    tax_paise: int
    net_paise: int


@dataclass
class Break:
    """A discrepancy the deterministic matcher could not close by itself."""

    break_id: str
    payment_ids: tuple[str, ...]
    settlement_row_ids: tuple[str, ...]
    expected_net_paise: int
    observed_net_paise: int
    note: str
    truth: Archetype | None = None  # populated only in generated eval data

    @property
    def delta_paise(self) -> int:
        return self.observed_net_paise - self.expected_net_paise


@dataclass
class Decision:
    """What the agent concluded. Deliberately not a free-text blob."""

    break_id: str
    category: Archetype
    explanation: str
    confidence: float
    proposed_adjustment_paise: int = 0
    needs_human: bool = False
    evidence: list[str] = field(default_factory=list)
    tool_calls: int = 0

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d
