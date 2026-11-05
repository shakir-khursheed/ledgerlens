"""The agent's tool surface.

Two properties matter more than the tool list itself:

1. Every tool is read-only. The agent cannot move money, write to the ledger or
   edit a settlement file. The most it can do is *propose* an adjustment, which
   a human or a downstream rule applies. In a payments company this is the
   difference between a shippable agent and one that never leaves the demo.
2. Every call is recorded. A decision you cannot audit is not a decision you
   can put in front of a finance team.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .feeschedule import (
    CURRENT_MDR_BP,
    GST_BP,
    LEGACY_MDR_BP,
    T_PLUS,
    expected_net,
    expected_window,
    fee_for,
)
from .models import Method, Payment, SettlementRow


class Workspace:
    """Read-only view over one reconciliation batch, plus a call log."""

    def __init__(self, payments: list[Payment], rows: list[SettlementRow]) -> None:
        self._payments = {p.payment_id: p for p in payments}
        self._rows = {r.row_id: r for r in rows}
        self._by_payment: dict[str, list[SettlementRow]] = defaultdict(list)
        self._by_utr: dict[str, list[SettlementRow]] = defaultdict(list)
        for r in rows:
            self._by_utr[r.utr].append(r)
            for pid in r.payment_ids:
                self._by_payment[pid].append(r)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _log(self, name: str, **kw: Any) -> None:
        self.calls.append((name, kw))

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def reset_log(self) -> None:
        self.calls.clear()

    # ---- tools -----------------------------------------------------------

    def get_payment(self, payment_id: str) -> dict[str, Any]:
        """Ledger view of a payment: amount, method, status, refunds, FX."""
        self._log("get_payment", payment_id=payment_id)
        p = self._payments.get(payment_id)
        if p is None:
            return {"error": "unknown payment_id"}
        return {
            "payment_id": p.payment_id,
            "merchant_id": p.merchant_id,
            "captured_on": p.captured_on.isoformat(),
            "amount_paise": p.amount_paise,
            "method": p.method.value,
            "currency": p.currency,
            "status": p.status.value,
            "refunded_paise": p.refunded_paise,
            "fx_rate_bp": p.fx_rate_bp,
        }

    def get_settlement_rows(self, payment_id: str) -> list[dict[str, Any]]:
        """Every bank settlement row that references this payment."""
        self._log("get_settlement_rows", payment_id=payment_id)
        return [self._row_json(r) for r in sorted(self._by_payment.get(payment_id, []), key=lambda r: r.row_id)]

    def find_rows_by_utr(self, utr: str) -> list[dict[str, Any]]:
        """All rows sharing a bank UTR -- how duplicate bank messages surface."""
        self._log("find_rows_by_utr", utr=utr)
        return [self._row_json(r) for r in self._by_utr.get(utr, [])]

    def get_fee_schedule(self, method: str) -> dict[str, Any]:
        """Current and legacy MDR, GST rate and expected settlement lag."""
        self._log("get_fee_schedule", method=method)
        try:
            m = Method(method)
        except ValueError:
            return {"error": "unknown method"}
        return {
            "method": m.value,
            "current_mdr_bp": CURRENT_MDR_BP[m],
            "legacy_mdr_bp": LEGACY_MDR_BP[m],
            "repriced": CURRENT_MDR_BP[m] != LEGACY_MDR_BP[m],
            "gst_bp": GST_BP,
            "t_plus_days": T_PLUS[m],
        }

    def compute_expected_net(
        self, amount_paise: int, method: str, legacy: bool = False
    ) -> dict[str, Any]:
        """Recompute what the payout should have been, under either rate card."""
        self._log(
            "compute_expected_net", amount_paise=amount_paise, method=method, legacy=legacy
        )
        m = Method(method)
        fee, gst = fee_for(amount_paise, m, legacy=legacy)
        return {
            "fee_paise": fee,
            "gst_paise": gst,
            "net_paise": expected_net(amount_paise, m, legacy=legacy),
            "rate_card": "legacy" if legacy else "current",
        }

    def get_expected_window(self, payment_id: str) -> dict[str, Any]:
        """The date range in which this capture should have settled."""
        self._log("get_expected_window", payment_id=payment_id)
        p = self._payments[payment_id]
        lo, hi = expected_window(p.captured_on, p.method)
        return {"due_from": lo.isoformat(), "due_to": hi.isoformat()}

    def get_refunds(self, payment_id: str) -> dict[str, Any]:
        """Refunds booked against the payment, which the bank nets off."""
        self._log("get_refunds", payment_id=payment_id)
        p = self._payments[payment_id]
        return {"refunded_paise": p.refunded_paise, "status": p.status.value}

    def get_disputes(self, payment_id: str) -> dict[str, Any]:
        """Open chargebacks. Disputed funds are withheld, not lost."""
        self._log("get_disputes", payment_id=payment_id)
        p = self._payments[payment_id]
        disputed = p.status.value == "disputed"
        return {
            "has_open_dispute": disputed,
            "withheld_paise": p.amount_paise if disputed else 0,
        }

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _row_json(r: SettlementRow) -> dict[str, Any]:
        return {
            "row_id": r.row_id,
            "utr": r.utr,
            "settled_on": r.settled_on.isoformat(),
            "payment_ids": list(r.payment_ids),
            "gross_paise": r.gross_paise,
            "fee_paise": r.fee_paise,
            "tax_paise": r.tax_paise,
            "net_paise": r.net_paise,
        }


#: Names exposed to the LLM, in the order they are usually useful.
TOOL_NAMES = [
    "get_payment",
    "get_settlement_rows",
    "get_expected_window",
    "get_fee_schedule",
    "compute_expected_net",
    "get_refunds",
    "get_disputes",
    "find_rows_by_utr",
]
