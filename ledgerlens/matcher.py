"""Deterministic first pass.

Design rule for the whole system: the LLM never touches anything a rule can
close. Rules are free, auditable and never hallucinate, so they handle the
~50-60% of rows that match exactly. The agent only sees the residual. That
keeps cost per reconciled batch near zero and keeps the blast radius small.
"""
from __future__ import annotations

from collections import defaultdict

from .feeschedule import expected_window
from .generate import expected_net_for
from .models import Break, Payment, SettlementRow


class MatchResult:
    def __init__(self) -> None:
        self.auto_matched: list[str] = []
        self.breaks: list[Break] = []
        self.orphan_rows: list[str] = []

    @property
    def auto_match_rate(self) -> float:
        total = len(self.auto_matched) + len(self.breaks)
        return len(self.auto_matched) / total if total else 0.0


def match(payments: list[Payment], rows: list[SettlementRow]) -> MatchResult:
    by_payment: dict[str, list[SettlementRow]] = defaultdict(list)
    for r in rows:
        for pid in r.payment_ids:
            by_payment[pid].append(r)

    res = MatchResult()
    known = {p.payment_id for p in payments}
    res.orphan_rows = sorted(
        r.row_id for r in rows if not set(r.payment_ids) & known
    )

    for pay in payments:
        mine = sorted(by_payment.get(pay.payment_id, []), key=lambda r: r.row_id)
        expected = expected_net_for(pay)
        observed = sum(r.net_paise for r in mine)

        if not mine:
            res.breaks.append(
                Break(
                    break_id=f"brk_{pay.payment_id}",
                    payment_ids=(pay.payment_id,),
                    settlement_row_ids=(),
                    expected_net_paise=expected,
                    observed_net_paise=0,
                    note="no settlement row references this payment",
                )
            )
            continue

        due_from, due_to = expected_window(pay.captured_on, pay.method)
        earliest = min(r.settled_on for r in mine)
        in_window = due_from <= earliest <= due_to

        if len(mine) == 1 and observed == expected and in_window:
            res.auto_matched.append(pay.payment_id)
            continue

        bits = []
        if len(mine) > 1:
            bits.append(f"{len(mine)} settlement rows reference this payment")
        if observed != expected:
            bits.append(f"net differs by {observed - expected} paise")
        if not in_window:
            bits.append(
                f"settled {earliest.isoformat()}, expected by {due_to.isoformat()}"
            )
        res.breaks.append(
            Break(
                break_id=f"brk_{pay.payment_id}",
                payment_ids=(pay.payment_id,),
                settlement_row_ids=tuple(r.row_id for r in mine),
                expected_net_paise=expected,
                observed_net_paise=observed,
                note="; ".join(bits),
            )
        )
    return res
