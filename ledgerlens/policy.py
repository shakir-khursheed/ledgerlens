"""The offline reasoner: an explicit decision policy over the same tools.

This exists for three reasons, in order of importance:

1. It is the control arm. If a deterministic policy already resolves a break
   family, routing it to an LLM is waste. The eval runs both and reports the
   delta, so "do we need a model here at all?" is answered with a number.
2. It makes the repo runnable, and the test suite meaningful, with no API key.
3. It is the fallback path in production when the model is unavailable or the
   per-batch cost budget is exhausted.
"""
from __future__ import annotations

from datetime import date

from .models import Archetype, Break, Decision
from .tools import Workspace

GST_TOLERANCE_PAISE = 3
FX_TOLERANCE_PAISE = 25


def decide(brk: Break, ws: Workspace) -> Decision:
    pid = brk.payment_ids[0]
    before = ws.call_count
    ev: list[str] = []

    pay = ws.get_payment(pid)
    rows = ws.get_settlement_rows(pid)
    amount = pay["amount_paise"]
    method = pay["method"]
    observed = sum(r["net_paise"] for r in rows)
    delta = observed - brk.expected_net_paise

    def done(cat: Archetype, why: str, conf: float, adj: int = 0, human: bool = False) -> Decision:
        return Decision(
            break_id=brk.break_id,
            category=cat,
            explanation=why,
            confidence=conf,
            proposed_adjustment_paise=adj,
            needs_human=human,
            evidence=ev,
            tool_calls=ws.call_count - before,
        )

    # 1. Nothing came back from the bank at all.
    if not rows:
        ev.append("no settlement row references this payment")
        return done(
            Archetype.MISSING_IN_BANK,
            "Captured but never settled. This is money owed, not a reporting "
            "artefact -- raise with the bank rather than adjusting the ledger.",
            0.95,
            adj=0,
            human=True,
        )

    # 2. Disputed funds are withheld, not lost. Never self-adjust these.
    disputes = ws.get_disputes(pid)
    if disputes["has_open_dispute"]:
        ev.append(f"open dispute, {disputes['withheld_paise']} paise withheld")
        return done(
            Archetype.CHARGEBACK_WITHHELD,
            "Payout withheld against an open chargeback. Expected to reverse on "
            "dispute closure; hold in suspense, do not write off.",
            0.93,
            adj=0,
            human=True,
        )

    # 3. More than one bank row: duplicate message or a genuine split.
    if len(rows) > 1:
        utrs = {r["utr"] for r in rows}
        nets = {r["net_paise"] for r in rows}
        if len(utrs) == 1 and len(nets) == 1 and len(rows) == 2:
            same = ws.find_rows_by_utr(rows[0]["utr"])
            ev.append(f"{len(same)} rows share UTR {rows[0]['utr']} with identical net")
            return done(
                Archetype.DUPLICATE_BANK_ROW,
                "The bank file repeats one credit under the same UTR. Drop the "
                "duplicate row; the underlying payout happened once.",
                0.90,
                adj=-rows[0]["net_paise"],
            )
        gross_sum = sum(r["gross_paise"] for r in rows)
        ev.append(f"{len(rows)} rows, gross sum {gross_sum} vs capture {amount}")
        if gross_sum == amount:
            return done(
                Archetype.SPLIT_SETTLEMENT,
                "One capture settled across multiple batches. Gross reconciles; "
                "the residual is per-row fee rounding, not a shortfall.",
                0.90,
                adj=delta,
            )

    # 4. Refund netted off the payout.
    refunds = ws.get_refunds(pid)
    if refunds["refunded_paise"] > 0 and delta == -refunds["refunded_paise"]:
        ev.append(f"refund of {refunds['refunded_paise']} paise exactly explains the delta")
        return done(
            Archetype.PARTIAL_REFUND_NETTED,
            "The bank netted a booked refund off this payout. Ledger and bank "
            "agree once the refund is applied; no adjustment needed.",
            0.95,
        )

    # 5. Settlement priced on the pre-revision rate card.
    sched = ws.get_fee_schedule(method)
    if sched.get("repriced") and delta != 0:
        legacy = ws.compute_expected_net(amount, method, legacy=True)
        if observed == legacy["net_paise"]:
            ev.append(
                f"observed net matches legacy card ({sched['legacy_mdr_bp']}bp) "
                f"not current ({sched['current_mdr_bp']}bp)"
            )
            return done(
                Archetype.FEE_SCHEDULE_DELTA,
                "Bank priced this at the pre-revision MDR. Recoverable: bill the "
                "difference back rather than absorbing it.",
                0.95,
                adj=-delta,
            )

    # 6. The amount is right (to within rounding) and only the date is wrong.
    #    Checked *before* the rounding branch: when a payout is both late and a
    #    paise off, the late payout is the material fact and the paise is noise.
    #    Getting that order wrong produced confidently-mislabelled SLA misses.
    win = ws.get_expected_window(pid)
    earliest = min(r["settled_on"] for r in rows)
    if earliest > win["due_to"] and abs(delta) <= GST_TOLERANCE_PAISE:
        late = (date.fromisoformat(earliest) - date.fromisoformat(win["due_to"])).days
        ev.append(f"settled {earliest}, {late}d past the T+{sched['t_plus_days']} window")
        exact = delta == 0
        return done(
            Archetype.T_PLUS_DELAY,
            "Full amount settled, just late. A reconciliation timing break, not a "
            "money break -- worth tracking as a bank SLA miss."
            + ("" if exact else f" Residual {delta}p is rounding, not shortfall."),
            0.90 if exact else 0.78,
        )

    # 7/8. Sub-rupee drift: GST rounding first, then FX conversion.
    if abs(delta) <= GST_TOLERANCE_PAISE:
        ev.append(f"delta of {delta} paise is within GST rounding tolerance")
        return done(
            Archetype.GST_ROUNDING,
            "Paise-level drift from rounding GST on the fee in a different "
            "direction. Immaterial individually; sweep to a rounding account.",
            0.88,
        )
    if method == "intl_card" and abs(delta) <= FX_TOLERANCE_PAISE:
        ev.append(f"intl card, fx_rate_bp={pay['fx_rate_bp']}, delta {delta} paise")
        return done(
            Archetype.FX_ROUNDING,
            "Cross-currency conversion rounding on an international capture. "
            "Expected drift at this rate; sweep to FX variance.",
            0.85,
        )

    ev.append(f"unexplained delta of {delta} paise")
    return done(
        Archetype.UNKNOWN,
        "No rule explains this delta. Routed to a human rather than guessed at.",
        0.30,
        human=True,
    )
