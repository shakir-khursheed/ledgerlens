"""Synthetic but realistically-shaped settlement data with ground-truth labels.

Real settlement files cannot be committed to a public repo, so the eval set is
generated. The point is not realism of merchant names -- it is that the *break
families* and their relative frequencies match what an ops analyst sees, so a
score on this set means something.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from .feeschedule import T_PLUS, expected_net, fee_for, mul_bp
from .models import Archetype, Method, Payment, PaymentStatus, SettlementRow


@dataclass
class Dataset:
    payments: list[Payment]
    rows: list[SettlementRow]
    truth: dict[str, Archetype]
    #: payment_ids whose break was perturbed into a shape the hand-written
    #: policy was never designed for. This is the held-out split that stops
    #: the eval from grading the policy against its own generator.
    wild: set[str] = field(default_factory=set)

#: Rough real-world mix. Clean dominates; the long tail is where the cost is.
MIX: list[tuple[Archetype, float]] = [
    (Archetype.CLEAN, 0.52),
    (Archetype.FEE_SCHEDULE_DELTA, 0.11),
    (Archetype.GST_ROUNDING, 0.08),
    (Archetype.T_PLUS_DELAY, 0.07),
    (Archetype.PARTIAL_REFUND_NETTED, 0.07),
    (Archetype.SPLIT_SETTLEMENT, 0.05),
    (Archetype.DUPLICATE_BANK_ROW, 0.04),
    (Archetype.FX_ROUNDING, 0.03),
    (Archetype.CHARGEBACK_WITHHELD, 0.02),
    (Archetype.MISSING_IN_BANK, 0.01),
]

METHOD_MIX: list[tuple[Method, float]] = [
    (Method.UPI, 0.55),
    (Method.CARD, 0.25),
    (Method.NETBANKING, 0.10),
    (Method.WALLET, 0.07),
    (Method.INTL_CARD, 0.03),
]


def _pick(rng: random.Random, weighted: list[tuple[object, float]]):
    r = rng.random()
    acc = 0.0
    for item, w in weighted:
        acc += w
        if r <= acc:
            return item
    return weighted[-1][0]


def _utr(rng: random.Random) -> str:
    return f"UTR{rng.randrange(10**11, 10**12)}"


def generate(
    n: int = 400,
    *,
    seed: int = 7,
    start: date = date(2026, 9, 1),
    wild_rate: float = 0.0,
) -> Dataset:
    """Build a labelled batch.

    ``wild_rate`` is the share of non-clean payments whose break is reshaped
    into a variant the deterministic policy was not written against: a
    third-party rate card, a refund clawed back net of its own fee, a split
    across three batches, a duplicate credit issued under a fresh UTR. These
    are all real shapes; they are held out precisely because hand-written rules
    do not generalise to them and an LLM arm should.
    """
    rng = random.Random(seed)
    payments: list[Payment] = []
    rows: list[SettlementRow] = []
    truth: dict[str, Archetype] = {}
    wild: set[str] = set()

    for i in range(n):
        pid = f"pay_{i:05d}"
        arche: Archetype = _pick(rng, MIX)  # type: ignore[assignment]
        method: Method = _pick(rng, METHOD_MIX)  # type: ignore[assignment]
        if arche is Archetype.FX_ROUNDING:
            method = Method.INTL_CARD
        elif arche is Archetype.FEE_SCHEDULE_DELTA:
            # only CARD and WALLET were repriced, so the break can only exist there
            method = rng.choice([Method.CARD, Method.WALLET])
        captured = start + timedelta(days=rng.randrange(0, 28))
        amount = rng.randrange(4_900, 25_000_00)  # Rs 49 to Rs 25k, in paise

        status = PaymentStatus.CAPTURED
        refunded = 0
        if arche is Archetype.PARTIAL_REFUND_NETTED:
            status = PaymentStatus.PARTIALLY_REFUNDED
            refunded = max(100, int(amount * rng.uniform(0.15, 0.6)))
        elif arche is Archetype.CHARGEBACK_WITHHELD:
            status = PaymentStatus.DISPUTED

        pay = Payment(
            payment_id=pid,
            order_id=f"order_{i:05d}",
            merchant_id=f"acc_{rng.randrange(1, 9)}",
            captured_on=captured,
            amount_paise=amount,
            method=method,
            status=status,
            refunded_paise=refunded,
            fx_rate_bp=rng.randrange(880_000, 920_000) if method is Method.INTL_CARD else None,
        )
        payments.append(pay)
        truth[pid] = arche

        is_wild = arche is not Archetype.CLEAN and rng.random() < wild_rate
        if is_wild:
            wild.add(pid)

        settled = captured + timedelta(days=T_PLUS[method])
        fee, gst = fee_for(amount, method)
        net = amount - fee - gst
        rid = f"row_{i:05d}"

        def row(**kw) -> SettlementRow:
            base = dict(
                row_id=rid,
                utr=_utr(rng),
                settled_on=settled,
                payment_ids=(pid,),
                gross_paise=amount,
                fee_paise=fee,
                tax_paise=gst,
                net_paise=net,
            )
            base.update(kw)
            return SettlementRow(**base)  # type: ignore[arg-type]

        if arche is Archetype.CLEAN:
            rows.append(row())

        elif arche is Archetype.MISSING_IN_BANK:
            pass  # no settlement row at all

        elif arche is Archetype.FEE_SCHEDULE_DELTA:
            if is_wild:
                # a bank-specific override rate that appears in no rate card we hold
                bp = rng.choice([245, 265, 310])
                wfee = mul_bp(amount, bp)
                wgst = mul_bp(wfee, 1800)
                rows.append(
                    row(fee_paise=wfee, tax_paise=wgst, net_paise=amount - wfee - wgst)
                )
            else:
                lfee, lgst = fee_for(amount, method, legacy=True)
                rows.append(
                    row(fee_paise=lfee, tax_paise=lgst, net_paise=amount - lfee - lgst)
                )

        elif arche is Archetype.GST_ROUNDING:
            drift = rng.choice([-9, -7, 7, 9]) if is_wild else rng.choice([-2, -1, 1, 2])
            rows.append(row(tax_paise=gst + drift, net_paise=net - drift))

        elif arche is Archetype.T_PLUS_DELAY:
            late = row(settled_on=settled + timedelta(days=rng.randrange(2, 6)))
            if is_wild:
                # late *and* a paise off, so "amounts agree" no longer identifies it
                late = row(
                    settled_on=late.settled_on, net_paise=net - 1, tax_paise=gst + 1
                )
            rows.append(late)

        elif arche is Archetype.PARTIAL_REFUND_NETTED:
            if is_wild:
                # bank also claws back the MDR it had already charged on the
                # refunded slice, so the delta no longer equals the refund
                clawback = mul_bp(refunded, 200)
                rows.append(row(net_paise=net - refunded - clawback))
            else:
                rows.append(row(net_paise=net - refunded))

        elif arche is Archetype.CHARGEBACK_WITHHELD:
            rows.append(row(net_paise=0, fee_paise=0, tax_paise=0))

        elif arche is Archetype.DUPLICATE_BANK_ROW:
            u = _utr(rng)
            rows.append(row(utr=u))
            # a re-issued credit carries a fresh UTR, so UTR equality misses it
            rows.append(row(row_id=rid + "_dup", utr=_utr(rng) if is_wild else u))

        elif arche is Archetype.SPLIT_SETTLEMENT:
            if is_wild:
                parts = [amount // 3, amount // 3, amount - 2 * (amount // 3)]
                for k, g in enumerate(parts):
                    f, gt = fee_for(g, method)
                    rows.append(
                        row(
                            row_id=f"{rid}_{k}",
                            settled_on=settled + timedelta(days=k),
                            gross_paise=g,
                            fee_paise=f,
                            tax_paise=gt,
                            net_paise=g - f - gt,
                        )
                    )
                continue
            half_gross = amount // 2
            rest_gross = amount - half_gross
            f1, g1 = fee_for(half_gross, method)
            f2, g2 = fee_for(rest_gross, method)
            rows.append(
                row(
                    row_id=rid + "_a",
                    gross_paise=half_gross,
                    fee_paise=f1,
                    tax_paise=g1,
                    net_paise=half_gross - f1 - g1,
                )
            )
            rows.append(
                row(
                    row_id=rid + "_b",
                    settled_on=settled + timedelta(days=1),
                    gross_paise=rest_gross,
                    fee_paise=f2,
                    tax_paise=g2,
                    net_paise=rest_gross - f2 - g2,
                )
            )

        elif arche is Archetype.FX_ROUNDING:
            drift = rng.choice([-9, -6, -4, 4, 6, 9])
            rows.append(row(net_paise=net + drift))

    rng.shuffle(rows)
    return Dataset(payments=payments, rows=rows, truth=truth, wild=wild)


def expected_net_for(pay: Payment) -> int:
    """What the ledger expects the bank to pay out for this capture."""
    return expected_net(pay.amount_paise, pay.method)
