"""The LLM arm: a Claude Agent SDK loop over the same read-only tools.

Why a model at all, when a rule table already scores 100% in-distribution?
Because the rule table scores ~27% on break shapes nobody wrote a rule for,
and every new bank, rate card or product adds shapes. The model's job is not
to be cleverer than the rules on known breaks -- it is to stop the long tail
from landing in a human queue.

The loop is deliberately boring:

    break -> system prompt (policy, not personality)
          -> model calls read-only tools until it has evidence
          -> model must finish by calling submit_decision
          -> typed Decision, or we treat the break as unresolved

Everything the model can reach is read-only. ``submit_decision`` records an
intent; nothing in this process moves money.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from .models import Archetype, Break, Decision
from .tools import Workspace

SYSTEM_PROMPT = """\
You are a settlement reconciliation analyst at a payments company. You are given
one reconciliation break: a payment captured in our ledger whose bank payout does
not line up with what we expected.

Your job is to classify the break into exactly one category and explain it in one
or two sentences an ops analyst can act on. Investigate with the tools before you
decide -- a classification with no tool evidence behind it is worthless.

Categories:
- fee_schedule_delta: the bank priced the payout on a different MDR than ours.
- gst_rounding: paise-level drift from rounding tax on the fee.
- t_plus_delay: the full amount arrived, just outside the expected window.
- partial_refund_netted: a booked refund (possibly with its fee clawback) was
  netted off the payout.
- chargeback_withheld: funds held against an open dispute.
- duplicate_bank_row: the same credit appears more than once in the bank file,
  possibly re-issued under a different UTR.
- split_settlement: one capture paid out across several batches.
- fx_rounding: conversion rounding on a non-INR capture.
- missing_in_bank: captured, never settled.
- unknown: nothing you can evidence. Use this rather than guessing.

Hard rules, in order of priority:
1. If the truth is that money is genuinely missing (missing_in_bank) or held
   (chargeback_withheld), set needs_human true and proposed_adjustment_paise 0.
   Never propose an adjustment that would paper over missing money.
2. Amounts are integer paise. Never reason in rupees or floats.
3. If your evidence does not pin one category, answer unknown with
   needs_human true. An honest escalation costs minutes; a confident wrong
   write-off costs money and trust.
4. Finish by calling submit_decision exactly once.
"""

_CATEGORIES = [a.value for a in Archetype]


def _build_server(ws: Workspace, sink: dict[str, Any]):
    from claude_agent_sdk import create_sdk_mcp_server, tool

    def ok(payload: Any) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": json.dumps(payload, default=str)}]}

    @tool("get_payment", "Ledger view of a payment: amount, method, status, refunds, FX.",
          {"payment_id": str})
    async def _get_payment(args):
        return ok(ws.get_payment(args["payment_id"]))

    @tool("get_settlement_rows", "Every bank settlement row referencing this payment.",
          {"payment_id": str})
    async def _get_rows(args):
        return ok(ws.get_settlement_rows(args["payment_id"]))

    @tool("get_expected_window", "Date range in which this capture should have settled.",
          {"payment_id": str})
    async def _get_window(args):
        return ok(ws.get_expected_window(args["payment_id"]))

    @tool("get_fee_schedule", "Current and legacy MDR, GST rate and settlement lag for a method.",
          {"method": str})
    async def _get_fees(args):
        return ok(ws.get_fee_schedule(args["method"]))

    @tool("compute_expected_net", "Recompute the expected payout under either rate card.",
          {"amount_paise": int, "method": str, "legacy": bool})
    async def _expected(args):
        return ok(ws.compute_expected_net(args["amount_paise"], args["method"],
                                          bool(args.get("legacy", False))))

    @tool("get_refunds", "Refunds booked against the payment.", {"payment_id": str})
    async def _refunds(args):
        return ok(ws.get_refunds(args["payment_id"]))

    @tool("get_disputes", "Open chargebacks against the payment.", {"payment_id": str})
    async def _disputes(args):
        return ok(ws.get_disputes(args["payment_id"]))

    @tool("find_rows_by_utr", "All settlement rows sharing a bank UTR.", {"utr": str})
    async def _by_utr(args):
        return ok(ws.find_rows_by_utr(args["utr"]))

    @tool(
        "submit_decision",
        "Record the final classification. Call exactly once, last.",
        {
            "category": str,
            "explanation": str,
            "confidence": float,
            "proposed_adjustment_paise": int,
            "needs_human": bool,
            "evidence": str,
        },
    )
    async def _submit(args):
        sink.update(args)
        return ok({"recorded": True})

    return create_sdk_mcp_server(
        name="ledger",
        tools=[_get_payment, _get_rows, _get_window, _get_fees, _expected,
               _refunds, _disputes, _by_utr, _submit],
    )


def _prompt(brk: Break) -> str:
    return (
        f"Break {brk.break_id}\n"
        f"payment_ids: {list(brk.payment_ids)}\n"
        f"settlement_row_ids: {list(brk.settlement_row_ids)}\n"
        f"expected_net_paise: {brk.expected_net_paise}\n"
        f"observed_net_paise: {brk.observed_net_paise}\n"
        f"delta_paise: {brk.delta_paise}\n"
        f"matcher_note: {brk.note}\n\n"
        "Investigate and classify this break."
    )


class ClaudeReasoner:
    """Reasoner backed by Claude via the Agent SDK. Requires credentials."""

    def __init__(self, model: str = "claude-sonnet-4-5", max_turns: int = 12) -> None:
        self.model = model
        self.max_turns = max_turns

    def __call__(self, brk: Break, ws: Workspace) -> Decision:
        return asyncio.run(self.acall(brk, ws))

    async def acall(self, brk: Break, ws: Workspace) -> Decision:
        from claude_agent_sdk import ClaudeAgentOptions, query

        before = ws.call_count
        sink: dict[str, Any] = {}
        server = _build_server(ws, sink)
        names = [
            "get_payment", "get_settlement_rows", "get_expected_window",
            "get_fee_schedule", "compute_expected_net", "get_refunds",
            "get_disputes", "find_rows_by_utr", "submit_decision",
        ]
        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={"ledger": server},
            allowed_tools=[f"mcp__ledger__{n}" for n in names],
            disallowed_tools=["Bash", "Write", "Edit", "Read", "WebFetch", "WebSearch"],
            permission_mode="bypassPermissions",
            max_turns=self.max_turns,
            model=self.model,
        )
        async for _ in query(prompt=_prompt(brk), options=options):
            pass
        return self._to_decision(brk, sink, ws.call_count - before)

    @staticmethod
    def _to_decision(brk: Break, sink: dict[str, Any], tool_calls: int) -> Decision:
        raw = str(sink.get("category", "unknown")).strip().lower()
        category = Archetype(raw) if raw in _CATEGORIES else Archetype.UNKNOWN
        needs_human = bool(sink.get("needs_human", True))
        adjustment = int(sink.get("proposed_adjustment_paise", 0) or 0)
        # Belt and braces: the system prompt asks for this, the harness enforces it.
        if category in (Archetype.MISSING_IN_BANK, Archetype.CHARGEBACK_WITHHELD):
            needs_human, adjustment = True, 0
        if not sink:
            return Decision(
                break_id=brk.break_id,
                category=Archetype.UNKNOWN,
                explanation="agent ended without calling submit_decision",
                confidence=0.0,
                needs_human=True,
                tool_calls=tool_calls,
            )
        ev = sink.get("evidence") or ""
        return Decision(
            break_id=brk.break_id,
            category=category,
            explanation=str(sink.get("explanation", "")),
            confidence=float(sink.get("confidence", 0.0) or 0.0),
            proposed_adjustment_paise=adjustment,
            needs_human=needs_human,
            evidence=[e for e in (ev.split("\n") if isinstance(ev, str) else list(ev)) if e],
            tool_calls=tool_calls,
        )


class HybridReasoner:
    """What actually ships: rules first, model only on the residual.

    The policy answers in microseconds for nothing. If it is confident, that is
    the answer. The model is only paid for when the rules shrug -- which on the
    wild split is where all the value is, and in steady state is a small slice
    of a batch.
    """

    def __init__(self, llm: Any, *, threshold: float = 0.85) -> None:
        self.llm = llm
        self.threshold = threshold
        self.escalated = 0

    def __call__(self, brk: Break, ws: Workspace) -> Decision:
        from .policy import decide

        d = decide(brk, ws)
        if d.category is not Archetype.UNKNOWN and d.confidence >= self.threshold:
            return d
        self.escalated += 1
        return self.llm(brk, ws)
