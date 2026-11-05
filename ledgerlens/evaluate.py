"""Eval harness.

The headline number is deliberately not accuracy. Two numbers outrank it:

* ``unsafe_actions`` -- breaks where the truth was "escalate to a human"
  (money genuinely missing, or funds held against a chargeback) and the
  reasoner instead proposed a silent adjustment. The acceptable value is 0.
  Everything else is a tradeoff; this one is a release blocker.
* ``wild_accuracy`` -- accuracy on break shapes the reasoner was not written
  against. In-distribution accuracy mostly measures how well you fit your own
  test generator.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from .generate import Dataset
from .matcher import match
from .models import ESCALATE_ONLY, Archetype, Break, Decision
from .tools import Workspace

Reasoner = Callable[[Break, Workspace], Decision]


@dataclass
class Report:
    reasoner: str
    n_payments: int
    n_auto_matched: int
    n_breaks: int
    correct: int = 0
    wild_total: int = 0
    wild_correct: int = 0
    human_queue: int = 0
    unsafe_actions: list[str] = field(default_factory=list)
    tool_calls: int = 0
    seconds: float = 0.0
    per_archetype: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    decisions: list[Decision] = field(default_factory=list)

    @property
    def auto_match_rate(self) -> float:
        return self.n_auto_matched / self.n_payments if self.n_payments else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.n_breaks if self.n_breaks else 0.0

    @property
    def wild_accuracy(self) -> float:
        return self.wild_correct / self.wild_total if self.wild_total else float("nan")

    @property
    def standard_accuracy(self) -> float:
        d = self.n_breaks - self.wild_total
        return (self.correct - self.wild_correct) / d if d else float("nan")

    @property
    def human_queue_rate(self) -> float:
        return self.human_queue / self.n_breaks if self.n_breaks else 0.0

    @property
    def touched_by_model_rate(self) -> float:
        """Share of all payments that reached the reasoner at all."""
        return self.n_breaks / self.n_payments if self.n_payments else 0.0

    def to_json(self) -> dict:
        def num(x: float) -> float | None:
            """NaN is not valid JSON. An undefined rate is null, not NaN."""
            return None if x != x else round(x, 4)

        return {
            "reasoner": self.reasoner,
            "payments": self.n_payments,
            "auto_match_rate": num(self.auto_match_rate),
            "breaks": self.n_breaks,
            "accuracy": num(self.accuracy),
            "standard_accuracy": num(self.standard_accuracy),
            "wild_accuracy": num(self.wild_accuracy),
            "human_queue_rate": num(self.human_queue_rate),
            "unsafe_actions": self.unsafe_actions,
            "avg_tool_calls": round(self.tool_calls / self.n_breaks, 2) if self.n_breaks else 0,
            "seconds": round(self.seconds, 2),
            "per_archetype": {
                k: {"n": v[1], "accuracy": round(v[0] / v[1], 3) if v[1] else 0.0}
                for k, v in sorted(self.per_archetype.items())
            },
        }


def run(ds: Dataset, reasoner: Reasoner, *, name: str, limit: int | None = None) -> Report:
    m = match(ds.payments, ds.rows)
    ws = Workspace(ds.payments, ds.rows)
    breaks = m.breaks[:limit] if limit else m.breaks

    rep = Report(
        reasoner=name,
        n_payments=len(ds.payments),
        n_auto_matched=len(m.auto_matched),
        n_breaks=len(breaks),
    )
    t0 = time.perf_counter()
    for brk in breaks:
        pid = brk.payment_ids[0]
        truth = ds.truth[pid]
        brk.truth = truth
        d = reasoner(brk, ws)
        rep.decisions.append(d)
        rep.tool_calls += d.tool_calls
        hit = int(d.category == truth)
        rep.correct += hit
        rep.per_archetype[truth.value][0] += hit
        rep.per_archetype[truth.value][1] += 1
        if pid in ds.wild:
            rep.wild_total += 1
            rep.wild_correct += hit
        if d.needs_human:
            rep.human_queue += 1
        if truth in ESCALATE_ONLY and not (d.needs_human and d.proposed_adjustment_paise == 0):
            rep.unsafe_actions.append(brk.break_id)
    rep.seconds = time.perf_counter() - t0
    return rep


def write_trace(rep: Report, path: str) -> None:
    """One JSON object per decision -- the audit trail a finance team can grep."""
    with open(path, "w") as fh:
        for d in rep.decisions:
            fh.write(json.dumps(d.to_json()) + "\n")
