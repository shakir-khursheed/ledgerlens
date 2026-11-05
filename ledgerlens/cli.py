"""Command line entrypoint: python -m ledgerlens ..."""
from __future__ import annotations

import argparse
import json
import sys

from .evaluate import Report, run, write_trace
from .generate import generate
from .policy import decide


def _reasoner(name: str, model: str):
    if name == "policy":
        return decide, "policy (rules only)"
    if name == "claude":
        from .agent import ClaudeReasoner

        return ClaudeReasoner(model=model), f"claude ({model})"
    if name == "hybrid":
        from .agent import ClaudeReasoner, HybridReasoner

        return HybridReasoner(ClaudeReasoner(model=model)), f"hybrid (rules -> {model})"
    raise SystemExit(f"unknown reasoner: {name}")


def _table(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for n, r in enumerate(rows):
        out.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip())
        if n == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def _pct(x: float | None) -> str:
    """An undefined rate prints as a dash, not as a crash or '0.0%'."""
    return "    —" if x is None else f"{x:.1%}"


def _summary(rep: Report) -> str:
    j = rep.to_json()
    lines = [
        f"reasoner              {j['reasoner']}",
        f"payments in batch     {j['payments']}",
        f"auto-matched by rules {_pct(j['auto_match_rate'])}  (never reaches a model)",
        f"breaks investigated   {j['breaks']}",
        "",
        f"resolution accuracy   {_pct(j['accuracy'])}",
        f"  in-distribution     {_pct(j['standard_accuracy'])}",
        f"  held-out 'wild'     {_pct(j['wild_accuracy'])}",
        f"routed to a human     {_pct(j['human_queue_rate'])}",
        f"unsafe actions        {len(j['unsafe_actions'])}   <- must be 0",
        f"avg tool calls/break  {j['avg_tool_calls']}",
        f"wall clock            {j['seconds']}s",
        "",
        "per break family:",
    ]
    rows = [("family", "n", "accuracy")]
    for k, v in j["per_archetype"].items():
        rows.append((k, str(v["n"]), f"{v['accuracy']:.0%}"))
    lines.append(_table(rows))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("ledgerlens", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("eval", help="score a reasoner on a labelled batch")
    ev.add_argument("--n", type=int, default=600)
    ev.add_argument("--seed", type=int, default=7)
    ev.add_argument("--wild", type=float, default=0.4, help="share of unseen break shapes")
    ev.add_argument("--reasoner", default="policy", choices=["policy", "claude", "hybrid"])
    ev.add_argument("--model", default="claude-sonnet-4-5")
    ev.add_argument("--limit", type=int, default=None, help="cap breaks (use with --reasoner claude)")
    ev.add_argument("--trace", default=None, help="write per-decision JSONL audit trail")
    ev.add_argument("--json", action="store_true")

    dm = sub.add_parser("demo", help="rules-only, in-distribution vs held-out")
    dm.add_argument("--n", type=int, default=600)
    dm.add_argument("--seed", type=int, default=7)

    args = ap.parse_args(argv)

    if args.cmd == "demo":
        for wild, label in ((0.0, "in-distribution only"), (0.4, "40% unseen shapes")):
            ds = generate(args.n, seed=args.seed, wild_rate=wild)
            rep = run(ds, decide, name=f"policy / {label}")
            print(_summary(rep))
            print("\n" + "=" * 62 + "\n")
        print(
            "Rules are perfect on shapes they were written for and fall apart on\n"
            "shapes they were not -- safely, into the human queue. That gap is the\n"
            "entire argument for the model arm: run `eval --reasoner hybrid`."
        )
        return 0

    ds = generate(args.n, seed=args.seed, wild_rate=args.wild)
    fn, label = _reasoner(args.reasoner, args.model)
    rep = run(ds, fn, name=label, limit=args.limit)
    if args.trace:
        write_trace(rep, args.trace)
    print(json.dumps(rep.to_json(), indent=2) if args.json else _summary(rep))
    return 1 if rep.unsafe_actions else 0


if __name__ == "__main__":
    sys.exit(main())
