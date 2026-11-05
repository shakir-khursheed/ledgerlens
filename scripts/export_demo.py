"""Run both arms over one labelled batch and export the result as JSON.

This is what backs the live break explorer. Nothing in the published page is
simulated: every decision, tool call and metric in the JSON came out of an
actual run of this code.

    python scripts/export_demo.py --sample 40 --out docs/data.json
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledgerlens.evaluate import run
from ledgerlens.generate import expected_net_for, generate
from ledgerlens.matcher import match
from ledgerlens.models import Archetype
from ledgerlens.policy import decide
from ledgerlens.tools import Workspace

SEED = 7
N = 600
WILD = 0.4


def metrics_grid() -> list[dict]:
    """Rules-arm metrics across seeds and noise levels, for the UI's switcher."""
    out = []
    for seed in (7, 11, 23):
        for wild in (0.0, 0.2, 0.4, 0.8):
            rep = run(generate(N, seed=seed, wild_rate=wild), decide, name="policy")
            j = rep.to_json()
            j["seed"] = seed
            j["wild_rate"] = wild
            out.append(j)
    return out


def pick_sample(breaks, truth, wild_ids, rules_by_id, n: int) -> list:
    """Stratify: mostly held-out shapes (where rules fail), some clean wins.

    A sample of only the failures would flatter the model; a random sample
    would mostly show breaks the rules already close. Both are shown.
    """
    wild_wrong, wild_right, standard = [], [], []
    for b in breaks:
        pid = b.payment_ids[0]
        correct = rules_by_id[b.break_id].category == truth[pid]
        if pid in wild_ids:
            (wild_right if correct else wild_wrong).append(b)
        else:
            standard.append(b)
    take = wild_wrong[: int(n * 0.6)]
    take += wild_right[: int(n * 0.15)]
    take += standard[: n - len(take)]
    return take[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40, help="breaks to send to Claude")
    ap.add_argument("--out", default="docs/data.json")
    ap.add_argument("--model", default="claude-sonnet-4-5")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    ds = generate(N, seed=SEED, wild_rate=WILD)
    m = match(ds.payments, ds.rows)
    ws = Workspace(ds.payments, ds.rows)
    pay_by_id = {p.payment_id: p for p in ds.payments}
    rows_by_payment: dict[str, list] = {}
    for r in ds.rows:
        for pid in r.payment_ids:
            rows_by_payment.setdefault(pid, []).append(r)

    rules_rep = run(ds, decide, name="policy (rules only)")
    rules_by_id = {d.break_id: d for d in rules_rep.decisions}

    sample = [] if args.no_llm else pick_sample(
        m.breaks, ds.truth, ds.wild, rules_by_id, args.sample
    )
    sample_ids = {b.break_id for b in sample}

    claude_by_id: dict[str, dict] = {}
    if sample:
        from ledgerlens.agent import ClaudeReasoner

        llm = ClaudeReasoner(model=args.model)
        lock = threading.Lock()
        done = [0]

        def judge(brk):
            """One break, one fresh Workspace so the tool log is this break's."""
            own = Workspace(ds.payments, ds.rows)
            t0 = time.perf_counter()
            try:
                d = llm(brk, own)
                rec = {**d.to_json(),
                       "tools_called": [c[0] for c in own.calls],
                       "seconds": round(time.perf_counter() - t0, 1)}
            except Exception as exc:  # a failed call is data, not a crash
                rec = {"error": f"{type(exc).__name__}: {exc}"}
            with lock:
                claude_by_id[brk.break_id] = rec
                done[0] += 1
                print(f"[{done[0]}/{len(sample)}] {brk.break_id} "
                      f"{rec.get('category', 'ERROR')} ({rec.get('seconds', 0)}s)", flush=True)
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out + ".partial").write_text(json.dumps(claude_by_id))

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(judge, sample))

    breaks_json = []
    for b in m.breaks:
        pid = b.payment_ids[0]
        pay = pay_by_id[pid]
        rd = rules_by_id[b.break_id]
        breaks_json.append({
            "break_id": b.break_id,
            "payment_id": pid,
            "truth": ds.truth[pid].value,
            "wild": pid in ds.wild,
            "note": b.note,
            "expected_net_paise": b.expected_net_paise,
            "observed_net_paise": b.observed_net_paise,
            "delta_paise": b.delta_paise,
            "payment": {
                "amount_paise": pay.amount_paise,
                "method": pay.method.value,
                "status": pay.status.value,
                "refunded_paise": pay.refunded_paise,
                "captured_on": pay.captured_on.isoformat(),
            },
            "rows": [
                {"row_id": r.row_id, "utr": r.utr, "settled_on": r.settled_on.isoformat(),
                 "gross_paise": r.gross_paise, "fee_paise": r.fee_paise,
                 "tax_paise": r.tax_paise, "net_paise": r.net_paise}
                for r in sorted(rows_by_payment.get(pid, []), key=lambda r: r.row_id)
            ],
            "rules": rd.to_json(),
            "claude": claude_by_id.get(b.break_id),
            "in_sample": b.break_id in sample_ids,
        })

    # Claude's score on the breaks it actually saw, and the rules' score on the
    # same subset -- the only honest comparison.
    judged = [b for b in breaks_json if b.get("claude") and "category" in b["claude"]]
    def acc(key: str, subset) -> float:
        return round(sum(b[key]["category"] == b["truth"] for b in subset) / len(subset), 4) if subset else 0.0
    wild_judged = [b for b in judged if b["wild"]]

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "model": args.model,
        "batch": {"n": N, "seed": SEED, "wild_rate": WILD,
                  "auto_matched": len(m.auto_matched), "breaks": len(m.breaks)},
        "rules_report": rules_rep.to_json(),
        "grid": metrics_grid(),
        "head_to_head": {
            "sample_size": len(judged),
            "wild_in_sample": len(wild_judged),
            "rules_accuracy": acc("rules", judged),
            "claude_accuracy": acc("claude", judged),
            "rules_accuracy_wild": acc("rules", wild_judged),
            "claude_accuracy_wild": acc("claude", wild_judged),
            "claude_unsafe": [
                b["break_id"] for b in judged
                if b["truth"] in ("missing_in_bank", "chargeback_withheld")
                and not (b["claude"].get("needs_human") and b["claude"].get("proposed_adjustment_paise") == 0)
            ],
            "avg_seconds": round(sum(b["claude"].get("seconds", 0) for b in judged) / len(judged), 1) if judged else 0,
            "avg_tool_calls": round(sum(b["claude"].get("tool_calls", 0) for b in judged) / len(judged), 1) if judged else 0,
        },
        "breaks": breaks_json,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload))
    print("wrote", args.out, Path(args.out).stat().st_size, "bytes")
    print(json.dumps(payload["head_to_head"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
