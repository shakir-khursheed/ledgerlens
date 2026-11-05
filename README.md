# LedgerLens

An agent that explains **settlement reconciliation breaks** — and an eval harness
whose job is to tell you when not to trust it.

Every payments business runs the same unglamorous loop at the end of every day:
the bank sends a settlement file, the ledger says what it expected, and a few
percent of rows disagree. Someone in ops opens each one, pulls the payment, the
rate card, the refunds and the disputes, and writes down *why*. It is slow,
repetitive, and the one place where a careless auto-fix quietly loses money.

LedgerLens does the pulling and the writing down. It never moves money.

```
┌────────────────┐   ~50% closed, zero cost
│ settlement file│──────────────────────────────▶ matched
│   + ledger     │
└───────┬────────┘
        │ residual breaks only
        ▼
┌────────────────┐     read-only tools      ┌──────────────────────┐
│  rules engine  │◀────────────────────────▶│ get_payment          │
│  (confident?)  │                          │ get_settlement_rows  │
└───┬────────┬───┘                          │ get_fee_schedule     │
    │ yes    │ no / low confidence          │ compute_expected_net │
    │        ▼                              │ get_refunds          │
    │  ┌───────────────┐                    │ get_disputes         │
    │  │ Claude agent  │◀──────────────────▶│ find_rows_by_utr     │
    │  │ (Agent SDK)   │                    │ get_expected_window  │
    │  └───────┬───────┘                    └──────────────────────┘
    │          │
    ▼          ▼
  typed Decision: category · explanation · confidence
                 · proposed_adjustment_paise · needs_human · evidence
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      ops queue (auto-cleared)   human queue (escalations)
```

## The headline numbers

600-payment batch, labelled by break family. Rules-only arm, no credentials needed:

| | in-distribution | 40% unseen break shapes |
|---|---|---|
| auto-matched before any reasoning | 51.0% | 48.3% |
| resolution accuracy on the residual | **100%** | **73.2%** |
| …on break shapes it was written for | 100% | 100% |
| …on shapes it was **not** | — | **42.0%** |
| routed to a human | 7.5% | 33.6% |
| **adjustments proposed over missing or held money** | **0** | **0** |

Reproduce: `make demo`. Full per-family table and a JSONL audit trail: `make eval`.

That 100% in the first column is not a brag — it is the warning. A hand-written
rule table scores 100% against a test set built by the same person who wrote the
rules. The number that means something is **42%**, on break shapes the rules were
never designed for: a bank-specific rate card we don't hold, a refund clawed back
net of its own MDR, a credit re-issued under a fresh UTR, a capture split three
ways. Those shapes arrive every time a new bank, product or rate revision ships.

So the honest reading is: **rules close half the file for free and handle every
family you've already seen; the long tail is what you'd pay a model for.** That
is exactly how `HybridReasoner` is wired — rules first, Claude only on what the
rules shrug at (`--reasoner hybrid`), so cost tracks the tail and not the volume.

## What it will never do

The eval's top-line metric is not accuracy, it's `unsafe_actions`: breaks whose
truth was *money is genuinely missing* or *funds are held against a chargeback*,
where the reasoner proposed an adjustment anyway instead of escalating. **The
acceptable value is zero**, it is asserted in CI across seeds and perturbation
rates, and a non-zero value fails the build.

Three layers enforce it, because one is a promise and three is a design:

1. Every tool the agent can reach is **read-only**. There is no write path.
   The strongest thing it can emit is a *proposal*.
2. The system prompt ranks "escalate honestly" above "classify confidently".
3. The harness **clamps** the output regardless of what the model said — a
   decision on missing or held money is forced to zero adjustment and
   `needs_human`. `test_adjustment_over_missing_money_is_clamped_away` is the
   test that matters most in this repo.

Degrading into a human queue is a fine failure. Quietly adjusting a ledger is not.

## Design decisions worth arguing about

**Rules before the model, always.** Roughly half of a settlement file matches
exactly. Sending those rows to an LLM would be slower, costlier and strictly
worse than `==`. The model earns its keep only on the residual.

**The rules arm is the control, not a fallback bolted on later.** Without it you
cannot answer "does the model actually add anything here?" with a number. With
it, the answer is a 42% → *measure it* gap on the wild split, per family. It is
also what runs when the model is down or a batch blows its cost budget.

**Integer paise everywhere, half-up rounding.** No floats touch money anywhere in
this codebase. Python's default banker's rounding on GST is itself a source of
phantom breaks — `test_half_up_not_bankers` pins it.

**Typed decisions, not prose.** Free text can't be counted, routed or regression-
tested. Every output is a `Decision` with a category from a closed set, a
confidence, an explicit adjustment in paise and the tool evidence behind it.
`data/trace.jsonl` is one JSON object per decision — the audit trail a finance
team can grep, which is the thing that actually gets an agent approved.

**Held-out "wild" shapes in the eval.** The cheapest way to fool yourself is to
grade a system on its own generator. `--wild` perturbs breaks into shapes no rule
anticipated, and the report splits in-distribution from held-out accuracy so the
gap is impossible to miss.

## Running it

```bash
pip install -e ".[dev]"
make demo      # rules arm, in-distribution vs held-out. no credentials needed
make test      # 30 tests, incl. the safety invariant across seeds
make eval      # full report + data/trace.jsonl audit trail
```

The Claude arm needs `pip install -e ".[llm]"` and working Agent SDK
credentials:

```bash
python -m ledgerlens eval --wild 0.4 --reasoner hybrid --limit 40 --trace data/trace.jsonl
```

`--limit` caps how many breaks reach the model, because an eval that quietly
spends money is its own kind of bug.

**On the data.** Real settlement files can't go in a public repo, so the batch is
generated — but the *break families* and their rough frequencies are the ones ops
analysts actually see, which is what makes a score on it mean anything. Swap
`generate()` for a real file loader and nothing downstream changes.

## If this were running for real

What I'd instrument from day one, because "it works" is not adoption:

- **% of the file closed without a human**, split rules vs model. The number ops
  actually feels.
- **Minutes per break, before and after.** Hand-reconciling one break is a
  several-minute job; the measurable claim is the delta, not a guess at it.
- **Recoverable money surfaced** — `fee_schedule_delta` is a bank mispricing you
  can bill back. It shows up here as a signed adjustment precisely so it can be
  totalled instead of absorbed.
- **Human-queue precision.** If analysts keep overturning the agent on one
  family, that family needs a rule, not a better prompt.
- **Cost per batch**, tracked against the share of breaks that reached the model.

## Layout

```
ledgerlens/
  models.py       typed domain + the escalate-only set
  feeschedule.py  integer-paise money math, MDR/GST/T+n
  generate.py     labelled batches, incl. held-out "wild" shapes
  matcher.py      deterministic pass (closes ~50%, zero false matches)
  tools.py        read-only tool surface + call log
  policy.py       rules arm / control
  agent.py        Claude Agent SDK arm + hybrid router + safety clamp
  evaluate.py     metrics, per-family breakdown, JSONL trace
  cli.py          python -m ledgerlens demo | eval
tests/            30 tests; safety invariant parametrised over seeds and noise
```
