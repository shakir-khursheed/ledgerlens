# Decision log

Short record of the calls that shaped this, and what would change my mind.

## 1. Deterministic matcher before any model
**Call.** Exact-match rows close in the rules pass; only the residual reaches a
reasoner.
**Why.** ~50% of a settlement file matches on the nose. An LLM there is slower,
costlier and less reliable than `==`.
**Changes my mind.** If the exact-match rate fell below ~20% on real files, the
split stops paying for itself and the matcher becomes a fuzzy-candidate
generator instead.

## 2. The rules arm is a control, not a fallback
**Call.** Keep a full hand-written policy and score it in the same harness.
**Why.** Without a control you cannot answer "is the model earning its cost?".
With it the answer is a per-family number on the held-out split.
**Changes my mind.** Nothing, really — the day the model beats rules everywhere,
the control is what proves it.

## 3. `unsafe_actions` outranks accuracy
**Call.** Breaks whose truth is "money missing" or "funds held" must leave as a
zero-adjustment escalation. Asserted in CI; non-zero fails the build.
**Why.** In payments the cost function is wildly asymmetric. An unnecessary
escalation costs an analyst two minutes. A confident wrong write-off costs money
and, worse, the finance team's trust — after which the agent gets switched off
and no amount of accuracy brings it back.

## 4. Clamp the model's output in the harness
**Call.** The escalation rule is enforced in code after the model answers, not
only asked for in the prompt.
**Why.** A prompt is a request. For the one failure mode that is unacceptable,
I want a mechanism. Prompt and clamp agree today; the clamp is what holds when a
model upgrade quietly changes behaviour.

## 5. Held-out "wild" shapes in the eval
**Call.** `--wild` perturbs breaks into shapes no rule anticipated; the report
splits in-distribution from held-out accuracy.
**Why.** Grading a rule table on a generator written by the same person produces
100% and tells you nothing. The in-distribution/held-out gap is the honest
measure, and it is the entire business case for the model arm.

## 6. Integer paise, half-up
**Call.** No floats on money, anywhere. Explicit half-up rounding.
**Why.** Python's banker's rounding on GST generates phantom breaks — the exact
thing this tool exists to eliminate. Pinned by a test.

## 7. Answered: does the model arm earn its cost?
**Measured.** On 40 breaks scored for both arms, Claude reached 86.7% on
held-out shapes against the rules' 20.0% on the same subset, with zero unsafe
adjustments. The sample is weighted toward rules failures on purpose, since
that is the only region where a model can add anything. Average 5.8 tool calls
and 58s per break, which is why the shipping shape routes rules first.

## Open questions
- The model's weak family is sub-rupee drift: it over-calls GST rounding on FX
  and late-settlement breaks. That is a prompt and tool problem (it cannot see
  the FX rate's own rounding behaviour), not a model-capability one.
- Confidence is hand-assigned in the rules arm. It should be calibrated against
  observed analyst overturn rates once there are any.
- Multi-payment settlement rows (one bank credit covering many captures) are
  modelled in the schema but not yet generated. That is the next break family.
