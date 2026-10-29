# Proposed acquisition policies: AEDAG and BBAI

This directory describes two proposed policies for crop-based visual question
answering. It is a design note, not an implementation or a claim of benchmark
improvement. The existing FOVEA baselines and the experiment protocol in the
repository remain the reference point for future comparisons.

## Goals

Both policies should spend visual-observation calls where they can change the
answer, keep all observations available to the final reader, and stop when the
remaining search is unlikely to justify its cost. Their shared infrastructure
should make candidate identity, observation history, budgets, and stop reasons
consistent across policies.

The implementation should use ordinary composable Python components and
configuration. Avoid monkeypatching. Keep protocol validation, candidate
management, policy decisions, and accounting as explicit interfaces so the
components can be reused by other acquisition strategies.

## AEDAG: adaptive evidence DAG

AEDAG represents possible visual facts as nodes in a directed acyclic graph.
An edge means that a later observation is useful only after a prerequisite
fact has been resolved. Independent facts can be explored without artificial
dependencies. A node identifies a claim, image region, prerequisites, possible
observation outcomes, and likelihoods for the answer options.

### Proposed policy

1. Maintain a posterior over answer options from the initial answer belief and
   all acquired evidence.
2. Generate concrete, executable observation candidates. Each candidate must
   identify a claim and valid crop; conditional follow-up branches must point to
   declared child nodes rather than free-form future actions.
3. Score candidates using expected answer-risk reduction and information gain,
   adjusted for observation cost and remaining budget. Keep the weights
   configurable and record their values with each run.
4. Execute only dependency-ready nodes. Reuse an already acquired view when it
   covers the needed region and claim; do not invalidate a whole plan because
   one independent candidate is malformed or infeasible.
5. Update beliefs from the actual observation outcome, then replan. Preserve
   resolved facts and node identities across replans, and record why candidates
   were rejected or the policy stopped.

### Questions to validate

- Does the policy generate valid, observable candidates often enough to avoid
  ending at the initial answer without looking at a crop?
- Does its utility score predict useful observations, or does it reject
  reasonable candidates because its expected gain is too conservative?
- Do selected dependency chains actually execute, and do their outcomes enable
  their declared follow-up actions?
- Does posterior change track evidence that helps distinguish answer options?

## BBAI: Bayesian best-arm identification

BBAI treats candidate image regions as arms. A probe estimates whether a view
contains evidence that can distinguish the remaining answer hypotheses. The
policy should explore spatially distinct regions, then allocate repeated probes
between the strongest candidate and its challenger. The total probe budget is
per question, not per candidate.

### Proposed policy

1. Ask for a small initial portfolio of complementary crop locations.
2. Track candidate identity in a shared registry. Compare normalized claims and
   crop geometry so paraphrases of the same fact or near-identical regions are
   rejected before consuming a probe. Keep a clear rejection reason and allow
   genuinely distinct locations.
3. Expand the portfolio gradually as evidence accumulates, subject to crop and
   request limits. An empty candidate list is a valid signal that expansion is
   exhausted; it must not trigger retries as if it were malformed output.
4. Use a best-arm identification rule, such as top-two Thompson sampling, to
   allocate the remaining question-wide probe budget between the current leader
   and a plausible challenger. Preserve every acquired view and observation for
   the final answer.
5. Stop on a declared condition: exhausted global probe budget, no distinct
   candidates, exhausted request or cost budget, or insufficient expected
   value. Record the condition explicitly so long searches are visible.

### Questions to validate

- Does spatial deduplication reduce rejected proposals without excluding
  useful neighboring views?
- Does progressive expansion find complementary evidence beyond scale variants
  of one location?
- Does the allocation rule use fewer probes than uniform allocation at similar
  answer quality, under a fixed question-wide budget?
- Does the stop rule prevent repeated search on a question after new evidence
  stops changing the decision?

## Shared contracts and diagnostics

The policies should share reusable components for:

- candidate identity and duplicate detection;
- crop validation and view reuse;
- answer-belief updates and observation outcomes, including unresolved results;
- question-wide request, crop, probe, and cost limits;
- structured stop reasons and candidate rejection logs.

For each proposed or executed observation, record the candidate identity,
selection score, belief immediately before the observation, actual outcome,
belief after the update, cost, and any rejection or stop reason. Offline
diagnostics should report observation coverage, duplicate/rejection counts,
unresolved outcomes, posterior entropy and distribution changes, and cost per
question. Posterior movement is an internal diagnostic; by itself it does not
show that the new belief is better calibrated or closer to the correct answer.

## Evaluation plan

Before a paid comparison:

1. Validate protocol behavior offline, including malformed candidates,
   independent and dependent DAG nodes, unresolved observations, duplicate
   regions, empty expansion responses, and global budget exhaustion.
2. Freeze code, configuration, model/provider settings, sample IDs, and scoring
   rules. Do not tune on the evaluation panel.
3. First run a small diagnostic focused on candidate validity, observation
   execution, posterior updates, duplicate rejection, stop reasons, and costs.
4. If those diagnostics are sound, compare against Direct VLM, ReAct-crop, and
   the native FOVEA policies on identical inputs. Report paired outcomes,
   uncertainty intervals, failures, requests, and costs. Use a separate
   untouched panel for a final claim after any tuning.

No accuracy gain, cost saving, or generalization benefit is claimed here. Those
are empirical questions for a protocol-matched evaluation.
