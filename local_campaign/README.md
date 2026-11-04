# V*191 Graph gate V2 campaign

The former run was stopped and deleted at the user's request. Only its aggregate
confirmed spend ($0.03757421) remains in budget_carryover.json to enforce the
existing cumulative $10 allowance. The new run is
`../runs/vstar191-graphgate-v2-8b-10usd-20261103/`.

All four methods restart on the same 191 verified local images from
`../../fovea_bench/prepared/vstar-full-20261102/`. No previous responses or results
are reused. Four method workers share one SQLite budget ledger and dispatch gate.
The model is Qwen3-VL-8B-Instruct on Parasail, temperature 1, max_tokens 2048.

Direct retains original-image inference. ReAct and Greedy retain their final
reader. Graph now follows this sequence:

1. Plan and search the original image/crops with the evidence-state controller.
2. An independent Graph gate receives the question, original/crop images and
   Graph state only. Its prompt builder does not access the Direct report.
3. The gate returns an option or NO_ANSWER, rationale, recorded evidence citations
   and uncertainties. Citations are validated against actual observations.
4. Persist gate.json before making the final aggregator request.
5. The final aggregator receives Graph evidence, the gate verdict, and Direct's
   independent analysis; persist aggregator.json and both categorical answers.

ReAct/Greedy/Graph have at most ten external crop attempts, including failures.
Direct has zero. Greedy uses 3 candidates × 3 probes. Plans, proposals, format
repairs, the gate and final aggregator are separate computation. Graph preserves
its 24-call controller limit and adds one gate/fusion call outside that count, for
at most 25 logical calls; bounded format retries are separately billed. Results
store both controller_logical_model_calls and actual logical_model_calls.

The runnable source snapshot relocates /workspace/ paths and records original and
runtime hashes. graph_gate.py is also frozen inside that snapshot. Labels are
kept out of inference and only read for reporting. The reporter saves paired
planner/gate versus final-reader accuracy, the four correctness transitions, and
per-case provenance in reader_comparison.json. Graph also reports all produced
gates, including gates whose final aggregator is pending or failed. NO_ANSWER
scores zero; unavailable planner answers are excluded from paired comparisons.

Commands use `/private/tmp/fovea-turn10-venv/bin/python`:

- `campaign.py prepare`: verify images and freeze protocol/source.
- `campaign.py run`: acquire exclusive worker lock and execute/resume.
- `campaign.py status`: update progress and comparison files without API calls.

The run uses a background session and caffeinate. API keys are read at runtime
from the existing restricted temporary file and are not copied into artifacts.
Requests start at a minimum 30-second gap. pacing.py extends the shared gate to
45 seconds after the first 429 and 60 seconds after the third; Retry-After and
exponential shared cooldown remain in force. Shared provider-pool limits can
still reject requests. Confirmed previous + new spend + reservations cannot
exceed $10. Unknown charges stop new dispatch until reconciliation.

The chat heartbeat reports every 30 minutes, including standalone gate accuracy,
paired aggregator accuracy, correct→wrong/wrong→correct counts, method progress,
mean interactions, cumulative/new spend, remaining time and HTTP429 changes.

## Fair dispatch and resume

The method-level round-robin scheduler is frozen in
`scheduling/round_robin_v1.py` under the run directory. It rotates through
Direct → ReAct → FOVEA-Greedy → ADAPTIVE_V4 among currently pending methods,
skipping idle methods without blocking the queue. Each physical retry rejoins
the queue; valid cache hits require no dispatch slot. The scheduler honors the
same persistent SQLite cooldown, 429 pause and effective 30/45/60-second gap.
Budget accounting and inference prompts are unchanged.

Resume skips all existing terminal rows without rewriting results, answers,
timestamps or call counts; completed Direct reports immediately unblock dependent
methods. Partial episodes reconstruct the same requests and reuse valid cached
responses. Startup refuses unresolved billing/holds. The initial run start time
is retained, with a separate resumed_at and resume_count. Earlier state/launch
and completed-row hashes are retained in scheduling/resume_checkpoint.json.
A read-only dry replay audit verifies partial request cache reuse before launch.
`rr_requests` and `rr_state` in the shared ledger expose request grants, waits,
cancellations and the persistent rotation pointer. `scheduling/progress.json` is
updated by the status command for the regular chat report.

## Source-only checkout

Results, source snapshots, ledgers, credentials and the local
`budget_carryover.json` accounting record are deliberately not published. To
resume an existing run, retain that accounting record and the complete existing
run directory; do not initialize a replacement ledger. The runner's fixed local
dataset, run and credential-file paths describe this campaign and must be
configured for a separate installation. The inactive HF provider diagnostics are
not part of the published Parasail runner.
