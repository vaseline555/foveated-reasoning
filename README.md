# Foveated Reasoning: V*191 experiments

This repository contains the code used by the November 2026 V*191 campaign:
Direct/ReAct/FOVEA-Greedy recovery v2, Adaptive V4, its execution-failure retries,
and the completed question-conditioned seed pilot. The previous partition/recheck implementation remains available in Git history
at commit `ef6a317`; the current default is this experiment implementation.
`source-manifest.json` records the original paths and SHA256 of every exported
source file. The manifest describes the original export; this checkout contains the interaction-budget revision documented below.

## Interaction budget (local revision)

The exported commit used a prompt-only ten-turn instruction for ReAct/Greedy,
with an executor permitting 49 planner rounds, and five probes per Greedy
candidate. This checkout now enforces the requested protocol below.

| Method | External interaction (hard maximum 10) | Separately counted computation |
| --- | --- | --- |
| Direct | None (0 interactions) | Initial answer generation |
| ReAct | Planner decision → tool attempt → observation/error feedback | Format retry, final reader |
| FOVEA-Greedy | Planner proposal → candidate evaluation/selection → selected crop observation/error feedback | 3 candidates × 3 probes, format retry, final reader |
| Adaptive(Graph) | Evidence-based selection → crop attempt → observation/evidence update | Initial role plan, candidate generation, format retry, final reader |

A failed tool/crop attempt consumes one interaction. Early stopping may use fewer
than ten. ReAct/Greedy allow the tenth crop and pass its observation to the final
reader without an eleventh planner call. Adaptive already counted crop attempts;
its controller now also enforces the hard ceiling if configured above ten. Its
separate 24-logical-call budget remains and can stop search earlier.

New result rows expose `interaction_turns`, `interaction_turn_limit`, and
`computation_counts`; successful crop count remains separate. Computation counts
report logical calls by role and format retries separately; physical requests,
including failed attempts, remain auditable in the API ledger. Direct baseline
computation is recorded in its own row. Existing cached campaign rows are preserved
and must not be presented as results of this revised protocol. The completed seed
pilot retains its historical settings. Frozen campaign code hashes intentionally
reject this modified source; use a separately prepared protocol for new runs,
rather than changing historical config or reusing old results for this comparison.

The current local Graph gate V2 runner is documented in
[local_campaign/README.md](local_campaign/README.md). It uses a cumulative $10
budget, a Direct-blind Graph gate followed by Direct fusion, and fair method
round-robin dispatch with a 30/45/60-second minimum gap depending on HTTP429
history. Runtime results, caches, ledgers and credentials are excluded from Git.

The original exported cloud workers used `qwen/qwen3-vl-8b-instruct`, Parasail only, temperature 1,
max_tokens 2048, the same cumulative $15 ledger and global 3-second dispatch gate.
Baseline and Adaptive may overlap requests, with one request per worker. The
$0.92410405 historical cost is already included in the campaign cap. Snapshot
JSON files document settings; they are not replacements for a live ledger/config.

## Environment and source placement

The experiment code retains the actual `/workspace/...` paths. It is the auditable implementation of the existing campaign, **not a self-contained migration of its private
ledger, frozen manifests, reused responses or health records**. Do not initialize
a new ledger to resume this campaign, overwrite frozen config, or launch a second
worker. A fresh independent experiment requires a separately prepared run protocol.

Python 3.12.14 was used. Dependency versions are in `requirements.txt`.
Upstream FOVEA is https://github.com/iamlilAJ/active-vlm at
`0dc77fb69a3ae31467e2416f677fae1d21bbc498` (MIT; see root LICENSE/NOTICE).
The adapter reads the vendored upstream source files. The local policy addition is
`fovea-official/src/cv_agent/policies/adaptive.py`; its design and tests are included.

On an **empty** workspace, install the exported sources at their recorded paths:

```bash
# Run from this repository; do not overwrite an active workspace.
python -m pip install -r requirements.txt
cp -R fovea-official /workspace/fovea-official
for name in fovea-reproduction-8b-recovery-v2 fovea-adaptive-experiment-v4 fovea-adaptive-retries fovea-seed-budget-pilot vstar-prepared; do
  cp -R "$name" "/workspace/$name"
done
python scripts/download_datasets.py --benchmarks vstar --data-dir /workspace/data
# prepare.py imports the downloader from /workspace/foveated-reasoning.
python /workspace/vstar-prepared/prepare.py
```

Dataset preparation separates labels (evaluation only) from the inference
manifest and verifies images. API credentials are supplied only through
`OPENROUTER_API_KEY` or the runtime secret manager. No keys are included here.

## Commands for the existing campaign

These commands use the original preserved run artifacts. Status commands do not
call the paid API; baseline status can create derived semantic overlays. Keep
original rows and responses intact.

```bash
SHARED=/workspace/fovea-reproduction-8b-recovery/runs/vstar191-8b-only-20261103
BASE=/workspace/fovea-reproduction-8b-recovery-v2
ADAPT=/workspace/fovea-adaptive-experiment-v4
OUT=$ADAPT/runs/vstar191-8b-v4

python "$BASE/run.py" status "$SHARED"
python "$BASE/audit_errors.py" "$SHARED"
python "$ADAPT/run.py" status "$OUT"
python /workspace/fovea-adaptive-retries/analyze.py

# Paid execution/resumption, ONLY if the corresponding worker is absent.
# Baseline reads config-v2.json and preserves existing results.
python -u "$BASE/run.py" run "$SHARED"
# Adaptive requires completed Direct reports, frozen config and health.ready.
python -u "$ADAPT/run.py" run "$OUT"
# Retries only the original nine execution failures; accepts first complete
# response, including wrong answers/NO_ANSWER. Existing accepted rows are reused.
python -u /workspace/fovea-adaptive-retries/retry.py
```

For an already prepared campaign before Adaptive health validation, the commands
are `python "$ADAPT/run.py" prepare "$OUT"`, followed by
`python "$ADAPT/run.py" smoke "$OUT" --case vstar_0` and the same command for
`vstar_115` and `vstar_1`. Full execution requires all three final responses and
at least one actual crop-observe-final path. The recorded campaign additionally
checked vstar_2 and vstar_3 to establish crop coverage; all successful checks count
within the same 191 questions. `launch_checks.py` coordinates this process using
`--first-smoke-pid`, and must not be launched alongside another coordinator.

For a background run, redirect the selected run command to its execution log and
use a process supervisor or `nohup ... > execution.log 2>&1 &`; check its PID and
worker lock before launching. The commands above are intentionally shown in the
foreground to avoid accidentally starting multiple paid workers.

Seed-pilot inspection (requires preserved `plan-v2.json` and run-v2):

```bash
python /workspace/fovea-seed-budget-pilot/pilot_v2.py status \
  --plan /workspace/fovea-seed-budget-pilot/plan-v2.json \
  --shared-root "$SHARED" --output /workspace/fovea-seed-budget-pilot/run-v2
```

Replace `status` with `run` only when intentionally executing the pilot under its
original serialized baseline/pilot protocol and $1.50 internal cap, including V1
cost. The 12-case pilot is complete; it is not a new spending allowance.

## Offline checks

Run each suite separately to avoid imports colliding between files named run.py:

```bash
(cd /workspace/fovea-reproduction-8b-recovery-v2 && python -m unittest discover -p 'test_*.py')
(cd /workspace/fovea-adaptive-experiment-v4 && python -m unittest discover -p 'test_*.py')
python -m unittest discover -s /workspace/fovea-official/tests -p test_adaptive.py
```

These checks make no inference calls. Runs, API payloads, datasets, evaluation
labels, credentials and SQLite ledgers are deliberately excluded from this export.
