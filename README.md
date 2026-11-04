# Foveated Reasoning: V*191 experiments

This repository contains the code used by the November 2026 V*191 campaign:
Direct/ReAct/FOVEA-Greedy recovery v2, Adaptive V4, its execution-failure retries,
and the completed question-conditioned seed pilot. The previous partition/recheck implementation remains available in Git history
at commit `ef6a317`; the current default is this experiment implementation.
`source-manifest.json` records the original paths and SHA256 of every exported
source file. Runtime source files have not been changed by this export.

## Actual search budgets

| Method | Actual stopping mechanism |
| --- | --- |
| Direct | Original image response; no crop search |
| ReAct / Greedy | Prompt requests termination at turn 10; executor allows turns 1–49 (`range(1, 50)`) to mirror upstream recursion_limit=100. This is not an enforced 10-turn or 3-turn cap. |
| Adaptive V4 / retries | Hard maximum 10 crop **attempts**, 24 logical model calls including final, one bounded format repair per logical call. A crop attempt is not an API request. |

Greedy evaluates three scale candidates with five probes per candidate. `3×5`
is candidate/probe count, not three search turns. These runs do **not** establish
a paper-matched three-turn comparison. Early stopping can use fewer turns.

All current workers use `qwen/qwen3-vl-8b-instruct`, Parasail only, temperature 1,
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
