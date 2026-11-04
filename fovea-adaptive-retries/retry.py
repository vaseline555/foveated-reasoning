"""Retry only the nine frozen V4 execution failures; accept first completed response."""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

sys.path.insert(0, '/workspace/fovea-adaptive-experiment-v4')
import run as original
import backend
import transport

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/workspace/fovea-adaptive-experiment-v4/runs/vstar191-8b-v4')
TARGETS = ['vstar_9', 'vstar_44', 'vstar_86', 'vstar_111', 'vstar_116', 'vstar_132', 'vstar_160', 'vstar_167', 'vstar_178']


def completed():
    return {p.stem: original.read(p) for p in (ROOT / 'accepted').glob('*.json')}


def state(status, **extra):
    value = dict(status=status, pid=os.getpid(), updated_at=time.time(), output=str(ROOT),
                 completed=len(completed()), planned=9, **extra)
    original.save(ROOT / 'state.json', value)
    original.save(original.SHARED / 'adaptive_retry_job.json', value)
    print(json.dumps(value), flush=True)


def main():
    with (ROOT / 'worker.lock').open('a+') as lock, (SOURCE / 'worker.lock').open('a+') as source_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = original.read(SOURCE / 'config.json')
        original.verify_sources(config)
        assert original.read(SOURCE / 'health.json')['ready']
        rows = {p.stem: original.read(p) for p in (SOURCE / 'rows').glob('*.json')}
        assert sorted(k for k, v in rows.items() if v['status'] == 'execution_failure') == sorted(TARGETS)
        plan_path = ROOT / 'plan.json'
        if not plan_path.exists():
            original.save(plan_path, dict(targets=TARGETS, original_output=str(SOURCE),
                original_row_sha256={k: original.sha(SOURCE / 'rows' / (k+'.json')) for k in TARGETS},
                config=config, retry_policy='Round robin unresolved execution failures until first complete response or shared budget/unknown-charge/STOP. Wrong and NO_ANSWER are accepted. No labels used.',
                namespace='ADAPTIVE_V4_RETRY_R{round}; namespace derives fresh stage seeds using unchanged backend hashing',
                created_at=time.time()))
        manifest = {c['key']: c for c in original.read(original.PREP / 'manifest.json')['cases']}
        round_number = 1
        while len(completed()) < len(TARGETS):
            method = f'ADAPTIVE_V4_RETRY_R{round_number}'
            original.METHOD = backend.METHOD = transport.METHOD = method
            output = ROOT / f'round-{round_number:03d}'
            output.mkdir(exist_ok=True)
            client = original.client_for(ROOT, config)
            for key in TARGETS:
                if key in completed():
                    continue
                if (ROOT / 'STOP').exists() or (original.SHARED / 'STOP').exists():
                    state('stopped', reason='STOP requested')
                    return
                state('running', current_case=key, round=round_number, method=method)
                try:
                    row = asyncio.run(original.execute_case(manifest[key], output, client, config))
                except original.RateLimitPause as exc:
                    state('paused', reason=str(exc), current_case=key, round=round_number)
                    return
                except (original.BudgetStop, original.UnknownCharge) as exc:
                    state('stopped', reason=str(exc), current_case=key, round=round_number)
                    return
                except (original.RequestFailure, ValueError, KeyError, TypeError) as exc:
                    row = original.failure_record(manifest[key], output, exc, 'execution_retry')
                    original.save(output / 'rows' / (key+'.json'), row)
                if row['status'] == 'complete':
                    original.save(ROOT / 'accepted' / (key+'.json'), dict(row,
                        retry_round=round_number, source_row=str(output / 'rows' / (key+'.json')),
                        initial_failure=str(SOURCE / 'rows' / (key+'.json'))))
                state('running', last_case=key, last_status=row['status'], round=round_number)
            round_number += 1
        state('complete', reason='All nine have a first completed response; correctness was not used for selection.')


if __name__ == '__main__':
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (ROOT / 'STOP').write_text('signal\n'))
    try:
        main()
    except Exception:
        original.save(ROOT / 'fatal_error.json', {'traceback': traceback.format_exc(), 'time': time.time()})
        state('attention_required', reason='See fatal_error.json')
        raise
