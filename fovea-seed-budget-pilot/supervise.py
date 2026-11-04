"""Serialize a bounded seed-budget pilot with the existing paid baseline.

Both jobs use the SAME ledger, frozen model/prices, dispatch gate and $15 cap.
This coordinator does not make inference calls and never resets accounting.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import os

HERE=Path(__file__).resolve().parent
SHARED=Path('/workspace/fovea-reproduction-8b-recovery/runs/vstar191-8b-only-20261103')
BASE_CODE=Path('/workspace/fovea-reproduction-8b-recovery-v2')
OUTPUT=HERE/'run'


def read(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def ledger_health():
    with sqlite3.connect('file:'+str(SHARED/'ledger.sqlite')+'?mode=ro',uri=True) as db:
        unresolved=db.execute("SELECT COUNT(*) FROM attempts WHERE cost IS NULL OR status='unknown'").fetchone()[0]
        paused=db.execute('SELECT paused FROM api_health WHERE singleton=1').fetchone()[0]
        paid,held=db.execute('SELECT COALESCE(SUM(cost),0),COALESCE(SUM(CASE WHEN cost IS NULL THEN reserved ELSE 0 END),0) FROM attempts').fetchone()
    return dict(unresolved=unresolved,api_paused=bool(paused),run_paid=paid,held=held)


def spawn(command,log):
    with Path(log).open('ab',buffering=0) as handle:
        return subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
            close_fds=True,start_new_session=True)


def main():
    coordinator=(HERE/'supervisor.lock').open('a+')
    fcntl.flock(coordinator,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=HERE/'plan.json'
    config=read(SHARED/'config-v2.json')
    assert config['cap_usd']==15 and all(v['id']=='qwen/qwen3-vl-8b-instruct' for v in config['models'].values())
    state=read(SHARED/'state.json',{})
    job=dict(kind='seed_budget_pilot',supervisor_pid=os.getpid(),pilot_dir=str(HERE),
        output_dir=str(OUTPUT),started_at=time.time(),baseline_pid=state.get('pid'),
        task_cap_usd=1.5,campaign_cap_usd=15,
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),status='waiting_for_baseline')
    save(SHARED/'active_job.json',job)
    # Request a graceful stop; in-flight receipts must settle before the lock
    # transfers. Never kill a request or erase an uncertain charge.
    if state.get('status')=='running':
        if (SHARED/'STOP').exists():
            raise RuntimeError('An existing stop request must be reviewed, not overwritten')
        (SHARED/'STOP').write_text('Coordinator pause for shared-budget seed proposal pilot.\n')
        deadline=time.time()+240
        while time.time()<deadline:
            state=read(SHARED/'state.json',{})
            if state.get('status')!='running':
                break
            time.sleep(.5)
        if state.get('status')=='running':
            job.update(status='blocked',reason='Baseline did not settle; no duplicate paid worker started')
            save(SHARED/'active_job.json',job);return 2
    health=ledger_health()
    if health['unresolved'] or health['api_paused']:
        job.update(status='attention_required',reason='Shared ledger/API is not ready',ledger=health)
        save(SHARED/'active_job.json',job);return 2
    # The existing baseline uses this same file lock.
    lock=(SHARED/'worker.lock').open('a+')
    lock_deadline=time.time()+15
    while True:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time()>=lock_deadline:
                raise RuntimeError('Baseline lock did not settle; no duplicate paid worker started')
            time.sleep(.2)
    try:
        if (SHARED/'STOP').exists():
            text=(SHARED/'STOP').read_text()
            if text!='Coordinator pause for shared-budget seed proposal pilot.\n':
                raise RuntimeError('External stop request preserved; pilot not started')
            (SHARED/'STOP').unlink()
        job.update(status='running',ledger_at_start=health)
        save(SHARED/'active_job.json',job)
        pilot=spawn([sys.executable,'-u',str(HERE/'pilot.py'),'run','--plan',str(plan),
            '--shared-root',str(SHARED),'--output',str(OUTPUT)],HERE/'execution.log')
        job['pilot_pid']=pilot.pid;save(SHARED/'active_job.json',job)
        result=pilot.wait()
        health=ledger_health()
        job.update(pilot_exit_code=result,pilot_finished_at=time.time(),ledger_after_pilot=health,
                   report_path=str(OUTPUT/'report.json'))
        # Unknown charge or persistent quota pauses need attention; the
        # baseline must not redispatch around them.
        if health['unresolved'] or health['api_paused'] or (SHARED/'STOP').exists():
            job.update(status='attention_required',reason='Pilot stopped with unresolved charge, API pause, or user stop')
            save(SHARED/'active_job.json',job);return 2
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()
    baseline=spawn([sys.executable,'-u',str(BASE_CODE/'run.py'),'run',str(SHARED)],SHARED/'worker-v2.log')
    job.update(status='complete' if result==0 else 'pilot_stopped_baseline_resuming',
               baseline_resumed_pid=baseline.pid,finished_at=time.time())
    save(SHARED/'active_job.json',job)
    deadline=time.time()+15
    while time.time()<deadline:
        state=read(SHARED/'state.json',{})
        if state.get('pid')==baseline.pid and state.get('status')=='running':
            audit=spawn([sys.executable,'-u',str(BASE_CODE/'audit_errors.py'),str(SHARED),'--watch'],SHARED/'audit-v2.log')
            job['audit_resumed_pid']=audit.pid;save(SHARED/'active_job.json',job)
            break
        if baseline.poll() is not None:
            job.update(status='attention_required',reason='Baseline resume exited; inspect worker-v2.log')
            save(SHARED/'active_job.json',job);return 2
        time.sleep(.5)
    return result


def emergency_resume():
    """Restore the baseline after a coordinator setup error when it is safe."""
    job=read(SHARED/'active_job.json',{})
    if job.get('supervisor_pid')!=os.getpid():
        return
    pilot_pid=job.get('pilot_pid')
    if pilot_pid:
        try:
            # Do not launch alongside an orphaned but still active pilot.
            status=Path(f'/proc/{pilot_pid}/stat').read_text().split()[2]
            if status!='Z':
                return
        except OSError:
            pass
    state=read(SHARED/'state.json',{})
    if state.get('status')=='running':
        return
    health=ledger_health()
    if health['unresolved'] or health['api_paused']:
        return
    stop=SHARED/'STOP'
    if stop.exists():
        if stop.read_text()!='Coordinator pause for shared-budget seed proposal pilot.\n':
            return
        stop.unlink()
    # The baseline itself also enforces its exclusive worker lock.
    worker=spawn([sys.executable,'-u',str(BASE_CODE/'run.py'),'run',str(SHARED)],SHARED/'worker-v2.log')
    job.update(status='pilot_setup_failed_baseline_resuming',baseline_resumed_pid=worker.pid,
               coordinator_error_at=time.time())
    save(SHARED/'active_job.json',job)


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        save(HERE/'supervisor_error.json',dict(type=type(exc).__name__,message=str(exc),at=time.time()))
        try:
            emergency_resume()
        except Exception as resume_error:
            save(HERE/'supervisor_resume_error.json',dict(type=type(resume_error).__name__,
                                                       message=str(resume_error),at=time.time()))
        raise
