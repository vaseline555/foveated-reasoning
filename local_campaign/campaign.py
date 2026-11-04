"""Independent local 191-case run. All paid calls share one capped SQLite ledger."""
from __future__ import annotations
import argparse
import asyncio
import concurrent.futures
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / 'fovea_bench/prepared/vstar-full-20261102'
RUN = REPO / 'runs/vstar191-graphgate-v2-8b-10usd-20261103'
METHODS = ('Direct', 'ReAct', 'Greedy', 'Graph')
MODEL = dict(id='qwen/qwen3-vl-8b-instruct', prompt_price=.25,
             completion_price=.75, context_length=262144, temperature=1.0,
             provider=dict(only=['Parasail'], allow_fallbacks=False, require_parameters=True))
MODELS = dict(primary=MODEL, fallback=MODEL)
CARRYOVER = REPO/'local_campaign/budget_carryover.json'
HISTORICAL_USD = json.loads(CARRYOVER.read_text())['previous_confirmed_usd'] if CARRYOVER.exists() else 0.
DISPATCH_GAP = 30.
STOP = threading.Event()
LOCK = threading.RLock()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    from PIL import Image
    if (RUN/'protocol.json').exists():
        return read(RUN/'protocol.json')
    manifest = read(DATA/'manifest.json'); labels = read(DATA/'labels.evaluation-only.json')
    assert len(manifest['cases']) == 191
    cases=[]; evaluation={}
    for c in manifest['cases']:
        p=(DATA/c['image_path']).resolve()
        assert sha(p) == c['image_sha256'], p
        with Image.open(p) as im: im.verify()
        key=c['source_id']; index=key.removeprefix('vstar_')
        cases.append(dict(key=key, id=index, dataset='vstar', question=c['question'],
                          image_path=str(p), image_sha256=c['image_sha256'],
                          original_size=c['original_size'], allowed_letters=c['allowed_letters']))
        evaluation[key]=labels[c['key']]
    assert len({c['key'] for c in cases}) == 191
    # Retain raw source and record each path-only relocation in the runnable snapshot.
    workspace=RUN/'source'
    original_hashes={}; runtime_hashes={}
    for folder in ('fovea-official','fovea-reproduction-8b-recovery-v2','fovea-adaptive-experiment-v4'):
        shutil.copytree(REPO/folder,workspace/folder,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
        for p in (workspace/folder).rglob('*'):
            if not p.is_file(): continue
            rel=str(p.relative_to(workspace)); original_hashes[rel]=sha(p)
            if p.suffix == '.py':
                s=p.read_text(); p.write_text(s.replace('/workspace/',str(workspace)+'/'))
            runtime_hashes[rel]=sha(p)
    gate_source=workspace/'fovea-adaptive-experiment-v4/graph_gate.py'
    shutil.copy2(REPO/'local_campaign/graph_gate.py',gate_source)
    original_hashes[str(gate_source.relative_to(workspace))]=sha(gate_source)
    runtime_hashes[str(gate_source.relative_to(workspace))]=sha(gate_source)
    random.Random(20261103).shuffle(cases)
    save(RUN/'prepared/manifest.json',dict(cases=cases,source=str(DATA/'manifest.json')))
    save(RUN/'prepared/labels.evaluation-only.json',evaluation)
    protocol=dict(created_at=time.time(), cap_usd=10., historical_usd=HISTORICAL_USD, methods=METHODS,
                  model=MODEL, max_interaction_turns=10, greedy_candidates=3,greedy_probes=3,
                  dispatch_interval_seconds=DISPATCH_GAP, concurrency=4, logical_call_limit_graph=25,
                  graph_version="graph-only-gate-then-direct-fusion-v2", gate_sees_direct=False,
                  manifest_sha256=sha(RUN/'prepared/manifest.json'),
                  labels_sha256=sha(RUN/'prepared/labels.evaluation-only.json'),
                  original_hashes=original_hashes,runtime_hashes=runtime_hashes,
                  seed=20261103, old_results_reused=False,
                  source_commit='6805a1d396c4bfb88f4a6bdcf4fa9ddb953df71d',
                  question_count=191, data=str(DATA),
                  policy='failed crop attempts count; retries/plan/probes/final reader counted separately; early stopping allowed')
    save(RUN/'protocol.json',protocol)
    return protocol


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def runtimes():
    workspace=RUN/'source'
    sys.path[:0]=[str(workspace/'fovea-adaptive-experiment-v4'),str(workspace/'fovea-reproduction-8b-recovery-v2')]
    import transport
    # Both runners must catch the exact transport exception classes in use.
    sys.modules['client']=transport._baseline
    base=load_module('campaign_baseline',workspace/'fovea-reproduction-8b-recovery-v2/run.py')
    graph=load_module('campaign_graph',workspace/'fovea-adaptive-experiment-v4/run.py')
    gate_module=load_module('campaign_graph_gate',workspace/'fovea-adaptive-experiment-v4/graph_gate.py')
    graph.Backend=gate_module.GraphGateBackend
    base.PREP=RUN/'prepared';graph.PREP=RUN/'prepared';graph.SHARED=RUN
    base.STOP=STOP
    opts=dict(cap=10.,historical_usd=HISTORICAL_USD,min_request_interval=DISPATCH_GAP,rate_retry_limit=8,
              stop_callback=lambda: STOP.is_set())
    client=transport._baseline.Client(RUN,MODELS,**opts)
    graph_client=transport.AdaptiveClient(RUN,MODELS,**opts)
    scheduler_path=RUN/'scheduling/round_robin_v1.py'
    if scheduler_path.exists():
        scheduler=load_module('campaign_round_robin',scheduler_path)
        scheduler.install(transport._baseline.Client,client,transport.RateLimitPause)
    return base,graph,client,graph_client


def rows_for(method):
    folder=RUN/'graph/rows' if method=='Graph' else RUN/'rows'/('FOVEA-Greedy' if method=='Greedy' else method)
    rows = {p.stem:read(p) for p in folder.glob('*.json')}
    if method != 'Graph':
        overlays = RUN/'result_overlays'/('FOVEA-Greedy' if method=='Greedy' else method)
        rows.update({p.stem:read(p) for p in overlays.glob('*.json')})
    return list(rows.values())


def report():
    protocol=read(RUN/'protocol.json'); state=read(RUN/'state.json') if (RUN/'state.json').exists() else {}
    labels=read(RUN/'prepared/labels.evaluation-only.json')
    budget=dict(confirmed_usd=0.,held_usd=0.,cap_usd=10.,remaining_usd=10.)
    paid={}; requests={}; stages={}; health={}; last_request=None
    if (RUN/'ledger.sqlite').exists():
        with sqlite3.connect(RUN/'ledger.sqlite',timeout=30) as db:
            db.row_factory=sqlite3.Row
            budget.update(dict(db.execute('SELECT COALESCE(SUM(cost),0) confirmed_usd,COALESCE(SUM(CASE WHEN cost IS NULL THEN reserved ELSE 0 END),0) held_usd FROM attempts').fetchone()))
            estimated=0.
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='attempt_routes'").fetchone():
                estimated=db.execute("SELECT COALESCE(SUM(a.cost),0) FROM attempts a JOIN attempt_routes r ON r.attempt_id=a.id WHERE r.accounting_basis='published_token_price_upper'").fetchone()[0]
            budget['published_price_upper_debit_usd']=estimated
            budget['accounting_note']='Includes published-price token upper estimates where the provider did not return a billed cost.' if estimated else 'Reported API costs.'
            budget['new_run_paid_usd']=budget['confirmed_usd']
            budget['historical_usd']=HISTORICAL_USD
            budget['confirmed_usd']+=HISTORICAL_USD
            budget['remaining_usd']=10.-budget['confirmed_usd']-budget['held_usd']
            for row in db.execute("SELECT method,COUNT(*) n,COALESCE(SUM(cost),0) cost FROM attempts WHERE status NOT IN ('reserved_final','cancelled_reservation') GROUP BY method"):
                paid[row['method']]=row['cost'];requests[row['method']]=row['n']
            health=dict(db.execute('SELECT * FROM api_health WHERE singleton=1').fetchone() or {})
            last_request=db.execute("SELECT MAX(started) FROM attempts WHERE status NOT IN ('reserved_final','cancelled_reservation')").fetchone()[0]
    pacing=read(RUN/'pacing.json') if (RUN/'pacing.json').exists() else {}
    gap=pacing.get('effective_min_gap_seconds',DISPATCH_GAP)
    planning=[3600*gap/3600,10800*gap/3600]
    now=time.time();elapsed=max(0,now-state.get('started',now));table={};etas=[]
    for method in METHODS:
        rows=rows_for(method); complete=[r for r in rows if r.get('status')=='complete'];n=len(complete)
        correct=sum(r.get('prediction')==labels[r['case']] for r in complete)
        # Completed episode wall times include shared-gate contention but omit initial dependency waiting.
        durations=[r['wall_seconds'] for r in rows if 'wall_seconds' in r]
        remaining=(191-len(rows))*(sum(durations)/len(durations)) if durations else None
        if remaining is not None: etas.append(remaining)
        ident={'Greedy':'FOVEA-Greedy','Graph':'ADAPTIVE_V4'}.get(method,method)
        table[method]=dict(completed=n,total=191,correct=correct,accuracy=correct/n if n else None,
                           failures=len(rows)-n, no_answer=sum(r.get('prediction') is None for r in complete),
                           interaction_turns=sum(r.get('interaction_turns',0) for r in complete),
                           paid_usd=paid.get(ident,0.),api_requests=requests.get(ident,0),
                           remaining_seconds=remaining)
    # Before all streams have observations, expose a protocol-based planning range.
    estimated=max(etas) if len(etas)==4 else None
    if state.get('status') in ('completed','stopped','failed'): estimated=0 if state['status']=='completed' else None
    out=dict(updated=now,state=state,elapsed_seconds=elapsed,budget=budget,methods=table,
             api_health=health,last_request_at=last_request,remaining_seconds=estimated,
             estimated_completion_at=now+estimated if estimated is not None else None,
             initial_planning_hours=planning,pacing=pacing)
    out['estimate_note']='Empirical ETA uses mean episode wall time per method and is unstable early. Before all methods finish a case, the planning range assumes 3600–10800 requests at the current effective dispatch gap. Budget exhaustion may stop earlier.'
    from reader_comparison import build, markdown
    audit=build(RUN,{m:rows_for(m) for m in METHODS},labels)
    audit['updated']=now
    save(RUN/'reader_comparison.json',audit)
    (RUN/'reader_comparison.md').write_text(markdown(audit))
    out['reader_comparison']=audit['methods']
    from scheduling_report import snapshot
    dispatch=snapshot(RUN/'ledger.sqlite') if (RUN/'ledger.sqlite').exists() else None
    if dispatch is not None:
        out['dispatch_schedule']=dispatch
        save(RUN/'scheduling/progress.json',dispatch)
    save(RUN/'progress.json',out)
    stamp=lambda x:datetime.fromtimestamp(x,ZoneInfo('America/Phoenix')).strftime('%m-%d %H:%M')
    lines=[f"Updated {stamp(now)} Arizona; state={state.get('status','prepared')}",
           f"Elapsed {elapsed/3600:.2f} h; paid ${budget['confirmed_usd']:.6f}; held ${budget['held_usd']:.6f}; cap $10",'',
           '| Method | Complete | Correct | Accuracy | Failures | Paid USD | Requests |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for m,t in table.items():
        acc=f"{t['accuracy']:.1%}" if t['accuracy'] is not None else '—'
        lines.append(f"| {m} | {t['completed']}/191 | {t['correct']} | {acc} | {t['failures']} | ${t['paid_usd']:.5f} | {t['api_requests']} |")
    lines+=['',f"ETA: {stamp(now+estimated)} ({estimated/3600:.2f} hours left)" if estimated is not None else f'ETA: collecting samples; planning range {planning[0]:.0f}–{planning[1]:.0f} hours at {gap:.0f}s/request.', f"HTTP429: {health.get('total_429',0)}"]
    lines+=['',markdown(audit)]
    (RUN/'progress.md').write_text('\n'.join(lines)+'\n')
    return out


def run():
    protocol=prepare()
    handle=(RUN/'worker.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for rel,digest in protocol['runtime_hashes'].items():
        if sha(RUN/'source'/rel)!=digest: raise ValueError('Snapshot changed: '+rel)
    if sha(RUN/'prepared/manifest.json')!=protocol['manifest_sha256']: raise ValueError('Manifest changed')
    if (RUN/'ledger.sqlite').exists():
        with sqlite3.connect(RUN/'ledger.sqlite') as db:
            if db.execute('SELECT COUNT(*) FROM attempts WHERE cost IS NULL').fetchone()[0]:
                raise RuntimeError('Unresolved billing/hold exists; reconcile before resuming')
    os.environ['OPENROUTER_API_KEY']=Path('/private/tmp/fovea-openrouter-api-key').read_text().strip()
    os.environ['SSL_CERT_FILE']='/etc/ssl/cert.pem'
    base,graph,client,graph_client=runtimes()
    signal.signal(signal.SIGTERM,lambda *_:STOP.set());signal.signal(signal.SIGINT,lambda *_:STOP.set())
    cases=read(RUN/'prepared/manifest.json')['cases']
    ready={c['key']:threading.Event() for c in cases};direct={};active={};errors=[]
    previous=read(RUN/'state.json') if (RUN/'state.json').exists() else {}
    existing={m:{r['case']:r for r in rows_for(m)} for m in METHODS}
    direct.update(existing['Direct'])
    for key in direct:ready[key].set()
    state=dict(pid=os.getpid(),started=previous.get('started',time.time()),resumed_at=time.time(),
               resume_count=previous.get('resume_count',0)+bool(previous),
               scheduling='method_round_robin_v1',status='running',active=active,errors=errors)
    def persist():
        with LOCK: save(RUN/'state.json',dict(state,updated=time.time()))
    def worker(method):
        for case in cases:
            if STOP.is_set():break
            if case['key'] in existing[method]:
                # Leave completed/terminal rows, metrics and timestamps byte-for-byte intact.
                continue
            if method!='Direct':
                while not ready[case['key']].wait(.5):
                    if STOP.is_set():return
                if direct.get(case['key'],{}).get('status')!='complete':continue
            with LOCK:active[method]=case['key'];persist()
            timing_path=RUN/'case_timings'/method/(case['key']+'.json')
            if timing_path.exists():timing=read(timing_path)
            else:
                # For a partial pre-upgrade case use its recorded predecessor finish
                # as an approximate first start; completed rows are never rewritten.
                was_active=previous.get('active',{}).get(method)==case['key']
                if not was_active and (RUN/'scheduling/resume_checkpoint.json').exists():
                    was_active=read(RUN/'scheduling/resume_checkpoint.json').get('active_before_stop',{}).get(method)==case['key']
                prior=max((r.get('completed_at',state['started']) for r in existing[method].values()),default=state['started'])
                timing=dict(first_started=prior if was_active else time.time(),estimated_pre_resume_start=was_active)
                save(timing_path,timing)
            begin=timing['first_started']
            try:
                if method=='Direct': row=base.direct(client,RUN,case)
                elif method=='Graph': row=asyncio.run(graph.execute_case(case,RUN/'graph',graph_client,{'limits':{'max_steps':10,'max_model_calls':24}}))
                else: row=base.trajectory(client,RUN,case,'FOVEA-Greedy' if method=='Greedy' else method,direct[case['key']])
                if method=='Graph':
                    gate=read(RUN/'graph/cases'/case['key']/'gate.json')
                    row['gate']=gate
                    row['gate_prediction']=None if gate['answer']=='NO_ANSWER' else gate['answer']
                    row['aggregator_prediction']=row['prediction']
                    row['controller_logical_model_calls']=row['logical_model_calls']
                    row['logical_model_calls']+=1
                    row['computation_counts']['gate_calls']=1
                row.update(started=begin,wall_seconds=time.time()-begin)
                if method=='Graph': save(RUN/'graph/rows'/(case['key']+'.json'),row)
                else: base.write_result(RUN,'FOVEA-Greedy' if method=='Greedy' else method,case,row)
                print(json.dumps(dict(event='case_complete',method=method,case=case['key'],prediction=row.get('prediction'),interactions=row.get('interaction_turns'),wall_seconds=row['wall_seconds'])),flush=True)
                if method=='Direct':direct[case['key']]=row
            except Exception as exc:
                kind=type(exc).__name__
                error=dict(method=method,case=case['key'],type=kind,error=str(exc),at=time.time())
                with LOCK:errors.append(error)
                save(RUN/'errors'/method/(case['key']+'.json'),error)
                print(json.dumps(error),flush=True)
                # Fail closed for uncertain charges, budget/rate exhaustion and unexpected code errors.
                if kind!='RequestFailure':
                    STOP.set(); state['stop_reason']=error
                else:
                    row=dict(case=case['key'],method=method,status='execution_failure',error=str(exc),completed_at=time.time(),wall_seconds=time.time()-begin)
                    if method=='Graph': save(RUN/'graph/rows'/(case['key']+'.json'),row)
                    else:base.write_result(RUN,'FOVEA-Greedy' if method=='Greedy' else method,case,row)
                    # Repeated execution failures indicate a systemic problem.
                    if sum(e['method']==method for e in errors)>=3:STOP.set();state['stop_reason']=error
            finally:
                if method=='Direct':ready[case['key']].set()
                with LOCK:active.pop(method,None);persist()
                report()
    persist()
    def monitor():
        while not STOP.wait(30):
            try:persist();report()
            except Exception as exc:print('REPORT_ERROR '+str(exc),flush=True)
    threading.Thread(target=monitor,daemon=True).start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(worker,m) for m in METHODS]
        for job in jobs:job.result()
    state['status']='stopped' if STOP.is_set() else (
        'completed' if all(len(rows_for(m))==191 for m in METHODS) else 'failed')
    state['finished']=time.time();persist();report()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','status']);args=parser.parse_args()
    if args.action=='prepare':print(json.dumps(prepare(),indent=2))
    elif args.action=='status':print(json.dumps(report(),ensure_ascii=False,indent=2))
    else:run()
