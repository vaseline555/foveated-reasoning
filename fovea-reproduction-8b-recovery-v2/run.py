"""Budgeted hosted reproduction of the public FOVEA crop-only workflow.

Upstream geometry, prompts, probe construction and aggregation are loaded by
official_adapter. Transport, local crops, accounting and reporting are adapters.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as futures
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import signal
import sqlite3
import subprocess
import threading
import time
import traceback

from PIL import Image
from client import Client, BudgetStop, UnknownCharge, RequestFailure, RateLimitPause, save
import official_adapter as official

HERE = Path(__file__).resolve().parent
PREP = Path('/workspace/vstar-prepared')
METHODS = ('Direct', 'ReAct', 'FOVEA-Greedy')
STOP = threading.Event()
HISTORICAL_ROOT = Path('/workspace/fovea-reproduction/runs/vstar191-20261103')
CONFIG_NAME = 'config-v2.json'
SOURCE_VERSION = 'semantic-no-answer-v2'


def make_client(root, config):
    if any(spec['id']!='qwen/qwen3-vl-8b-instruct' for spec in config['models'].values()):
        raise ValueError('Recovery campaign is restricted to the 8B backbone')
    return Client(root, config['models'], cap=config['cap_usd'],
                  historical_usd=config['historical']['committed_usd'],
                  min_request_interval=config['dispatch_interval_seconds'],
                  rate_retry_limit=config['rate_retry_limit'],
                  stop_callback=lambda: STOP.is_set() or (root/'STOP').exists())


def import_pure_rows(root):
    imported = {}
    for method in METHODS:
        imported[method] = []
        for path in sorted((HISTORICAL_ROOT/'rows'/method).glob('*.json')):
            row=read(path)
            if row.get('status')!='complete' or row.get('models')!=['qwen/qwen3-vl-8b-instruct']:
                continue
            target=root/'rows'/method/path.name
            if target.exists():
                raise ValueError('Refuse to overwrite a recovery row')
            row['reused_from']=str(path)
            row['source_row_sha256']=sha(path)
            row['reuse_note']='Completed entirely on 8B; no model response altered or rebilled.'
            save(target,row)
            imported[method].append(dict(case=row['case'],source=str(path),sha256=sha(path)))
    return imported


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def answer(message):
    # Use the official grader's cleaning rule, constrained to V*'s four options.
    text = message.get('content') or ''
    if not isinstance(text,str):
        return None
    found = re.search(r'<answer>(.*?)</answer>', text, re.I | re.S)
    cleaned = (found.group(1) if found else text).strip().upper()
    if 'ERROR' in cleaned or 'FAILED' in cleaned:
        return None
    match = re.search(r'(?:^|[^A-Z])\(?([A-D])\)?\.?$', cleaned)
    return match.group(1) if match else None


def compact(value, root):
    """Content-address image data; retain exact JPEG bytes without repetition."""
    if isinstance(value, str) and value.startswith('data:image/'):
        raw = base64.b64decode(value.split(',', 1)[1])
        digest = hashlib.sha256(raw).hexdigest()
        asset = root / 'assets' / (digest + '.jpg')
        asset.parent.mkdir(exist_ok=True, parents=True)
        if not asset.exists():
            asset.write_bytes(raw)
        return {'asset': str(asset), 'sha256': digest}
    if isinstance(value, list):
        return [compact(v, root) for v in value]
    if isinstance(value, dict):
        return {k: compact(v, root) for k, v in value.items()}
    return value


def result_path(root, method, case):
    return root / 'rows' / method / (case['key'] + '.json')


def overlay_path(root, method, case):
    return root / 'result_overlays' / method / (case['key'] + '.json')


def effective_row(root, method, case):
    for path in (overlay_path(root, method, case), result_path(root, method, case)):
        if path.exists():
            return read(path)
    return None


def effective_rows(root, method):
    rows = {p.stem:read(p) for p in (root/'rows'/method).glob('*.json')}
    rows.update({p.stem:read(p) for p in (root/'result_overlays'/method).glob('*.json')})
    return list(rows.values())


def write_result(root, method, case, row):
    """Preserve every original result; v2 replacements are explicit overlays."""
    original = result_path(root, method, case)
    if original.exists():
        row = dict(row, overrides_result=str(original), original_result_sha256=sha(original),
                   source_version=SOURCE_VERSION)
        save(overlay_path(root, method, case), row)
    else:
        save(original, row)


def usable_analysis(message):
    return (isinstance(message, dict) and not message.get('tool_calls')
            and isinstance(message.get('content'), str)
            and bool(message['content'].strip()))


def answer_grade(message):
    return 'NO_ANSWER' if answer(message) is None else 'UNSCORED_MCQ'


def recover_failed_results(root, only_case=None, only_method=None):
    """Reinterpret paid final analyses without rebilling or editing artifacts.

    A semantic abstention is a terminal model outcome, with a zero benchmark
    score; it remains useful as the Direct analysis supplied to the agent judge.
    Truncated, empty, unbilled, non-8B, and planner/probe responses are excluded.
    """
    root = Path(root)
    database = root/'ledger.sqlite'
    recovered = []
    if not database.exists():
        return recovered
    connection = sqlite3.connect('file:'+str(database)+'?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        for method in METHODS:
            if only_method and method != only_method:
                continue
            for original in sorted((root/'rows'/method).glob('*.json')):
                case = {'key':original.stem}
                if only_case and original.stem != only_case:
                    continue
                row = effective_row(root, method, case)
                if row is None or row.get('status') == 'complete':
                    continue
                prefix = 'direct' if method == 'Direct' else 'aggregator'
                attempts = connection.execute(
                    """SELECT * FROM attempts WHERE case_id=? AND method=?
                       AND model='qwen/qwen3-vl-8b-instruct' AND role='primary'
                       AND cost IS NOT NULL AND status IN ('invalid','valid')
                       AND finished IS NOT NULL AND (stage=? OR stage=?)
                       ORDER BY started DESC,id DESC""",
                    (case['key'],method,prefix,prefix+'_format_retry1')).fetchall()
                selected = None
                for raw_attempt in attempts:
                    attempt = dict(raw_attempt)
                    response_path = attempt.get('response_path')
                    if not response_path or not Path(response_path).exists():
                        continue
                    body = read(response_path)
                    choices = body.get('choices') or []
                    if not choices or choices[0].get('finish_reason') == 'length':
                        continue
                    message = choices[0].get('message')
                    if usable_analysis(message):
                        selected = (attempt, clean_message(message), response_path)
                        break
                if selected is None:
                    continue
                attempt,message,response_path = selected
                checkpoint_path = root/'checkpoints-v2'/method/(case['key']+'.json')
                if not checkpoint_path.exists():
                    checkpoint_path = root/'checkpoints'/method/(case['key']+'.json')
                checkpoint = read(checkpoint_path) if checkpoint_path.exists() else {}
                rounds = checkpoint.get('rounds') or []
                records = list(checkpoint.get('records') or []) if method != 'Direct' else []
                records.append(dict(attempt,cache_hit=True,
                                    reinterpreted_without_dispatch=True))
                output = dict(case=case['key'],method=method,status='complete',
                    prediction=answer(message),grade=answer_grade(message),message=message,
                    models=['qwen/qwen3-vl-8b-instruct'],records=records,
                    crops=0 if method=='Direct' else sum('crop_path' in x for x in rounds),
                    completed_at=attempt.get('finished'),
                    overlay_created_at=time.time(),source_version=SOURCE_VERSION,
                    semantic_reclassification=True,source_response=str(response_path),
                    source_response_sha256=sha(response_path),source_attempt_id=attempt['id'],
                    source_result=str(original),source_result_sha256=sha(original),
                    interpretation='Nonempty fully billed final analysis; NO_ANSWER is scored zero, not an API failure.')
                if method != 'Direct':
                    output.update(rounds=rounds,stop_reason='recovered_final_analysis')
                write_result(root,method,case,output)
                recovered.append(dict(case=case['key'],method=method,grade=output['grade'],
                                      source_attempt_id=attempt['id']))
    finally:
        connection.close()
    return recovered


def recover_direct_failures(root):
    return recover_failed_results(root,only_method='Direct')


def check_stop(root):
    if STOP.is_set() or (root / 'STOP').exists():
        raise BudgetStop('STOP requested')


def clean_message(message):
    # Strip provider-specific fields before sending assistant messages back.
    return {k: message[k] for k in ('role', 'content', 'tool_calls') if k in message}


def call(client, root, case, method, stage, messages, validator, tools=None):
    check_stop(root)
    seed = int(hashlib.sha256((case['key'] + '/' + method + '/' + stage).encode()).hexdigest()[:8], 16)
    try:
        message, record = client.call(messages, stage=stage, case=case['key'],
            method=method, tools=tools, seed=seed, validator=validator, force_model='primary')
    except RequestFailure as exc:
        if 'OUTPUT_VALIDATION_FAILED' not in str(exc):
            raise
        # One same-model format retry; preserve the original billed attempt.
        # No ground-truth labels, content repair, or extra views are supplied.
        check_stop(root)
        message, record = client.call(messages, stage=stage+'_format_retry1', case=case['key'],
            method=method, tools=tools, seed=(seed+1) % (2**32),
            validator=validator, force_model='primary')
        record['format_retry_of'] = stage
    return clean_message(message), record


def direct(client, root, case):
    path = result_path(root, 'Direct', case)
    existing = effective_row(root,'Direct',case)
    if existing and existing.get('status') == 'complete':
        return existing
    if existing:
        recover_failed_results(root,only_case=case['key'],only_method='Direct')
        existing = effective_row(root,'Direct',case)
        if existing and existing.get('status') == 'complete':
            return existing
    with Image.open(case['image_path']) as source:
        source.load()
        original = official.jpeg_url(source)
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': official.template('direct', question=case['question'])},
        {'type': 'image_url', 'image_url': {'url': original}},
    ]}]
    message, record = call(client, root, case, 'Direct', 'direct', messages,
                           usable_analysis)
    row = dict(case=case['key'], method='Direct', status='complete',
               prediction=answer(message),grade=answer_grade(message),message=message,
               models=[record['model']], records=[record], crops=0,
               completed_at=time.time())
    write_result(root,'Direct',case,row)
    return row


def planner_valid(message, images, greedy):
    calls = message.get('tool_calls') or []
    if not calls:
        # Official planner emits a natural-language analysis/answer; the
        # separate final aggregator is responsible for the MCQ answer.
        return isinstance(message.get('content'), str) and bool(message['content'].strip())
    if len(calls) != 1:
        return False
    for tool in calls:
        if tool.get('function', {}).get('name') != 'cropping':
            return False
        try:
            args = json.loads(tool['function']['arguments'])
            if args.get('image_url') not in images:
                return False
            box = args['coordinates']
            if len(box) != 4 or not all(type(x) in (int, float) for x in box):
                return False
            if not all(0 <= x <= 1000 for x in box) or box[2] <= box[0] or box[3] <= box[1]:
                return False
        except (ValueError, KeyError, TypeError):
            return False
    return True


def probe_valid(message):
    if message.get('tool_calls') or not isinstance(message.get('content'), str):
        return False
    # Preserve upstream tag-optional yes/no parsing; missing both is an error,
    # rather than silently inventing an uninformative zero score.
    return bool(re.search(r'\b(?:yes|no)\b', message['content'], re.I))


def trajectory(client, root, case, method, baseline):
    path = result_path(root, method, case)
    existing = effective_row(root,method,case)
    if existing and existing.get('status') == 'complete':
        return existing
    if existing:
        recover_failed_results(root,only_case=case['key'],only_method=method)
        existing = effective_row(root,method,case)
        if existing and existing.get('status') == 'complete':
            return existing
    greedy = method == 'FOVEA-Greedy'
    with Image.open(case['image_path']) as source:
        original_image = source.convert('RGB')
    original_url = official.jpeg_url(original_image)
    # The official hosted original is JPEG85; crop the same decoded pixels.
    original_image = Image.open(io.BytesIO(base64.b64decode(original_url.split(',', 1)[1])))
    original_image.load()
    fake_original = f"http://{case['key']}_original.jpg"
    images = {fake_original: original_image}
    history = [
        {'role': 'system', 'content': official.template('planner')},
        {'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': original_url}},
            {'type': 'text', 'text': case['question']},
            {'type': 'text', 'text': f'\nOrignal image URL: {fake_original}\n'},
            {'type': 'text', 'text': '\nYou are on TURN 1 (max 10)'},
        ]},
    ]
    records, rounds = [], []
    crops = 0
    stop_reason = None
    # Upstream graph recursion_limit=100; two graph nodes per search turn,
    # plus direct/final. Keep nominal turn10 instruction, not a hard10 cutoff.
    for turn in range(1, 50):
        if turn > 1:
            history.append({'role': 'user', 'content':
                f'SYSTEM STATUS UPDATE: You are currently on Turn {turn} out of 10.\n'
                f'If {turn} == 10, you MUST stop calling tools and '
                'provide your best final answer immediately in the required format.'})
        message, record = call(client, root, case, method, f'planner_{turn}', history,
            lambda m: planner_valid(m, images, greedy), tools=[official.crop_tool()])
        records.append(record)
        history.append(message)
        calls = message.get('tool_calls') or []
        if not calls:
            stop_reason = 'model_stop'
            break
        for ci, tool in enumerate(calls):
            args = json.loads(tool['function']['arguments'])
            parent_key = args['image_url']
            parent = images[parent_key]
            proposal = args['coordinates']
            selected = proposal
            round_record = dict(turn=turn, tool_index=ci, parent=parent_key, proposal=proposal)
            if greedy:
                try:
                    validated = official.validate_crop_coordinates(proposal)
                except ValueError as exc:
                    # Match the official executor: report tool failure to the
                    # agent, retaining the proposal, then let it re-propose.
                    error = dict(status='error',message=f'Tool call failed: {exc}')
                    history.append({'role':'tool','tool_call_id':tool['id'],
                                    'content':json.dumps(error)})
                    event = dict(case=case['key'],method=method,stage=f'crop_{turn}_{ci}',
                                 proposal=proposal,parent=parent_key,error=str(exc),at=time.time())
                    save(root/'tool_errors-v2'/method/case['key']/f'{turn}_{ci}.json',event)
                    round_record['tool_error'] = event
                    rounds.append(round_record)
                    continue
                candidates = official.generate_candidates(validated)
                scores, probe_records = [], []
                for candidate_index, box in enumerate(candidates):
                    candidate = parent.crop(official.absolute_coords(box, parent.size, probe=True))
                    messages = official.probe_messages(case['question'], original_image,
                        None if parent_key == fake_original else parent, candidate)
                    contents = []
                    for draw in range(5):
                        probe, receipt = call(client, root, case, method,
                            f'probe_{turn}_{ci}_{candidate_index}_{draw}', messages, probe_valid)
                        records.append(receipt)
                        probe_records.append(receipt)
                        contents.append(probe['content'])
                    scores.append(official.yes_score(contents))
                winner = max(range(len(scores)), key=lambda i: scores[i])
                selected = candidates[winner]
                round_record.update(candidates=candidates, scores=scores, selected_index=winner,
                                    probe_records=probe_records)
            absolute = official.absolute_coords(selected, parent.size)
            cropped = parent.crop(absolute)
            if min(cropped.size) <= 0:
                raise RequestFailure('Selected crop is empty')
            # Official bundled MCP emits JPEG90; serve bytes locally as data URL.
            buffer = io.BytesIO()
            cropped.save(buffer, format='JPEG', quality=90, optimize=True)
            raw = buffer.getvalue()
            crops += 1
            crop_file = root / 'crops' / method / case['key'] / f'{crops}.jpg'
            crop_file.parent.mkdir(parents=True, exist_ok=True)
            crop_file.write_bytes(raw)
            decoded = Image.open(io.BytesIO(raw)); decoded.load()
            fake_crop = f"http://{case['key']}_cropping_{crops}.jpg"
            images[fake_crop] = decoded
            data = {'status': 'success', 'output_image': fake_crop}
            if greedy:
                data.update(original_coordinates=proposal, boed_selected_coordinates=list(selected))
            history.append({'role': 'tool', 'tool_call_id': tool['id'],
                            'content': official.process_output(data)})
            history.append({'role': 'user', 'content': [
                {'type': 'text', 'text': 'The tool successfully generated a new image. '
                 f'You can now refer to it as url: {fake_crop}. It can'
                 'works as tool input using the url.'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode()}},
            ]})
            round_record.update(selected=list(selected), absolute=absolute, crop_path=str(crop_file))
            rounds.append(round_record)
        save(root / 'checkpoints-v2' / method / (case['key'] + '.json'),
             compact(dict(history=history, rounds=rounds, records=records), root))
    if stop_reason is None:
        # Do not score an extra reader after exhausting upstream recursion.
        raise RequestFailure('Official recursion-equivalent search limit reached')
    messages = official.aggregator_messages(case['question'], original_url, history,
                                            baseline['message'].get('content') or '')
    final, record = call(client, root, case, method, 'aggregator', messages,
                         usable_analysis)
    records.append(record)
    models = sorted({r['model'] for r in records} | set(baseline['models']))
    row = dict(case=case['key'], method=method, status='complete', prediction=answer(final),
               grade=answer_grade(final),message=final, models=models, records=records, crops=crops,
               baseline_models=baseline['models'], stop_reason=stop_reason,
               rounds=rounds, completed_at=time.time())
    write_result(root,method,case,row)
    save(root / 'trajectories' / method / (case['key'] + '.json'), compact(history, root))
    return row


def summarize(root, client):
    labels = read(PREP / 'labels.evaluation-only.json')
    summary = dict(updated_at=time.time(), total_questions=191,
                   budget=client.stats(), methods={})
    for method in METHODS:
        rows = effective_rows(root,method)
        complete = [r for r in rows if r['status'] == 'complete']
        pure = [r for r in complete if r['models'] == [client.models['primary']['id']]]
        correct = sum(r['prediction'] == labels[r['case']] for r in complete)
        pure_correct = sum(r['prediction'] == labels[r['case']] for r in pure)
        summary['methods'][method] = dict(completed=len(complete), correct=correct,
            accuracy_completed=correct/len(complete) if complete else None,
            accuracy_full191=correct/191 if len(complete)==191 else None,
            backbone=client.models['primary']['id'],
            pure8b_completed=len(pure), pure8b_correct=pure_correct,
            reused8b_completed=sum(bool(r.get('reused_from')) for r in complete),
            other_or_mixed_completed=len(complete)-len(pure),
            no_answer=sum(r.get('prediction') is None for r in complete),
            semantic_reclassifications=sum(bool(r.get('semantic_reclassification')) for r in complete),
            errors=len(rows)-len(complete), pending=191-len(rows),
            crops=sum(r.get('crops', 0) for r in complete))
    save(root / 'progress.json', summary)
    return summary


def reporter(root, client):
    while not STOP.wait(1800):
        report = summarize(root, client)
        save(root / 'reports' / (str(int(time.time())) + '.json'), report)
        print('PERIODIC_REPORT ' + json.dumps(report), flush=True)


def prepare(root,config_name=CONFIG_NAME):
    root.mkdir(parents=True,exist_ok=True)
    destination=root/config_name
    if destination.exists():
        return read(destination)
    previous=root/'config.json'
    if not previous.exists():
        raise ValueError('Versioned recovery requires the preserved original config.json')
    config=copy.deepcopy(read(previous))
    config.update(source_version=SOURCE_VERSION,version=2,created_at=time.time(),
                  previous_config=str(previous),previous_config_sha256=sha(previous),
                  code_hashes={name:sha(HERE/name) for name in ('run.py','client.py','official_adapter.py')})
    config.setdefault('protocol',{}).setdefault('deviations',[]).append(
        'Nonempty Direct/final analyses are model outcomes: absent MCQ choices are NO_ANSWER and score zero; paid original responses are retained via explicit overlays.')
    save(destination,config)
    return config


def run(root,config_name=CONFIG_NAME):
    # A process lock prevents accidentally launching another paid worker.
    import fcntl
    handle = (root / 'worker.lock').open('a+')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(root / config_name)
    for name, digest in config['code_hashes'].items():
        if sha(HERE / name) != digest:
            raise ValueError('Frozen runner changed: ' + name)
    if sha(PREP / 'manifest.json') != config['manifest_sha256']:
        raise ValueError('Manifest changed')
    if sha(HISTORICAL_ROOT/'ledger.sqlite') != config['historical']['ledger_sha256']:
        raise ValueError('Historical spend source changed; do not start a separate allowance')
    for method, records in config.get('reused_pure8b_rows',{}).items():
        for record in records:
            if sha(record['source'])!=record['sha256']:
                raise ValueError('Historical result changed: '+record['case'])
            reused=read(root/'rows'/method/(record['case']+'.json'))
            if reused.get('models')!=['qwen/qwen3-vl-8b-instruct']:
                raise ValueError('Non-8B reused result')
    if config.get('official_adapter') != official.provenance():
        raise ValueError('Official source or adapter provenance changed')
    for case in read(PREP / 'manifest.json')['cases']:
        if sha(case['image_path']) != case['image_sha256']:
            raise ValueError('Source image changed: ' + case['key'])
    recoveries=recover_failed_results(root)
    client = make_client(root, config)
    save(root / 'state.json', dict(status='running',pid=os.getpid(),started_at=time.time(),
        source_version=SOURCE_VERSION,config_name=config_name,recovered_model_results=recoveries))
    cases = read(PREP / 'manifest.json')['cases']
    random.Random(config['seed']).shuffle(cases)
    threading.Thread(target=reporter, args=(root,client), daemon=True).start()
    summarize(root,client)

    def work(case, only_direct=False):
        try:
            baseline = direct(client, root, case)
            if baseline.get('status') != 'complete':
                return
            if not only_direct:
                methods = ['ReAct','FOVEA-Greedy']
                if int(case['id']) % 2:
                    methods.reverse()
                for method in methods:
                    try:
                        trajectory(client,root,case,method,baseline)
                    except RequestFailure as exc:
                        write_result(root,method,case,dict(case=case['key'],method=method,
                            status='request_failure',error=str(exc),completed_at=time.time()))
        except (BudgetStop,UnknownCharge,RateLimitPause) as exc:
            STOP.set()
            save(root / 'stop_reason-v2.json',dict(type=type(exc).__name__,message=str(exc),at=time.time()))
        except RequestFailure as exc:
            write_result(root,'Direct',case,dict(case=case['key'],method='Direct',
                 status='request_failure',error=str(exc),completed_at=time.time()))
        except Exception as exc:
            STOP.set()
            save(root / 'fatal_error-v2.json',dict(type=type(exc).__name__,message=str(exc),
                                              traceback=traceback.format_exc()))
        finally:
            print('CASE_PROGRESS ' + json.dumps(summarize(root,client)), flush=True)

    try:
        for phase in ('direct','search'):
            save(root / 'phase.json',dict(phase=phase,at=time.time()))
            with futures.ThreadPoolExecutor(max_workers=config['workers']) as pool:
                pending = {}
                iterator = iter(cases)
                def submit_one():
                    if STOP.is_set(): return
                    case = next(iterator,None)
                    if case is not None:
                        pending[pool.submit(work,case,phase=='direct')] = case['key']
                for _ in range(config['workers']): submit_one()
                while pending:
                    done,_ = futures.wait(pending,return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        del pending[future]
                        future.result()
                        submit_one()
            if STOP.is_set(): break
        report = summarize(root,client)
        done = all(x['completed']==191 for x in report['methods'].values())
        stopped_reason=read(root/'stop_reason-v2.json') if (root/'stop_reason-v2.json').exists() else {}
        status='complete' if done else 'paused' if stopped_reason.get('type')=='RateLimitPause' else 'stopped'
        save(root / 'state.json',dict(status=status,
                                    pid=os.getpid(),finished_at=time.time(),summary=report,
                                    source_version=SOURCE_VERSION,config_name=config_name))
        save(root / 'reports' / 'final-v2.json',report)
    finally:
        STOP.set()
        fcntl.flock(handle,fcntl.LOCK_UN)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run','status'])
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--config-name',default=CONFIG_NAME)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.run_dir,args.config_name),indent=2))
    elif args.command == 'status':
        cfg=read(args.run_dir/args.config_name)
        recover_failed_results(args.run_dir)
        print(json.dumps(summarize(args.run_dir,make_client(args.run_dir,cfg)),indent=2))
    else:
        for sig in (signal.SIGTERM,signal.SIGINT):
            signal.signal(sig,lambda *_:STOP.set())
        run(args.run_dir,args.config_name)
