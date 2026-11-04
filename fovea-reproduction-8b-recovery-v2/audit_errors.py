"""Read-only audit of harness failures; never imports the inference client.

Only this script's output files are written. The inference source, ledger,
request/response artifacts, and result rows are never changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import quote
import uuid


ERROR_STATUSES = {'invalid', 'rejected', 'unknown', 'error'}
ACTIVE_STATUSES = {'reserved', 'dispatched'}


def load(path, warnings=None):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        if warnings is not None:
            warnings.append({'path': str(path), 'error': type(exc).__name__})
        return None


def save(path, value):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def number(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def answer_valid(message):
    content = message.get('content') or ''
    if not isinstance(content, str):
        return False
    match = re.search(r'<answer>(.*?)</answer>', content, re.I | re.S)
    text = (match.group(1) if match else content).strip().upper()
    if 'ERROR' in text or 'FAILED' in text:
        return False
    return bool(re.search(r'(?:^|[^A-Z])\(?([A-D])\)?\.?$', text))


def classify(attempt, request, response):
    """Describe observed contract violations, without attributing model truth."""
    stage = re.sub(r'_format_retry\d+$','',attempt.get('stage') or '')
    error = attempt.get('error') or ''
    http = response.get('http_status') if isinstance(response, dict) else None
    found = re.search(r'HTTP\s+(\d{3})', error)
    if http is None and found:
        http = int(found.group(1))
    if http is not None:
        group = ('authentication' if http in (401, 403) else
                 'credit_or_quota' if http == 402 else
                 'rate_limit' if http == 429 else
                 'server_error' if http >= 500 else 'request_rejected')
        return {'category': 'api_' + group, 'http_status': http,
                'certainty': 'observed_http_status', 'origin': 'api'}
    if error == 'COST_EXCEEDED_RESERVATION':
        return {'category': 'accounting_reservation_exceeded',
                'certainty': 'ledger_error', 'origin': 'accounting'}
    if attempt.get('status') == 'unknown':
        category = ('accounting_cost_unknown' if response is not None and not error else
                    'network_or_response_unknown')
        return {'category': category, 'certainty': 'cause_not_established',
                'origin': 'network_accounting_or_response'}
    choices = response.get('choices') if isinstance(response, dict) else None
    if not choices or not isinstance(choices[0], dict):
        return {'category': 'api_response_format', 'certainty': 'missing_or_invalid_choices',
                'origin': 'response_contract'}
    choice = choices[0]
    if choice.get('finish_reason') == 'length':
        return {'category': 'output_truncated', 'certainty': 'finish_reason_length',
                'origin': 'response_contract'}
    message = choice.get('message')
    if not isinstance(message, dict):
        return {'category': 'api_response_format', 'certainty': 'missing_message',
                'origin': 'response_contract'}
    calls = message.get('tool_calls') or []
    if stage.startswith('planner_') and calls:
        if len(calls) != 1:
            return {'category': 'tool_call_count', 'certainty': 'observed_contract_violation',
                    'origin': 'generated_tool_input'}
        function = calls[0].get('function') or {}
        if function.get('name') != 'cropping':
            return {'category': 'tool_name', 'certainty': 'observed_contract_violation',
                    'origin': 'generated_tool_input'}
        try:
            args = json.loads(function.get('arguments') or '')
            if not isinstance(args, dict):
                raise ValueError('Arguments are not an object')
        except (ValueError, TypeError):
            return {'category': 'tool_arguments_json', 'certainty': 'observed_contract_violation',
                    'origin': 'generated_tool_input'}
        # Recover all prior fake image references from the exact request artifact.
        known = {url.rstrip('.,') for text in strings(request or {})
                 for url in re.findall(r'http://[^\s<>"\']+', text)}
        if args.get('image_url') not in known:
            return {'category': 'tool_image_reference',
                    'certainty': 'reference_absent_from_recorded_request',
                    'origin': 'generated_tool_input'}
        box = args.get('coordinates')
        valid = (isinstance(box, list) and len(box) == 4 and
                 all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1000 for x in box))
        if valid:
            valid = box[2] > box[0] and box[3] > box[1]
        if valid and attempt.get('method') == 'FOVEA-Greedy':
            rounded = list(map(round, box))
            valid = rounded[2]-rounded[0] >= 50 and rounded[3]-rounded[1] >= 50
        if not valid:
            return {'category': 'crop_coordinates', 'certainty': 'observed_contract_violation',
                    'origin': 'generated_tool_input'}
    if stage.startswith('probe_'):
        content = message.get('content')
        if calls or not isinstance(content, str) or not re.search(r'\b(?:yes|no)\b', content, re.I):
            return {'category': 'probe_answer_format', 'certainty': 'observed_contract_violation',
                    'origin': 'generated_output'}
    elif stage.startswith('planner_') and not calls:
        if not isinstance(message.get('content'),str) or not message['content'].strip():
            return {'category':'planner_empty_output','certainty':'observed_contract_violation',
                    'origin':'generated_output'}
    elif stage in ('direct', 'aggregator'):
        if not calls and isinstance(message.get('content'),str) and message['content'].strip() and not answer_valid(message):
            return {'category':'model_no_valid_choice','certainty':'generated_answer_not_an_allowed_MC_option',
                    'origin':'model_answer_quality_not_harness'}
        if calls or not isinstance(message.get('content'),str) or not message['content'].strip():
            return {'category': 'final_answer_format', 'certainty': 'observed_contract_violation',
                    'origin': 'generated_output'}
    return {'category': 'validation_failure_unresolved', 'certainty': 'cause_not_established',
            'origin': 'unknown'}


def runtime_category(error):
    error = str(error).lower()
    if any(term in error for term in ('crop', 'coordinates', 'tile', 'image', 'jpeg', 'pil.')):
        return 'crop_or_image_runtime'
    if any(term in error for term in ('http ', 'api', 'openrouter', 'both backbones', 'validation')):
        return 'request_pipeline_failure'
    if 'recursion' in error:
        return 'search_recursion_limit'
    return 'runtime_failure_unresolved'


def snapshot(root):
    root = Path(root).resolve()
    warnings = []
    db = root / 'ledger.sqlite'
    attempts = []
    if db.exists():
        uri = 'file:' + quote(str(db), safe='/') + '?mode=ro'
        connection = sqlite3.connect(uri, uri=True, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA query_only=ON')
            connection.execute('BEGIN')
            attempts = [dict(row) for row in connection.execute('SELECT * FROM attempts ORDER BY started,id')]
            connection.rollback()
        finally:
            connection.close()
    state = load(root / 'state.json', warnings) or {}
    progress = load(root / 'progress.json', warnings) or {}
    results = {}
    failed_runs = []
    paths={str(p.relative_to(root/'rows')):p for p in (root/'rows').glob('*/*.json')}
    paths.update({str(p.relative_to(root/'result_overlays')):p for p in (root/'result_overlays').glob('*/*.json')})
    for path in sorted(paths.values()):
        row = load(path, warnings)
        if not isinstance(row, dict):
            continue
        key = (row.get('case') or path.stem, row.get('method') or path.parent.name)
        results[key] = row
        if row.get('status') != 'complete':
            failed_runs.append(dict(case=key[0], method=key[1], status=row.get('status'),
                error=row.get('error'), category=runtime_category(row.get('error')),
                completed_at=row.get('completed_at'), source_path=str(path)))
    valid_by_logical = defaultdict(list)
    for attempt in attempts:
        if attempt.get('status') == 'valid':
            valid_by_logical[attempt.get('logical_key')].append(attempt)
        elif attempt.get('status')=='invalid' and attempt.get('cost') is not None:
            # A fully received semantic abstention resolves an API retry even
            # when the old harness labelled it invalid. Preserve ledger bytes.
            response=load(attempt.get('response_path'),warnings)
            if classify(attempt,None,response)['category']=='model_no_valid_choice':
                valid_by_logical[attempt.get('logical_key')].append(attempt)
    events = []
    model_answer_events=[]
    for attempt in attempts:
        if attempt.get('status') not in ERROR_STATUSES:
            continue
        response = load(attempt.get('response_path'), warnings)
        request = load(attempt.get('request_path'), warnings)
        event = {key: attempt.get(key) for key in (
            'id','logical_key','case_id','method','stage','model','role','started','finished',
            'status','error','generation_id','request_path','response_path')}
        event['case'] = event.pop('case_id')
        event['confirmed_cost_usd'] = attempt.get('cost')
        event['held_usd'] = number(attempt.get('reserved')) if attempt.get('cost') is None else 0.0
        event.update(classify(attempt, request, response))
        if event['category']=='model_no_valid_choice':
            event['note']='Original invalid ledger status preserved; this is graded NO_ANSWER, not an API/harness failure.'
            model_answer_events.append(event)
            continue
        choices = response.get('choices') if isinstance(response, dict) else None
        event['finish_reason'] = choices[0].get('finish_reason') if choices and isinstance(choices[0], dict) else None
        later = [a for a in valid_by_logical[attempt.get('logical_key')]
                 if number(a.get('started')) >= number(attempt.get('started'))]
        recovered = min(later, key=lambda a: number(a.get('started'))) if later else None
        completed = results.get((event['case'], event['method']), {}).get('status') == 'complete'
        event['recovery'] = ({'status': 'same_logical_request_recovered',
            'attempt_id': recovered['id'], 'model': recovered.get('model'),
            'role': recovered.get('role'), 'via_30b_fallback': recovered.get('role') == 'fallback'}
            if recovered else {'status': 'case_completed_but_request_recovery_not_established'
                              if completed else 'not_recovered_as_of_snapshot'})
        events.append(event)
    runtime_events = []
    for name in ('fatal_error-v2.json', 'stop_reason-v2.json'):
        path = root / name
        item = load(path, warnings)
        if not isinstance(item, dict):
            continue
        runtime_events.append(dict(source_path=str(path), details=item,
            category='budget_or_user_stop' if item.get('type') == 'BudgetStop' else
            'charge_safety_stop' if item.get('type') == 'UnknownCharge' else
            runtime_category(item.get('message')), case=None, method=None))
    counts = Counter(event['category'] for event in events)
    tool_events=[]
    for path in sorted([*(root/'tool_errors').glob('*/*/*.json'),*(root/'tool_errors-v2').glob('*/*/*.json')]):
        item=load(path,warnings)
        if isinstance(item,dict):
            item=dict(item,source_path=str(path),category=runtime_category(item.get('error')),
                      handling='Returned to planner as a tool error for re-proposal; not a failed API request.')
            item['case_completed']=results.get((item.get('case'),item.get('method')),{}).get('status')=='complete'
            tool_events.append(item)
    unique_affected = {(event['case'], event['method']) for event in events}
    report = dict(schema_version=1, updated_at=time.time(), run_dir=str(root),
        worker_state=state.get('status', 'not_recorded'), worker_pid=state.get('pid'),
        snapshot_semantics='Ledger read-only transaction; atomic files may reflect a slightly later state.',
        definitions={
            'request_error_events':'One physical invalid/rejected/unknown attempt. Active requests are excluded.',
            'failed_case_methods':'Terminal non-complete result rows, counted separately from request events.',
            'model_wrong_answers':'Valid responses with incorrect labels are excluded; this audit never loads labels.',
            'attribution':'Categories describe observable contract/runtime symptoms, not proven model or harness root causes.',
            'recovery':'Only a later valid attempt with the identical logical key establishes request recovery.',
        },
        summary=dict(total_attempts=len(attempts),
            active_requests=sum(a.get('status') in ACTIVE_STATUSES for a in attempts),
            request_error_events=len(events), affected_case_methods=len(unique_affected),
            failed_case_methods=len(failed_runs),
            model_no_valid_choice_events=len(model_answer_events),
            no_answer_results=sum(r.get('grade')=='NO_ANSWER' for r in results.values()),
            request_error_categories=dict(sorted(counts.items())),
            recovered_request_error_events=sum(e['recovery']['status']=='same_logical_request_recovered' for e in events),
            recovered_via_30b_events=sum(e['recovery'].get('via_30b_fallback', False) for e in events),
            unresolved_request_error_events=sum(e['recovery']['status']!='same_logical_request_recovered' for e in events),
            confirmed_error_cost_usd=sum(number(e['confirmed_cost_usd']) for e in events),
            unresolved_error_hold_usd=sum(e['held_usd'] for e in events)),
        request_events=events, failed_case_methods=failed_runs, runtime_events=runtime_events,
        tool_execution_errors=tool_events,
        model_answer_quality_events=model_answer_events,
        progress_reference=str(root/'progress.json'), progress_updated_at=progress.get('updated_at'),
        audit_warnings=warnings)
    save(root / 'harness_errors.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        report = snapshot(args.run_dir)
        print(json.dumps({'updated_at':report['updated_at'], 'worker_state':report['worker_state'],
                          'summary':report['summary']}), flush=True)
        if not args.watch or report['worker_state'] in ('complete','stopped','paused'):
            if args.watch:
                save(args.run_dir / 'reports' / 'harness_errors_final.json', report)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
