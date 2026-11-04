"""Read-only, label-blind answer extraction; labels are used only for scoring.

Graph has no pre-reader categorical answer in its current schema. Evidence claims
are not silently converted into predictions. No inference requests are made.
"""
import json
from pathlib import Path
import re
import sqlite3


def answer(message):
    # Same option extraction as the frozen baseline runner's answer().
    text=message.get('content') or ''
    if not isinstance(text,str):return None
    found=re.search(r'<answer>(.*?)</answer>',text,re.I|re.S)
    cleaned=(found.group(1) if found else text).strip().upper()
    if 'ERROR' in cleaned or 'FAILED' in cleaned:return None
    match=re.search(r'(?:^|[^A-Z])\(?([A-D])\)?\.?$',cleaned)
    return match.group(1) if match else None


def extract_planner(run, method, row):
    if method=='Direct':
        return dict(available=False,reason='no_separate_planner')
    gate_path=run/'graph/cases'/row['case']/'gate.json'
    if method=='Graph' and 'gate_prediction' not in row and gate_path.exists():
        gate=json.loads(gate_path.read_text())
        return dict(available=True,prediction=None if gate['answer']=='NO_ANSWER' else gate['answer'],reason='graph_only_gate',source=str(gate_path))
    if method=='Graph' and 'gate_prediction' in row:
        return dict(available=True,prediction=row['gate_prediction'],reason='graph_only_gate',
                    source=str(run/'graph/cases'/row['case']/'gate.json'))
    if method=='Graph':
        return dict(available=False,reason='graph_state_has_no_categorical_answer',
                    source=str(run/'graph/cases'/row['case']/'state.json'))
    records=[]
    for record in row.get('records',[]):
        stage=record.get('format_retry_of',record.get('stage',''))
        match=re.fullmatch(r'planner_(\d+)',stage)
        if match:records.append((int(match.group(1)),record))
    if not records:return dict(available=False,reason='no_planner_record')
    _,record=max(records,key=lambda x:x[0]);path=record.get('response_path')
    if not path or not Path(path).exists():return dict(available=False,reason='missing_raw_response')
    payload=json.loads(Path(path).read_text());choices=payload.get('choices') or []
    if not choices:return dict(available=False,reason='missing_choice')
    message=choices[0].get('message') or {}
    if message.get('tool_calls'):
        return dict(available=False,reason='last_planner_action_is_tool_not_answer',source=path)
    if not isinstance(message.get('content'),str) or not message['content'].strip():
        return dict(available=False,reason='empty_planner_response',source=path)
    prediction=answer(message)
    return dict(available=True,prediction=prediction,
                reason='terminal_answer' if prediction else 'terminal_NO_ANSWER',source=path,
                stage=record.get('stage'))


def summarize_cases(cases):
    completed=len(cases)
    paired=[c for c in cases if c['planner']['available']]
    unavailable={}
    for c in cases:
        if not c['planner']['available']:
            reason=c['planner']['reason'];unavailable[reason]=unavailable.get(reason,0)+1
    pc=sum(c['planner_correct'] for c in paired)
    rc=sum(c['reader_correct'] for c in cases)
    prc=sum(c['reader_correct'] for c in paired)
    matrix={
        'correct_to_correct':sum(c['planner_correct'] and c['reader_correct'] for c in paired),
        'correct_to_wrong':sum(c['planner_correct'] and not c['reader_correct'] for c in paired),
        'wrong_to_correct':sum(not c['planner_correct'] and c['reader_correct'] for c in paired),
        'wrong_to_wrong':sum(not c['planner_correct'] and not c['reader_correct'] for c in paired),
    }
    return dict(completed=completed,paired_count=len(paired),
                planner_correct=pc,planner_accuracy=pc/len(paired) if paired else None,
                reader_correct=rc,reader_accuracy=rc/completed if completed else None,
                paired_reader_correct=prc,paired_reader_accuracy=prc/len(paired) if paired else None,
                planner_NO_ANSWER=sum(c['planner'].get('prediction') is None for c in paired),
                unavailable_count=completed-len(paired),unavailable_reasons=unavailable,
                net_correct_change=prc-pc if paired else None,**matrix)


def build(run, rows_by_method, labels):
    details={};summary={}
    for method,rows in rows_by_method.items():
        details[method]=[]
        for row in rows:
            if row.get('status')!='complete':continue
            planner=extract_planner(run,method,row)  # Extraction never receives labels.
            truth=labels[row['case']]
            item=dict(case=row['case'],planner=planner,reader_prediction=row.get('prediction'),
                      reader_correct=row.get('prediction')==truth,
                      planner_correct=planner.get('prediction')==truth if planner['available'] else None)
            if planner['available']:
                item['transition']=('correct' if item['planner_correct'] else 'wrong')+'_to_'+('correct' if item['reader_correct'] else 'wrong')
            details[method].append(item)
        summary[method]=summarize_cases(details[method])
    gates=[]
    for path in (run/'graph/cases').glob('*/gate.json'):
        gate=json.loads(path.read_text());key=path.parent.name
        gates.append(dict(case=key,answer=gate['answer'],correct=gate['answer']==labels[key]))
    if 'Graph' in summary:
        summary['Graph'].update(gate_generated=len(gates),gate_correct_all=sum(g['correct'] for g in gates),
            gate_accuracy_all=sum(g['correct'] for g in gates)/len(gates) if gates else None,
            gate_pending_or_failed_final=len(gates)-summary['Graph']['paired_count'])
    return dict(methods=summary,cases=details,graph_gate_outputs=gates,
                definition='Only completed cases. Planner accuracy and transitions use available terminal planner answers; reader overall accuracy uses all completed cases. NO_ANSWER is incorrect. Graph uses its persisted Direct-blind gate answer. Legacy graph states without a categorical gate are unavailable. Direct has no separate planner or reader comparison.')


def markdown(audit):
    lines=['| Method | Complete | Paired | Planner acc. | Reader acc. (paired) | Reader acc. (all) | Correct→Wrong | Wrong→Correct | Unavailable |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    pct=lambda x:'—' if x is None else f'{x:.1%}'
    for method,t in audit['methods'].items():
        paired=t['paired_count']
        lines.append(f"| {method} | {t['completed']} | {paired} | {pct(t['planner_accuracy'])} | {pct(t['paired_reader_accuracy'])} | {pct(t['reader_accuracy'])} | {t['correct_to_wrong'] if paired else '—'} | {t['wrong_to_correct'] if paired else '—'} | {t['unavailable_count']} |")
    if 'Graph' in audit['methods']:
        g=audit['methods']['Graph']
        lines+=['',f"Graph gate-only all produced outputs: {g.get('gate_correct_all',0)}/{g.get('gate_generated',0)} correct; accuracy {pct(g.get('gate_accuracy_all'))}; pending/failed final: {g.get('gate_pending_or_failed_final',0)}."]
    lines+=['','Planner/paired-reader accuracy and transitions share the paired denominator. Unavailable is not scored as wrong. A terminal NO_ANSWER is scored wrong. Graph comparison uses its explicit Direct-blind gate answer; Direct has no separate planner. No new model calls are made.']
    return '\n'.join(lines)+'\n'
