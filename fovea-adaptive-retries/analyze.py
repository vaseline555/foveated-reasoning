"""Offline analysis only: labels never enter the retry worker."""
import json
import math
from pathlib import Path
import sqlite3
import time

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/workspace/fovea-adaptive-experiment-v4/runs/vstar191-8b-v4')
SHARED = Path('/workspace/fovea-reproduction-8b-recovery/runs/vstar191-8b-only-20261103')


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return {f.stem: read(f) for f in p.glob('*.json')}


def report():
    labels = read(Path('/workspace/vstar-prepared/labels.evaluation-only.json'))
    initial = rows(SOURCE / 'rows')
    accepted = rows(ROOT / 'accepted')
    supplemented = initial | accepted
    def correct(key, row):
        return row.get('status') == 'complete' and row.get('prediction') == labels[key]
    comparisons = {}
    for method in ('Direct', 'ReAct', 'FOVEA-Greedy'):
        other = rows(SHARED / 'rows' / method) | rows(SHARED / 'result_overlays' / method)
        common = [k for k,v in supplemented.items() if v['status']=='complete' and other.get(k,{}).get('status')=='complete']
        wins = sum(correct(k,supplemented[k]) and not correct(k,other[k]) for k in common)
        losses = sum(not correct(k,supplemented[k]) and correct(k,other[k]) for k in common)
        n = wins+losses
        p = min(1, 2*sum(math.comb(n,j) for j in range(min(wins,losses)+1))/2**n) if n else 1
        comparisons[method] = dict(paired_completed=len(common), adaptive_correct=sum(correct(k,supplemented[k]) for k in common),
            other_correct=sum(correct(k,other[k]) for k in common), adaptive_only=wins, other_only=losses, exact_mcnemar_p=p,
            other_complete=sum(v['status']=='complete' for v in other.values()), other_correct_total=sum(correct(k,v) for k,v in other.items()))
    c=sqlite3.connect(f'file:{SHARED}/ledger.sqlite?mode=ro',uri=True)
    c.row_factory=sqlite3.Row
    physical=[dict(v) for v in c.execute("select method,status,count(*) n,coalesce(sum(cost),0) cost_usd from attempts where method like 'ADAPTIVE_V4_RETRY_%' and status not in ('reserved_final','cancelled_reservation') group by method,status")]
    attempt_rows=[(str(f),read(f)) for f in sorted(ROOT.glob('round-*/rows/*.json'))]
    result=dict(updated_at=time.time(),retry_completed=len(accepted),retry_planned=9,
        initial_correct=sum(correct(k,v) for k,v in initial.items()),initial_denominator=191,
        supplemented_correct=sum(correct(k,v) for k,v in supplemented.items()), supplemented_completed=sum(v['status']=='complete' for v in supplemented.values()),
        supplemented_accuracy_full191=sum(correct(k,v) for k,v in supplemented.items())/191 if len(accepted)==9 else None,
        recovered=[dict(case=k,correct=correct(k,v),prediction=v['prediction'],retry_round=v['retry_round'],source=v['source_row']) for k,v in accepted.items()],
        failed_retry_attempts=[dict(path=f,case=v['case'],error=v.get('error')) for f,v in attempt_rows if v['status']!='complete'],
        retry_physical=physical,retry_paid_usd=sum(v['cost_usd'] for v in physical),comparisons=comparisons,
        limitation='Supplemented result uses extra attempts for Adaptive only. Report initial failures and all retry cost; paired comparison is not equal-compute first-pass superiority.')
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':
    report()
