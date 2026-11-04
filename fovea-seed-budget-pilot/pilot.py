"""Paired, label-blind first-crop seed-budget diagnostic.

Inference reuses the existing 8B-only transport, global pacing and shared ledger.
Only terminal evaluation reads benchmark labels.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, '/workspace/fovea-reproduction-8b-recovery-v2')
import run as baseline
from client import BudgetStop, RequestFailure, input_upper, save
import official_adapter as official
from PIL import Image

PREP = Path('/workspace/vstar-prepared')
METHOD = 'SEED_BUDGET_PILOT_V1'


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def paired_seed(key,stage):
    return int(hashlib.sha256((key+'/SEED_BUDGET_P1/'+stage).encode()).hexdigest()[:8],16)


def validate_seed(message,fake_url):
    calls=message.get('tool_calls') or []
    if len(calls)!=1 or calls[0].get('function',{}).get('name')!='cropping':
        return False
    try:
        args=json.loads(calls[0]['function']['arguments'])
        box=args['coordinates']
        if args.get('image_url')!=fake_url or len(box)!=4:
            return False
        if not all(type(x) in (int,float) and 0<=x<=1000 for x in box):
            return False
        official.validate_crop_coordinates(box)
    except (ValueError,TypeError,KeyError):
        return False
    return True


def seed_box(message):
    return list(official.validate_crop_coordinates(json.loads(message['tool_calls'][0]['function']['arguments'])['coordinates']))


def proposal_messages(case,original_url,fake_url,prior_boxes):
    text=('Propose exactly one image region useful for answering the question. '
          'Call cropping exactly once; do not answer the multiple-choice question. '
          'Use normalized [0,1000] coordinates; the box width and height must each be at least 50. '
          f'Use image_url {fake_url}.\n\nQuestion and answer choices:\n'+case['question'])
    if prior_boxes:
        text+='\nPreviously proposed region(s): '+json.dumps(prior_boxes)+'. Propose a different plausible location or broader useful context. Do not repeat the same region.'
    # The first proposal receives no total-budget or condition information.
    return [{'role':'system','content':'You are a visual region proposal generator.'},
            {'role':'user','content':[{'type':'image_url','image_url':{'url':original_url}},
                                      {'type':'text','text':text}]}]


def bbox_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return intersection/(aa+bb-intersection) if aa+bb>intersection else 0.0


def choose_candidate(scored,seed_cap):
    eligible=[x for x in scored if x['seed_index']<seed_cap]
    if not eligible:
        raise ValueError('No eligible candidates')
    # Stable order implements official first-candidate tie breaking.
    return max(eligible,key=lambda x:x['score'])


def shared_cost(client,method=METHOD):
    with client.connect() as connection:
        row=connection.execute("SELECT COALESCE(SUM(cost),0) paid,COALESCE(SUM(CASE WHEN cost IS NULL THEN reserved ELSE 0 END),0) held,COUNT(*) attempts FROM attempts WHERE method=?",(method,)).fetchone()
    return dict(row)


def stage_cost(client,case,stages,method=METHOD):
    if not stages:return 0.0
    expanded=[s for stage in stages for s in (stage,stage+'_format_retry1')]
    with client.connect() as connection:
        row=connection.execute('SELECT COALESCE(SUM(cost),0) FROM attempts WHERE method=? AND case_id=? AND stage IN ('+','.join('?' for _ in expanded)+')',(method,case,*expanded)).fetchone()
    return float(row[0])


def candidate_history(case,original_url,fake_url,choice,crop_url):
    tool_id='seed-budget-selected-crop'
    fake_crop=f"http://{case['key']}_seed_budget_selected.jpg"
    tool={'id':tool_id,'type':'function','function':{'name':'cropping','arguments':json.dumps(dict(image_url=fake_url,coordinates=choice['box']))}}
    data=dict(status='success',output_image=fake_crop)
    return [{'role':'assistant','content':'Inspect this selected region to answer the original question.','tool_calls':[tool]},
        {'role':'tool','tool_call_id':tool_id,'content':official.process_output(data)},
        {'role':'user','content':[{'type':'text','text':f'The tool successfully generated a new image. You can now refer to it as url: {fake_crop}.'},
                                  {'type':'image_url','image_url':{'url':crop_url}}]}]


class PilotClient(baseline.Client):
    """One shared physical ledger, with a smaller task reservation ceiling."""
    def __init__(self,*args,pilot_cap=1.5,**kwargs):
        self.pilot_cap=float(pilot_cap)
        super().__init__(*args,**kwargs)

    def _reserve(self,key,model,role,case,method,stage,upper,path):
        if method.startswith('SEED_BUDGET_PILOT'):
            with self.connect() as connection:
                committed=connection.execute(
                    "SELECT COALESCE(SUM(COALESCE(cost,reserved)),0) FROM attempts WHERE method LIKE 'SEED_BUDGET_PILOT%'"
                ).fetchone()[0]
            # The supervisor holds the campaign worker lock, and this pilot
            # dispatches one call at a time; no other pilot may race this check.
            if committed+upper>self.pilot_cap+1e-10:
                raise BudgetStop('Pilot physical paid+held reservation would exceed $'+str(self.pilot_cap))
        return super()._reserve(key,model,role,case,method,stage,upper,path)


def _call(client,shared_root,case,stage,messages,validator,tools=None,seed=None):
    baseline.check_stop(shared_root)
    seed=paired_seed(case['key'],stage) if seed is None else seed
    for retry in range(2):
        try:
            message,record=client.call(messages,stage=stage if not retry else stage+'_format_retry1',
                case=case['key'],method=METHOD,tools=tools,seed=(seed+retry)%(2**32),
                validator=validator,force_model='primary')
            return baseline.clean_message(message),record
        except RequestFailure as exc:
            if retry or 'OUTPUT_VALIDATION_FAILED' not in str(exc):raise
    raise RuntimeError('unreachable')


def original_view(case):
    with Image.open(case['image_path']) as source:
        original_url=official.jpeg_url(source)
    image=Image.open(io.BytesIO(base64.b64decode(original_url.split(',',1)[1])))
    image.load()
    return image,original_url


def score_seed(client,shared_root,output,case,image,original_url,index,previous):
    destination=output/'pools'/case['key']/f'seed_{index}.json'
    if destination.exists():
        return read(destination)
    fake_url=f"http://{case['key']}_original.jpg"
    proposal_stage=f'seed_{index}_proposal'
    messages=proposal_messages(case,original_url,fake_url,previous)
    proposal,proposal_record=_call(client,shared_root,case,proposal_stage,messages,
        lambda m:validate_seed(m,fake_url),tools=[official.crop_tool()])
    box=seed_box(proposal)
    scales=official.generate_candidates(box)
    scored=[]
    for ci,candidate_box in enumerate(scales):
        candidate=image.crop(official.absolute_coords(candidate_box,image.size,probe=True))
        messages=official.probe_messages(case['question'],image,None,candidate)
        contents=[];records=[];stages=[]
        for draw in range(5):
            stage=f'seed_{index}_candidate_{ci}_probe_{draw}'
            response,record=_call(client,shared_root,case,stage,messages,baseline.probe_valid)
            contents.append(response['content']);records.append(record);stages.append(stage)
        # Commit the exact crop that the native executor would return: floor
        # coordinates, JPEG90. Probe candidates above use official round coords.
        committed=image.crop(official.absolute_coords(candidate_box,image.size))
        stream=io.BytesIO();committed.save(stream,format='JPEG',quality=90,optimize=True)
        raw=stream.getvalue();asset=output/'crops'/case['key']/f'seed_{index}_candidate_{ci}.jpg'
        asset.parent.mkdir(parents=True,exist_ok=True);asset.write_bytes(raw)
        scored.append(dict(seed_index=index,candidate_index=ci,box=list(candidate_box),
            score=official.yes_score(contents),probe_contents=contents,records=records,stages=stages,
            crop_path=str(asset),crop_sha256=hashlib.sha256(raw).hexdigest()))
    row=dict(seed_index=index,box=box,proposal=proposal,proposal_record=proposal_record,
             proposal_stage=proposal_stage,candidates=scored,completed_at=time.time())
    save(destination,row)
    return row


def direct_logical_cost(row):
    # Common selected Direct response cost, reused identically by all arms;
    # past discarded attempts remain in the campaign's physical historical bill.
    records={r.get('id',str(i)):r for i,r in enumerate(row.get('records') or [])}
    return sum(float(r.get('cost') or r.get('cost_usd') or 0) for r in records.values())


def execute_case(client,shared_root,output,case):
    destination=output/'rows'/(case['key']+'.json')
    if destination.exists() and read(destination).get('status')=='complete':
        return read(destination)
    image,original_url=original_view(case)
    fake_url=f"http://{case['key']}_original.jpg"
    direct=baseline.effective_row(shared_root,'Direct',case)
    if not direct or direct.get('status')!='complete':
        raise ValueError('A fixed completed Direct analysis is required: '+case['key'])
    if direct.get('models')!=['qwen/qwen3-vl-8b-instruct']:
        raise ValueError('Direct analysis is not pure 8B')
    max_seeds=int(case.get('max_seeds',case.get('question_budget',0)))
    if max_seeds not in (1,2):
        raise ValueError('Frozen question budget must be 1 or 2')
    existing=read(destination) if destination.exists() else {}
    seeds=[];scored=[];readouts={};condition_results=copy.deepcopy(existing.get('conditions') or {})
    attempted=set(existing.get('attempted_seed_budgets') or [])
    common_cost=direct_logical_cost(direct)
    for index in range(2):
        attempted.add(index+1)
        save(destination,dict(case=case['key'],category=case['category'],status='partial',
            question_budget=max_seeds,required_roles=case.get('required_roles'),
            conditions=copy.deepcopy(condition_results),attempted_seed_budgets=sorted(attempted),at=time.time()))
        seed=score_seed(client,shared_root,output,case,image,original_url,index,[s['box'] for s in seeds])
        seeds.append(seed);scored.extend(seed['candidates']);k=index+1
        choice=choose_candidate(scored,k)
        raw=Path(choice['crop_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=choice['crop_sha256']:
            raise ValueError('Committed crop artifact changed')
        crop_url='data:image/jpeg;base64,'+base64.b64encode(raw).decode()
        history=candidate_history(case,original_url,fake_url,choice,crop_url)
        messages=official.aggregator_messages(case['question'],original_url,history,
                                             direct['message'].get('content') or '')
        readout_seed=paired_seed(case['key'],'common_final_readout')
        wire_key=digest(dict(messages=messages,seed=readout_seed))
        stage='final_'+wire_key[:20]
        if wire_key not in readouts:
            final,record=_call(client,shared_root,case,stage,messages,baseline.usable_analysis,seed=readout_seed)
            readouts[wire_key]=(final,record)
            save(output/'readouts'/case['key']/(wire_key+'.json'),dict(message=final,record=record,
                history=baseline.compact(history,output),seed=readout_seed,input_sha256=wire_key))
        else:
            final,record=readouts[wire_key]
        stages=[s['proposal_stage'] for s in seeds[:k]]+[stage]
        stages.extend(stage for s in seeds[:k] for c in s['candidates'] for stage in c['stages'])
        condition_results[str(k)]=dict(seed_budget=k,prediction=baseline.answer(final),
            grade=baseline.answer_grade(final),message=final,selected_seed=choice['seed_index'],
            selected_candidate=choice['candidate_index'],selected_box=choice['box'],
            selected_crop_sha256=choice['crop_sha256'],readout_input_sha256=wire_key,
            readout_seed=readout_seed,readout_stage=stage,readout_record=record,
            logical_cost_usd=stage_cost(client,case['key'],stages)+common_cost,
            common_direct_cost_usd=common_cost,logical_stage_names=stages,committed_crops=1)
        # Checkpoint K=1 before proposing seed2. A later failure cannot erase
        # this available arm or quietly remove the question from its denominator.
        partial=dict(case=case['key'],category=case['category'],status='partial',
            question_budget=max_seeds,required_roles=case.get('required_roles'),
            seed_boxes=[s['box'] for s in seeds],conditions=copy.deepcopy(condition_results),
            attempted_seed_budgets=sorted(attempted),
            completed_seed_budgets=list(condition_results),at=time.time())
        save(destination,partial)
    a,b=seeds[0]['box'],seeds[1]['box']
    distances=((a[0]+a[2]-b[0]-b[2])/2)**2+((a[1]+a[3]-b[1]-b[3])/2)**2
    row=dict(case=case['key'],category=case['category'],status='complete',
        question_budget=max_seeds,required_roles=case.get('required_roles'),
        seed_boxes=[s['box'] for s in seeds],seed_iou=bbox_iou(a,b),
        seed_center_distance_normalized=distances**.5/1000,
        question_rule_selected_k=max_seeds,conditions=condition_results,
        attempted_seed_budgets=[1,2],
        second_seed_selected=condition_results['2']['selected_seed']==1,
        readout_physically_shared=condition_results['1']['readout_input_sha256']==condition_results['2']['readout_input_sha256'],
        source_direct_row=direct.get('reused_from') or direct.get('source_result') or
            str(baseline.overlay_path(shared_root,'Direct',case) if baseline.overlay_path(shared_root,'Direct',case).exists()
                else baseline.result_path(shared_root,'Direct',case)),completed_at=time.time())
    save(destination,row)
    return row


def percentile(values,q):
    if not values:return None
    values=sorted(values);position=(len(values)-1)*q
    left=int(position);right=min(left+1,len(values)-1)
    return values[left]*(right-position)+values[right]*(position-left) if right!=left else values[left]


def policy_result(rows,labels,allocation):
    successes=[];costs=[]
    for row,k in zip(rows,allocation):
        result=row['conditions'][str(k)]
        successes.append(int(result['prediction']==labels[row['case']]))
        costs.append(result['logical_cost_usd'])
    return dict(correct=sum(successes),n=len(rows),accuracy=sum(successes)/len(rows) if rows else None,
                logical_cost_usd=sum(costs),seeds=sum(allocation),successes=successes,costs=costs)


def bootstrap_delta(a,b,draws=5000):
    import random
    if not a:return [None,None]
    rng=random.Random(20261103);values=[];n=len(a)
    differences=[x-y for x,y in zip(a,b)]
    for _ in range(draws):
        values.append(sum(differences[rng.randrange(n)] for _ in range(n))/n)
    return [percentile(values,.025),percentile(values,.975)]


def report(output,plan,client=None):
    """Terminal evaluator: the only function that reads ground-truth labels."""
    import itertools
    labels=read(PREP/'labels.evaluation-only.json')
    cases=plan.get('cohort',plan.get('cases',[]))
    order=[c['key'] for c in cases]
    rows=[read(output/'rows'/(key+'.json')) for key in order if (output/'rows'/(key+'.json')).exists()]
    complete=[r for r in rows if r.get('status')=='complete']
    n=len(complete)
    allocations={'fixed1':[1]*n,'fixed2':[2]*n,
                 'question_conditioned':[r['question_budget'] for r in complete]}
    policies={name:policy_result(complete,labels,allocation) for name,allocation in allocations.items()}
    random_results=[]
    high=sum(k==2 for k in allocations['question_conditioned'])
    for selected in itertools.combinations(range(n),high) if n else []:
        high_set=set(selected);allocation=[2 if i in high_set else 1 for i in range(n)]
        random_results.append(policy_result(complete,labels,allocation))
    random_accuracies=[x['accuracy'] for x in random_results]
    random_costs=[x['logical_cost_usd'] for x in random_results]
    q=policies['question_conditioned'];one=policies['fixed1'];two=policies['fixed2']
    gains=sum(r['conditions']['1']['prediction']!=labels[r['case']] and
              r['conditions']['2']['prediction']==labels[r['case']] for r in complete)
    harms=sum(r['conditions']['1']['prediction']==labels[r['case']] and
              r['conditions']['2']['prediction']!=labels[r['case']] for r in complete)
    categories={}
    for category in sorted({r['category'] for r in complete}):
        group=[r for r in complete if r['category']==category]
        categories[category]={name:policy_result(group,labels,
            [1 if name=='fixed1' else 2 if name=='fixed2' else r['question_budget'] for r in group])
            for name in policies}
    exact=dict(allocations=len(random_results),total_seed_budget=sum(allocations['question_conditioned']),
        mean_accuracy=sum(random_accuracies)/len(random_accuracies) if random_accuracies else None,
        accuracy_2_5_97_5=[percentile(random_accuracies,.025),percentile(random_accuracies,.975)],
        mean_logical_cost_usd=sum(random_costs)/len(random_costs) if random_costs else None,
        fraction_strictly_below_q=sum(x<q['accuracy'] for x in random_accuracies)/len(random_accuracies) if random_accuracies else None,
        fraction_equal_q=sum(x==q['accuracy'] for x in random_accuracies)/len(random_accuracies) if random_accuracies else None,
        fraction_at_least_q=sum(x>=q['accuracy'] for x in random_accuracies)/len(random_accuracies) if random_accuracies else None)
    if random_accuracies:
        exact['question_midrank_percentile']=100*(exact['fraction_strictly_below_q']+.5*exact['fraction_equal_q'])
    physical=shared_cost(client) if client else None
    coverage={str(k):dict(planned_n=len(cases),attempted_n=sum(k in r.get('attempted_seed_budgets',[]) for r in rows),
        result_n=sum(str(k) in r.get('conditions',{}) for r in rows),
        missing_result_n=len(cases)-sum(str(k) in r.get('conditions',{}) for r in rows)) for k in (1,2)}
    summary=dict(updated_at=time.time(),planned_n=len(cases),completed_n=n,failed_n=len(rows)-n,
        arm_coverage=coverage,
        partial=n!=len(cases),protocol='Single-round forced-seed first-crop diagnostic; not end-to-end V* reproduction.',
        physical_pilot=physical,shared_campaign=client.stats() if client else None,
        logical_cost_definition='Per-arm selected seed-generation, all native 3x5 probe calls for allowed seeds, one common reader, and the fixed reused Direct response cost. Shared physical artifacts are not billed repeatedly.',
        policies=policies,question_independent_exact_allocation=exact,
        fixed2_vs_fixed1=dict(recoveries=gains,harms=harms,
            paired_bootstrap_accuracy_delta_95=bootstrap_delta(two['successes'],one['successes'])),
        question_vs_fixed1_delta_95=bootstrap_delta(q['successes'],one['successes']),
        question_vs_fixed2_delta_95=bootstrap_delta(q['successes'],two['successes']),
        by_category=categories,
        manipulation=dict(mean_seed_iou=sum(r['seed_iou'] for r in complete)/n if n else None,
            high_overlap_pairs=sum(r['seed_iou']>=.8 for r in complete),
            second_seed_won=sum(r['second_seed_selected'] for r in complete),
            identical_final_inputs=sum(r['readout_physically_shared'] for r in complete)),
        limitations=['Question budget is frozen from question roles/category before labels; this pilot does not establish online adaptation.',
            'Geometric diversity does not establish target inclusion.',
            'A forced two-seed paired diagnostic estimates spending effects, not an early-stop seed-cap policy.',
            'Small exploratory cohort; all unfinished/error cases remain visible.'],
        details=complete)
    save(output/'evaluation.json',summary)
    save(output/'report.json',summary)
    return summary


def progress(output,plan,client):
    """Label-free live progress; terminal report alone loads labels."""
    cases=plan.get('cohort',plan.get('cases',[]))
    rows=[read(p) for p in (output/'rows').glob('*.json')]
    result=dict(updated_at=time.time(),planned_n=len(cases),
        complete_pairs=sum(r.get('status')=='complete' for r in rows),
        cases_attempted=len(rows),arm_result_n={str(k):sum(str(k) in r.get('conditions',{}) for r in rows) for k in (1,2)},
        physical_pilot=shared_cost(client),shared_campaign=client.stats())
    save(output/'progress.json',result)
    return result


def make_client(shared_root,plan,output):
    cfg=read(shared_root/'config-v2.json')
    if any(m['id']!='qwen/qwen3-vl-8b-instruct' for m in cfg['models'].values()):
        raise ValueError('8B-only model guard')
    return PilotClient(shared_root,cfg['models'],cap=cfg['cap_usd'],
        historical_usd=cfg['historical']['committed_usd'],min_request_interval=cfg['dispatch_interval_seconds'],
        rate_retry_limit=cfg['rate_retry_limit'],stop_callback=lambda:(output/'STOP').exists(),
        pilot_cap=plan.get('pilot_cap_usd',plan.get('protocol',{}).get('max_additional_usd',1.5)))


def run(shared_root,output,plan):
    from client import UnknownCharge,RateLimitPause
    output.mkdir(parents=True,exist_ok=True)
    if plan.get('code_sha256') and plan['code_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Frozen pilot source changed')
    manifest=Path(plan.get('source_manifest',str(PREP/'manifest.json')))
    if plan.get('source_manifest_sha256') and hashlib.sha256(manifest.read_bytes()).hexdigest()!=plan['source_manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    cases=plan.get('cohort',plan.get('cases',[]))
    if len({c['key'] for c in cases})!=len(cases) or not cases:
        raise ValueError('Invalid frozen cohort')
    for case in cases:
        if int(case.get('max_seeds',0)) not in (1,2):
            raise ValueError('Missing frozen question seed allowance')
        if any(key in case for key in ('label','gt','correct_answer','boxes','bbox')):
            raise ValueError('Inference plan contains evaluation-only fields')
        if case.get('image_sha256') and hashlib.sha256(Path(case['image_path']).read_bytes()).hexdigest()!=case['image_sha256']:
            raise ValueError('Frozen image changed: '+case['key'])
    save(output/'provenance.json',dict(plan_sha256=digest(plan),pilot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_config_sha256=hashlib.sha256((shared_root/'config-v2.json').read_bytes()).hexdigest(),
        official=official.provenance(),method=METHOD))
    client=make_client(shared_root,plan,output)
    save(output/'state.json',dict(status='running',pid=__import__('os').getpid(),started_at=time.time(),planned_n=len(cases)))
    code=0;status='complete';error=None
    try:
        for case in cases:
            try:
                execute_case(client,shared_root,output,case)
            except RequestFailure as exc:
                destination=output/'rows'/(case['key']+'.json')
                prior=read(destination) if destination.exists() else dict(case=case['key'],category=case.get('category'))
                prior.update(status='partial' if prior.get('conditions') else 'request_failure',error=str(exc),at=time.time())
                save(destination,prior)
            progress(output,plan,client)
    except BudgetStop as exc:
        status='budgetstop';error=str(exc)
    except RateLimitPause as exc:
        status='paused';error=str(exc);code=2
    except UnknownCharge as exc:
        status='stopped';error=str(exc);code=2
    except Exception as exc:
        import traceback
        status='stopped';error=type(exc).__name__+': '+str(exc)
        save(output/'fatal_error.json',dict(error=error,traceback=traceback.format_exc()))
        code=2
    summary=report(output,plan,client)
    if status=='complete' and summary['completed_n']!=len(cases):status='stopped'
    save(output/'state.json',dict(status=status,error=error,finished_at=time.time(),planned_n=len(cases),
                                  completed_n=summary['completed_n'],physical_pilot=shared_cost(client),exit_code=code))
    return code


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['run','status'])
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--shared-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plan=read(args.plan)
    if args.command=='run':raise SystemExit(run(args.shared_root,args.output,plan))
    client=make_client(args.shared_root,plan,args.output)
    print(json.dumps(report(args.output,plan,client),indent=2))
