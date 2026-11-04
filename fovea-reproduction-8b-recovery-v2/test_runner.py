"""Offline integration tests for the hosted runner's official crop workflow."""
import json
import copy
import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import run


class FakeClient:
    def __init__(self, tool_count=1):
        self.calls=[]
        self.tool_count=tool_count

    def call(self, messages, stage, case, method, tools=None, seed=None, validator=None, **kwargs):
        # Enforce the API's tool-result ordering, not just expected call counts.
        waiting=set()
        for item in messages:
            role=item['role']
            if role=='assistant' and item.get('tool_calls'):
                if waiting: raise AssertionError('Assistant while tool results pending')
                waiting={t['id'] for t in item['tool_calls']}
            elif role=='tool':
                if item['tool_call_id'] not in waiting:
                    raise AssertionError('Unexpected tool result')
                waiting.remove(item['tool_call_id'])
            elif role in {'user','assistant'} and waiting:
                raise AssertionError('User/assistant before all tool results')
        if waiting: raise AssertionError('Missing tool results at API dispatch')
        self.calls.append(dict(stage=stage,method=method,tools=tools,
            messages=copy.deepcopy(messages),seed=seed,kwargs=copy.deepcopy(kwargs)))
        if stage=='planner_1':
            message={'role':'assistant','content':'Inspect relevant objects.','tool_calls':[
                {'id':f'tool-{i}','type':'function','function':{'name':'cropping',
                 'arguments':json.dumps({'image_url':f'http://{case}_original.jpg',
                                         'coordinates':[250,250,750,750]})}}
                for i in range(self.tool_count)]}
        elif stage.startswith('probe_'):
            candidate=int(stage.split('_')[-2])
            message={'role':'assistant','content':'<answer>yes</answer>' if candidate==1 else '<answer>no</answer>'}
        else:
            message={'role':'assistant','content':'<answer>B</answer>'}
        if validator and not validator(message):
            raise AssertionError('Fake output rejected by runner validator: '+stage)
        return message,dict(id='fake-'+stage,model='qwen/qwen3-vl-8b-instruct',cost=0,stage=stage)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.image=self.root/'source.jpg'
        Image.new('RGB',(512,512),'green').save(self.image)
        self.case={'key':'vstar_test','id':'0','question':'What color?\n(A) red\n(B) green\n(C) blue\n(D) white',
                   'image_path':str(self.image)}
        run.STOP.clear()

    def test_direct_one_call_and_terminal_cache(self):
        client=FakeClient()
        row=run.direct(client,self.root,self.case)
        repeated=run.direct(client,self.root,self.case)
        self.assertEqual(len(client.calls),1)
        self.assertEqual(client.calls[0]['stage'],'direct')
        self.assertEqual(row['prediction'],'B')
        self.assertEqual(row,repeated)
        self.assertEqual(row['crops'],0)

    def test_react_uses_requested_crop_without_probes_and_has_final_reader(self):
        client=FakeClient()
        baseline=run.direct(client,self.root,self.case)
        row=run.trajectory(client,self.root,self.case,'ReAct',baseline)
        calls=[c for c in client.calls if c['method']=='ReAct']
        self.assertEqual([c['stage'] for c in calls],['planner_1','planner_2','aggregator'])
        self.assertEqual(row['crops'],1)
        self.assertEqual(row['rounds'][0]['selected'],[250,250,750,750])
        self.assertEqual(row['prediction'],'B')

    def test_greedy_three_candidates_five_draws_then_commits_best(self):
        client=FakeClient()
        baseline=run.direct(client,self.root,self.case)
        row=run.trajectory(client,self.root,self.case,'FOVEA-Greedy',baseline)
        calls=[c for c in client.calls if c['method']=='FOVEA-Greedy']
        probes=[c for c in calls if c['stage'].startswith('probe_')]
        self.assertEqual(len(probes),15)
        self.assertEqual(len({c['stage'] for c in probes}),15)
        self.assertEqual(len(calls),18)
        self.assertEqual(row['rounds'][0]['scores'],[0,1,0])
        self.assertEqual(row['rounds'][0]['selected_index'],1)
        self.assertEqual(row['crops'],1)
        for candidate in range(3):
            self.assertEqual(sum(int(c['stage'].split('_')[-2])==candidate for c in probes),5)

    def test_multiple_tools_rejected_before_invalid_protocol_history(self):
        client=FakeClient(tool_count=2)
        message,_=client.call(self.case_messages(),'planner_1',self.case['key'],'ReAct')
        images={f"http://{self.case['key']}_original.jpg":object()}
        self.assertFalse(run.planner_valid(message,images,False))

    def case_messages(self):
        return [{'role':'user','content':self.case['question']}]

    def test_stop_prevents_network_dispatch(self):
        run.STOP.set()
        client=FakeClient()
        with self.assertRaises(run.BudgetStop): run.direct(client,self.root,self.case)
        self.assertEqual(client.calls,[])

    def test_planner_natural_language_stop_goes_to_final_reader_without_format_retry(self):
        class NaturalStopClient(FakeClient):
            def call(self,messages,stage,case,method,validator=None,**kwargs):
                if stage=='planner_1':
                    response={'role':'assistant','content':'The requested object appears green.'}
                    self.calls.append(dict(stage=stage,method=method,messages=copy.deepcopy(messages),kwargs=kwargs))
                    if validator and not validator(response):
                        raise AssertionError('Official natural-language planner stop rejected')
                    return response,dict(id='natural-stop',model='qwen/qwen3-vl-8b-instruct',cost=0,stage=stage)
                return super().call(messages,stage,case,method,validator=validator,**kwargs)
        client=NaturalStopClient()
        baseline=run.direct(client,self.root,self.case)
        row=run.trajectory(client,self.root,self.case,'ReAct',baseline)
        calls=[c for c in client.calls if c['method']=='ReAct']
        self.assertEqual([c['stage'] for c in calls],['planner_1','aggregator'])
        self.assertEqual(row['crops'],0)
        self.assertEqual(row['prediction'],'B')
        self.assertEqual(row['stop_reason'],'model_stop')

    def test_greedy_small_crop_is_tool_feedback_then_agent_reproposes(self):
        class SmallCropClient(FakeClient):
            def call(self,messages,stage,case,method,validator=None,**kwargs):
                if stage=='planner_1':
                    response={'role':'assistant','content':'Inspect a small region.','tool_calls':[
                        {'id':'small-tool','type':'function','function':{'name':'cropping',
                         'arguments':json.dumps({'image_url':f'http://{case}_original.jpg',
                                                 'coordinates':[100,100,120,120]})}}]}
                    self.calls.append(dict(stage=stage,method=method,messages=copy.deepcopy(messages),kwargs=kwargs))
                    if validator and not validator(response):
                        raise AssertionError('Small positive crop must reach official executor')
                    return response,dict(id='small-crop',model='qwen/qwen3-vl-8b-instruct',cost=0,stage=stage)
                return super().call(messages,stage,case,method,validator=validator,**kwargs)
        client=SmallCropClient()
        baseline=run.direct(client,self.root,self.case)
        row=run.trajectory(client,self.root,self.case,'FOVEA-Greedy',baseline)
        calls=[c for c in client.calls if c['method']=='FOVEA-Greedy']
        self.assertEqual([c['stage'] for c in calls],['planner_1','planner_2','aggregator'])
        feedback=[m for m in calls[1]['messages'] if m['role']=='tool']
        self.assertEqual(len(feedback),1)
        self.assertEqual(feedback[0]['tool_call_id'],'small-tool')
        self.assertEqual(json.loads(feedback[0]['content'])['status'],'error')
        self.assertIn('width < 50',json.loads(feedback[0]['content'])['message'])
        self.assertEqual(row['crops'],0)
        self.assertEqual(len(list((self.root/'tool_errors-v2').rglob('*.json'))),1)
        self.assertEqual(row['rounds'][0]['tool_error']['proposal'],[100,100,120,120])

    def test_all_runner_calls_force_primary_and_models_exclude_30b(self):
        client=FakeClient()
        baseline=run.direct(client,self.root,self.case)
        row=run.trajectory(client,self.root,self.case,'FOVEA-Greedy',baseline)
        self.assertTrue(client.calls)
        self.assertTrue(all(c['kwargs']['force_model']=='primary' for c in client.calls))
        self.assertEqual(row['models'],['qwen/qwen3-vl-8b-instruct'])
        config={'models':{'primary':{'id':'qwen/qwen3-vl-8b-instruct'},
                          'fallback':{'id':'qwen/qwen3-vl-30b-a3b-instruct'}}}
        with patch.object(run,'Client') as constructor:
            with self.assertRaisesRegex(ValueError,'restricted to the 8B'):
                run.make_client(self.root,config)
        constructor.assert_not_called()

    def test_only_pure_completed_8b_rows_imported_with_source_hash(self):
        history=self.root/'old-campaign'
        recovery=self.root/'recovery'
        oldrows={
            'pure':dict(case='pure',method='Direct',status='complete',
                        models=['qwen/qwen3-vl-8b-instruct'],prediction='A'),
            'mixed':dict(case='mixed',method='Direct',status='complete',
                         models=['qwen/qwen3-vl-8b-instruct','qwen/qwen3-vl-30b-a3b-instruct'],prediction='B'),
            'thirty':dict(case='thirty',method='Direct',status='complete',
                          models=['qwen/qwen3-vl-30b-a3b-instruct'],prediction='C'),
            'failed':dict(case='failed',method='Direct',status='request_failure',
                          models=['qwen/qwen3-vl-8b-instruct']),
        }
        for name,row in oldrows.items():
            run.save(history/'rows'/'Direct'/(name+'.json'),row)
        original={p: p.read_bytes() for p in (history/'rows'/'Direct').glob('*.json')}
        with patch.object(run,'HISTORICAL_ROOT',history):
            imported=run.import_pure_rows(recovery)
        self.assertEqual([x['case'] for x in imported['Direct']],['pure'])
        self.assertEqual(list((recovery/'rows'/'Direct').glob('*.json')),[recovery/'rows'/'Direct'/'pure.json'])
        reused=run.read(recovery/'rows'/'Direct'/'pure.json')
        self.assertEqual(reused['reused_from'],str(history/'rows'/'Direct'/'pure.json'))
        self.assertEqual(reused['source_row_sha256'],run.sha(history/'rows'/'Direct'/'pure.json'))
        self.assertEqual(reused['prediction'],'A')
        self.assertTrue(all(p.read_bytes()==data for p,data in original.items()))

    def test_same_model_format_retry_preserves_messages_and_forces_primary(self):
        class FormatRetryClient:
            def __init__(self): self.calls=[]
            def call(self,messages,**kwargs):
                self.calls.append((copy.deepcopy(messages),kwargs))
                if len(self.calls)==1:
                    raise run.RequestFailure('Requested backbone output validation failed: primary: OUTPUT_VALIDATION_FAILED')
                return {'role':'assistant','content':'<answer>B</answer>'},dict(
                    id='retry',model='qwen/qwen3-vl-8b-instruct',cost=0,stage=kwargs['stage'])
        client=FormatRetryClient()
        messages=self.case_messages()
        message,record=run.call(client,self.root,self.case,'Direct','direct',messages,
                               lambda m:run.answer(m) is not None)
        self.assertEqual(len(client.calls),2)
        self.assertEqual(client.calls[0][0],client.calls[1][0])
        self.assertEqual(client.calls[1][1]['stage'],'direct_format_retry1')
        self.assertEqual(client.calls[1][1]['seed'],(client.calls[0][1]['seed']+1)%(2**32))
        self.assertTrue(all(kwargs['force_model']=='primary' for _,kwargs in client.calls))
        self.assertEqual(record['format_retry_of'],'direct')

    def test_semantic_abstention_is_a_completed_direct_analysis_supplied_to_search(self):
        class AbstentionClient(FakeClient):
            def call(self,messages,stage,case,method,validator=None,**kwargs):
                response,record=super().call(messages,stage,case,method,validator=validator,**kwargs)
                if stage=='direct':
                    response={'role':'assistant','content':'<answer>There is no helmet visible.</answer>'}
                    self.assert_valid=bool(validator(response))
                    if not self.assert_valid: raise AssertionError('Semantic abstention rejected')
                return response,record
        client=AbstentionClient()
        baseline=run.direct(client,self.root,self.case)
        self.assertEqual(baseline['status'],'complete')
        self.assertIsNone(baseline['prediction'])
        self.assertEqual(baseline['grade'],'NO_ANSWER')
        row=run.trajectory(client,self.root,self.case,'ReAct',baseline)
        self.assertEqual(row['prediction'],'B')
        final=next(c for c in client.calls if c['stage']=='aggregator')
        self.assertTrue(any('There is no helmet visible.' in str(m['content']) for m in final['messages']))
        self.assertEqual(sum(c['stage'].startswith('direct') for c in client.calls),1)

    def test_semantic_abstention_from_final_reader_is_no_answer_without_retry(self):
        class AbstentionFinalClient(FakeClient):
            def call(self,messages,stage,case,method,validator=None,**kwargs):
                response,record=super().call(messages,stage,case,method,validator=validator,**kwargs)
                if stage=='aggregator':
                    response={'role':'assistant','content':'<answer>None of the above</answer>'}
                    if not validator(response): raise AssertionError('Final abstention rejected')
                return response,record
        client=AbstentionFinalClient()
        baseline=run.direct(client,self.root,self.case)
        row=run.trajectory(client,self.root,self.case,'ReAct',baseline)
        self.assertIsNone(row['prediction'])
        self.assertEqual(row['grade'],'NO_ANSWER')
        self.assertEqual(row['status'],'complete')
        self.assertEqual(sum(c['stage'].startswith('aggregator') for c in client.calls),1)

    def _paid_attempt(self,method,stage,content='None of the above',finish_reason='stop'):
        database=self.root/'ledger.sqlite'
        response=self.root/(method+'-'+stage+'.response.json')
        run.save(response,{'choices':[{'finish_reason':finish_reason,'message':
            {'role':'assistant','content':content}}],'usage':{'cost':.001}})
        with sqlite3.connect(database) as connection:
            connection.execute('''CREATE TABLE IF NOT EXISTS attempts
                (id TEXT,case_id TEXT,method TEXT,model TEXT,role TEXT,stage TEXT,
                 started REAL,finished REAL,status TEXT,cost REAL,response_path TEXT)''')
            connection.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (method+'-'+stage,self.case['key'],method,'qwen/qwen3-vl-8b-instruct','primary',
                 stage,1,2,'invalid',.001,str(response)))
        return database,response

    def test_paid_direct_failure_recovered_into_overlay_without_changing_original_or_ledger(self):
        original=run.result_path(self.root,'Direct',self.case)
        run.save(original,dict(case=self.case['key'],method='Direct',status='request_failure',error='OUTPUT_VALIDATION_FAILED'))
        database,response=self._paid_attempt('Direct','direct_format_retry1')
        before={p:p.read_bytes() for p in (original,database,response)}
        recovered=run.recover_direct_failures(self.root)
        self.assertEqual(len(recovered),1)
        self.assertEqual(recovered[0]['grade'],'NO_ANSWER')
        overlay=run.effective_row(self.root,'Direct',self.case)
        self.assertEqual(overlay['status'],'complete')
        self.assertIsNone(overlay['prediction'])
        self.assertEqual(overlay['source_response'],str(response))
        self.assertEqual(overlay['source_result_sha256'],run.sha(original))
        self.assertTrue(all(p.read_bytes()==data for p,data in before.items()))
        client=FakeClient()
        self.assertEqual(run.direct(client,self.root,self.case),overlay)
        self.assertEqual(client.calls,[])
        self.assertEqual(run.recover_direct_failures(self.root),[])

    def test_search_recovery_requires_billed_final_not_planner_or_truncated_content(self):
        original=run.result_path(self.root,'ReAct',self.case)
        run.save(original,dict(case=self.case['key'],method='ReAct',status='request_failure',error='OUTPUT_VALIDATION_FAILED'))
        self._paid_attempt('ReAct','planner_2')
        self._paid_attempt('ReAct','aggregator',finish_reason='length')
        self.assertEqual(run.recover_failed_results(self.root),[])
        self.assertEqual(run.effective_row(self.root,'ReAct',self.case)['status'],'request_failure')
        self._paid_attempt('ReAct','aggregator_format_retry1')
        recovered=run.recover_failed_results(self.root)
        self.assertEqual(len(recovered),1)
        self.assertEqual(recovered[0]['method'],'ReAct')
        self.assertEqual(recovered[0]['grade'],'NO_ANSWER')
        self.assertEqual(run.read(original)['status'],'request_failure')

    def test_no_answer_counts_zero_and_overlay_does_not_double_count_original_failure(self):
        original=run.result_path(self.root,'Direct',self.case)
        run.save(original,dict(case=self.case['key'],method='Direct',status='request_failure'))
        self._paid_attempt('Direct','direct')
        run.recover_direct_failures(self.root)
        prep=self.root/'prepared'
        run.save(prep/'labels.evaluation-only.json',{self.case['key']:'B'})
        class ReportingClient:
            models={'primary':{'id':'qwen/qwen3-vl-8b-instruct'}}
            def stats(self): return {'confirmed_usd':.001}
        with patch.object(run,'PREP',prep):
            summary=run.summarize(self.root,ReportingClient())['methods']['Direct']
        self.assertEqual(summary['completed'],1)
        self.assertEqual(summary['correct'],0)
        self.assertEqual(summary['no_answer'],1)
        self.assertEqual(summary['errors'],0)
        self.assertEqual(summary['pending'],190)


if __name__=='__main__': unittest.main()
