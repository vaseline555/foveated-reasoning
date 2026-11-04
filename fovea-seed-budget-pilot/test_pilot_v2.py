"""Offline pilot invariants; no inference or dataset labels leave this process."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import pilot_v2 as pilot


MODELS={role:dict(id='qwen/qwen3-vl-8b-instruct',prompt_price=.25,
    completion_price=.75,context_length=262144) for role in ('primary','fallback')}


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'output';self.output.mkdir()
        image=Image.new('RGB',(512,512),'green')
        for x in range(256):
            for y in range(512):image.putpixel((x,y),(x%256,y%256,20))
        path=self.root/'image.jpg';image.save(path)
        self.case=dict(key='vstar_test',id='0',category='relative_position',question='Where is the cup relative to the chair?\n(A) left\n(B) right',
            image_path=str(path),image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),max_seeds=2,required_roles=['cup','chair'])
        self.direct=dict(case='vstar_test',status='complete',models=['qwen/qwen3-vl-8b-instruct'],
                         message=dict(role='assistant',content='<answer>A</answer>'),records=[dict(id='direct',cost=.001)])
        self.calls=[]

    def fake_call(self,client,shared_root,case,stage,messages,validator,tools=None,seed=None):
        self.calls.append(dict(stage=stage,messages=copy.deepcopy(messages),seed=seed))
        if '_proposal' in stage:
            index=int(stage.split('_')[1]);box=[100,100,400,400] if index==0 else [600,600,900,900]
            message=dict(role='assistant',content='PRIVATE_SEED_THOUGHT_'+str(index),tool_calls=[dict(
                id='proposal-'+str(index),type='function',function=dict(name='cropping',arguments=json.dumps(dict(
                    image_url='http://vstar_test_original.jpg',coordinates=box))))])
        elif '_probe_' in stage:
            fields=stage.split('_');yes=int(fields[1])==1 and int(fields[3])==0
            message=dict(role='assistant',content='<answer>yes</answer>' if yes else '<answer>no</answer>')
        else:message=dict(role='assistant',content='<answer>B</answer>')
        if validator and not validator(message):raise AssertionError('Fake message invalid '+stage)
        return message,dict(id='record-'+stage,model='qwen/qwen3-vl-8b-instruct',cost=.0001,stage=stage)

    def execute(self,caller=None):
        with patch.object(pilot,'_call',side_effect=caller or self.fake_call),patch.object(pilot,'stage_cost',return_value=.02),patch.object(pilot.baseline,'effective_row',return_value=self.direct):
            return pilot.execute_case(object(),self.root,self.output,self.case)

    def test_prefix_pool_and_official_tie_break(self):
        scored=[dict(seed_index=0,score=.5,name='first'),dict(seed_index=0,score=.5,name='second'),dict(seed_index=1,score=1,name='other')]
        self.assertEqual(pilot.choose_candidate(scored,1)['name'],'first')
        self.assertEqual(pilot.choose_candidate(scored,2)['name'],'other')

    def test_seed1_is_blind_to_condition_and_seed2_has_only_box_hint(self):
        a=copy.deepcopy(self.case);a['max_seeds']=1
        b=copy.deepcopy(self.case);b['max_seeds']=2
        self.assertEqual(pilot.proposal_messages(a,'image','original',[]),pilot.proposal_messages(b,'image','original',[]))
        text=json.dumps(pilot.proposal_messages(a,'image','original',[[0,0,100,100]]))
        self.assertIn('Previously proposed region',text)
        self.assertNotIn('PRIVATE',text)

    def test_two_seeds_exact_native_probes_and_single_crop_readout(self):
        row=self.execute()
        self.assertEqual(row['status'],'complete')
        probes=[c for c in self.calls if '_probe_' in c['stage']]
        self.assertEqual(len(probes),30)
        self.assertEqual(row['conditions']['1']['selected_seed'],0)
        self.assertEqual(row['conditions']['2']['selected_seed'],1)
        final=[c for c in self.calls if c['stage'].startswith('final_')]
        self.assertEqual(len(final),2)
        self.assertEqual(final[0]['seed'],final[1]['seed'])
        order=[c['stage'] for c in self.calls]
        self.assertLess(order.index(final[0]['stage']),order.index('seed_1_proposal'))
        for item in final:
            wire=json.dumps(item['messages'])
            self.assertNotIn('PRIVATE_SEED_THOUGHT',wire)
            self.assertNotIn('Previously proposed region',wire)
            images=[b for m in item['messages'] for b in (m.get('content') if isinstance(m.get('content'),list) else []) if b.get('type')=='image_url']
            self.assertEqual(len(images),2) # original + exactly one selected crop

    def test_second_seed_failure_retains_k1_result(self):
        def caller(*args,**kwargs):
            stage=args[3]
            if stage=='seed_1_proposal':raise pilot.RequestFailure('simulated seed2 failure')
            return self.fake_call(*args,**kwargs)
        with self.assertRaises(pilot.RequestFailure):self.execute(caller)
        row=pilot.read(self.output/'rows'/'vstar_test.json')
        self.assertEqual(row['status'],'partial')
        self.assertIn('1',row['conditions']);self.assertNotIn('2',row['conditions'])
        self.assertEqual(row['attempted_seed_budgets'],[1,2])

    def test_inference_never_reads_labels(self):
        original_read=pilot.read
        def guarded(path):
            if Path(path).name=='labels.evaluation-only.json':raise AssertionError('Label leakage')
            return original_read(path)
        with patch.object(pilot,'read',side_effect=guarded):self.execute()

    def test_pilot_reservation_cap_and_historical_campaign_cap(self):
        client=pilot.PilotClient(self.root/'ledger',MODELS,cap=15,historical_usd=.92410405,min_request_interval=0,pilot_cap=1.5)
        with self.assertRaises(pilot.BudgetStop):
            client._reserve('key',MODELS['primary'],'primary','case',pilot.METHOD,'stage',1.6,self.root/'request')
        self.assertEqual(client.stats()['attempts'],0)
        other=pilot.PilotClient(self.root/'other',MODELS,cap=15,historical_usd=14.9,min_request_interval=0,pilot_cap=1.5)
        with self.assertRaises(pilot.BudgetStop):
            other._reserve('key',MODELS['primary'],'primary','case',pilot.METHOD,'stage',.2,self.root/'request')
        self.assertEqual(other.stats()['attempts'],0)

    def test_empty_and_partial_report_remain_defined(self):
        plan=dict(cohort=[self.case])
        with patch.object(pilot,'read',side_effect=lambda path:{} if Path(path).name=='labels.evaluation-only.json' else json.loads(Path(path).read_text())):
            summary=pilot.report(self.output,plan)
        self.assertEqual(summary['completed_n'],0)
        self.assertEqual(summary['question_independent_exact_allocation']['allocations'],0)
        self.assertTrue((self.output/'report.json').exists())
        self.assertEqual(summary['arm_coverage']['1']['missing_result_n'],1)

    def test_exact_equal_budget_allocations_are_924_and_role_rule_visible(self):
        cases=[];labels={}
        for i in range(12):
            key='case_'+str(i);cases.append(dict(key=key));labels[key]='A'
            pilot.save(self.output/'rows'/(key+'.json'),dict(case=key,status='complete',category='attribute' if i<6 else 'relation',
                question_budget=1 if i<6 else 2,attempted_seed_budgets=[1,2],seed_iou=.1,second_seed_selected=False,readout_physically_shared=True,
                conditions={str(k):dict(prediction='A',logical_cost_usd=k*.01) for k in (1,2)}))
        original_read=pilot.read
        with patch.object(pilot,'read',side_effect=lambda path:labels if Path(path).name=='labels.evaluation-only.json' else original_read(path)):
            summary=pilot.report(self.output,dict(cohort=cases))
        exact=summary['question_independent_exact_allocation']
        self.assertEqual(exact['allocations'],924)
        self.assertEqual(exact['total_seed_budget'],18)
        self.assertEqual(exact['fraction_equal_q'],1)
        self.assertEqual(exact['question_midrank_percentile'],50)

    def test_small_box_is_valid_tool_input_then_feedback_reproposal(self):
        def caller(*args,**kwargs):
            stage=args[3]
            if stage=='seed_0_proposal':
                message=dict(role='assistant',content='Inspect small detail.',tool_calls=[dict(
                    id='small',type='function',function=dict(name='cropping',arguments=json.dumps(dict(
                        image_url='http://vstar_test_original.jpg',coordinates=[100,100,120,125]))))])
                self.assertTrue(args[5](message))
                self.calls.append(dict(stage=stage,messages=copy.deepcopy(args[4]),seed=kwargs.get('seed')))
                return message,dict(id='small-receipt',model='qwen/qwen3-vl-8b-instruct',cost=.0001,stage=stage)
            return self.fake_call(*args,**kwargs)
        row=self.execute(caller)
        self.assertEqual(row['status'],'complete')
        self.assertEqual(len([x for x in self.calls if '_probe_' in x['stage']]),30)
        repair=next(x for x in self.calls if x['stage']=='seed_0_proposal_minsize_retry1')
        tools=[x for x in repair['messages'] if x.get('role')=='tool']
        self.assertEqual(len(tools),1)
        self.assertEqual(json.loads(tools[0]['content'])['status'],'error')
        self.assertIn('width < 50',json.loads(tools[0]['content'])['message'])
        pool=pilot.read(self.output/'pools'/'vstar_test'/'seed_0.json')
        self.assertEqual(pool['proposal_stages'],['seed_0_proposal','seed_0_proposal_minsize_retry1'])
        self.assertEqual(len(pool['proposal_rejections']),1)
        self.assertIn('seed_0_proposal_minsize_retry1',row['conditions']['1']['logical_stage_names'])

    def test_small_second_seed_feedback_limit_retains_first_arm(self):
        def caller(*args,**kwargs):
            stage=args[3]
            if stage.startswith('seed_1_proposal'):
                self.calls.append(dict(stage=stage,messages=copy.deepcopy(args[4]),seed=kwargs.get('seed')))
                message=dict(role='assistant',content='Small target.',tool_calls=[dict(
                    id='small-'+stage,type='function',function=dict(name='cropping',arguments=json.dumps(dict(
                        image_url='http://vstar_test_original.jpg',coordinates=[600,600,620,625]))))])
                self.assertTrue(args[5](message))
                return message,dict(id=stage,model='qwen/qwen3-vl-8b-instruct',cost=.0001,stage=stage)
            return self.fake_call(*args,**kwargs)
        with self.assertRaisesRegex(pilot.RequestFailure,'after 3 proposals'):
            self.execute(caller)
        row=pilot.read(self.output/'rows'/'vstar_test.json')
        self.assertIn('1',row['conditions']);self.assertNotIn('2',row['conditions'])
        attempts=[x for x in self.calls if x['stage'].startswith('seed_1_proposal')]
        self.assertEqual(len(attempts),3)
        evidence=pilot.read(self.output/'proposal_rejections'/'vstar_test'/'seed_1.json')
        self.assertEqual(len(evidence['rejections']),3)
        self.assertEqual(len([x for x in self.calls if '_probe_' in x['stage']]),15)

    def test_task_cap_includes_prior_pilot_version(self):
        client=pilot.PilotClient(self.root/'ledger',MODELS,cap=15,historical_usd=.92410405,min_request_interval=0,pilot_cap=1.5)
        with client.connect() as connection:
            connection.execute("INSERT INTO attempts (id,logical_key,model,role,case_id,method,stage,started,status,reserved,cost) VALUES ('old','old','8b','primary','c','SEED_BUDGET_PILOT_V1','seed',1,'valid',1.49,1.49)")
        with self.assertRaises(pilot.BudgetStop):
            client._reserve('new',MODELS['primary'],'primary','c',pilot.METHOD,'seed',.02,self.root/'request')
        self.assertEqual(client.stats()['attempts'],1)


if __name__=='__main__':unittest.main()
