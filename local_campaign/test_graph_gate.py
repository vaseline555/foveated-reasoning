"""No-network tests of Direct isolation, citation validity and persistence order."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO=Path(__file__).resolve().parents[1]
# backend's production path loader is relocated by the test runner to a temp workspace.
import backend
from graph_gate import GraphGateBackend

class FakeClient:
    def __init__(self,out):self.calls=[];self.out=out
    def call(self,messages,stage,validator,**kwargs):
        self.calls.append((stage,copy.deepcopy(messages)))
        if stage=='graph_gate':
            assert 'DIRECT_CANARY_SECRET' not in json.dumps(messages)
            value=dict(answer='A',rationale='Graph observation shows red.',
                       evidence=[dict(role_id='color',image_key='original',observation_index=0)],uncertainties=[])
            message=dict(role='assistant',content=None,tool_calls=[dict(function=dict(name='submit_graph_gate',arguments=json.dumps(value)))])
        else:
            assert (self.out/'gate.json').exists(), 'Gate not persisted before aggregator'
            assert 'DIRECT_CANARY_SECRET' in json.dumps(messages)
            message=dict(role='assistant',content='Final fusion chooses blue. <answer>B</answer>')
        assert validator(message)
        return message,dict(stage=stage,model='qwen/qwen3-vl-8b-instruct',cost=0)

class GateTests(unittest.TestCase):
    def setup_backend(self,root):
        from PIL import Image
        image=root/'image.jpg';Image.new('RGB',(32,32),'red').save(image)
        out=root/'case';client=FakeClient(out)
        b=GraphGateBackend(dict(key='x',question='What color?\n(A) red\n(B) blue',image_path=str(image)),root,out,client,
                           dict(status='complete',models=['qwen/qwen3-vl-8b-instruct'],message=dict(content='DIRECT_CANARY_SECRET <answer>B</answer>')))
        a=backend.adaptive
        plan=a.Plan((a.Role('color','color','visible color'),),('color',),
                    (a.Claim('color','red','supported'),))
        state=a.CaseState('What color?','original',plan)
        state.image_keys.add('original');a._set_plan(state,plan)
        a._record(state,a.Observation(None,'original',a.Report(plan.original_claims)),a.Limits())
        return b,state,client

    def test_independent_gate_then_fusion_and_both_outputs_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            b,state,client=self.setup_backend(Path(tmp))
            result=asyncio.run(b.finalize(state))
            self.assertEqual([s for s,_ in client.calls],['graph_gate','aggregator'])
            self.assertEqual(b.gate_result['answer'],'A')
            self.assertIn('<answer>B</answer>',result)
            self.assertEqual(json.loads((b.outputdir/'gate.json').read_text())['answer'],'A')
            self.assertTrue((b.outputdir/'aggregator.json').exists())

    def test_common_reader_modes_exclude_gate_and_control_direct(self):
        with tempfile.TemporaryDirectory() as tmp:
            b,state,_=self.setup_backend(Path(tmp))
            for mode in ('common_direct', 'common_no_direct', 'legacy_fusion'):
                b.reader_mode=mode
                text=json.dumps(b.reader_messages(state, {'rationale':'GATE_CANARY'}))
                self.assertEqual('DIRECT_CANARY_SECRET' in text, mode != 'common_no_direct')
                self.assertEqual('GATE_CANARY' in text, mode == 'legacy_fusion')
            b.reader_mode='invalid'
            with self.assertRaises(ValueError): b.reader_messages(state, {})

    def test_rejects_fabricated_citations_and_absent_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            b,state,_=self.setup_backend(Path(tmp))
            good=dict(answer='A',rationale='red',evidence=[dict(role_id='color',image_key='original',observation_index=0)],uncertainties=[])
            for change in ({'answer':'C'},{'evidence':[]},{'evidence':[dict(role_id='color',image_key='invented',observation_index=0)]},
                           {'evidence':[dict(role_id='invented',image_key='original',observation_index=0)]},
                           {'evidence':[dict(role_id='color',image_key='original',observation_index=99)]}):
                with self.subTest(change=change),self.assertRaises(ValueError):b.validate_gate(good|change,state)
            self.assertEqual(b.validate_gate(good|dict(answer='NO_ANSWER',evidence=[]),state)['answer'],'NO_ANSWER')

if __name__=='__main__':unittest.main()
