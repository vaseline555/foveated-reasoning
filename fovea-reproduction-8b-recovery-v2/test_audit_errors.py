"""Offline audit tests: real SQLite fixtures, never model calls."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from audit_errors import classify, snapshot, save


class ErrorAuditTests(unittest.TestCase):
    def test_classifies_observed_tool_contract_and_truncation(self):
        attempt={'stage':'planner_1','status':'invalid','method':'FOVEA-Greedy'}
        request={'messages':[{'content':'Orignal image URL: http://vstar_0_original.jpg'}]}
        def response(args):
            return {'choices':[{'message':{'tool_calls':[{'function':dict(name='cropping',arguments=args)}]}}]}
        self.assertEqual(classify(attempt,request,response('{'))['category'],'tool_arguments_json')
        args=json.dumps(dict(image_url='http://vstar_0_original.jpg',coordinates=[0,0,10,10]))
        self.assertEqual(classify(attempt,request,response(args))['category'],'crop_coordinates')
        args=json.dumps(dict(image_url='http://imaginary.jpg',coordinates=[0,0,100,100]))
        self.assertEqual(classify(attempt,request,response(args))['category'],'tool_image_reference')
        self.assertEqual(classify(attempt,request,{'choices':[{'finish_reason':'length'}]})['category'],'output_truncated')

    def test_api_probe_and_unknown_accounting_are_separate(self):
        self.assertEqual(classify({'stage':'probe_1','status':'invalid'},None,
            {'choices':[{'message':{'content':'Cannot determine'}}]})['category'],'probe_answer_format')
        self.assertEqual(classify({'status':'rejected','error':'HTTP 429'},None,None)['category'],'api_rate_limit')
        self.assertEqual(classify({'status':'unknown','error':None},None,
            {'choices':[{'message':{'content':'yes'}}]})['category'],'accounting_cost_unknown')

    def test_snapshot_counts_physical_errors_and_terminal_runs_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            request=root/'request.json';response=root/'invalid.json'
            save(request,{'messages':[]})
            save(response,{'choices':[{'message':{'content':''}}]})
            db=sqlite3.connect(root/'ledger.sqlite')
            db.execute('''CREATE TABLE attempts(id TEXT,logical_key TEXT,case_id TEXT,method TEXT,
                stage TEXT,model TEXT,role TEXT,started REAL,status TEXT,cost REAL,reserved REAL,
                response_path TEXT,request_path TEXT,error TEXT)''')
            rows=[('bad','logical','c1','Direct','direct','8b','primary',1,'invalid',.01,.1,str(response),str(request),'OUTPUT_VALIDATION_FAILED'),
                  ('good','logical','c1','Direct','direct','30b','fallback',2,'valid',.02,.1,None,None,None),
                  ('active','other','c2','Direct','direct','8b','primary',3,'dispatched',None,.1,None,None,None),
                  ('valid_wrong','wrong','c3','Direct','direct','8b','primary',4,'valid',.01,.1,None,None,None),
                  ('unknown','unknown','c4','ReAct','planner_1','8b','primary',5,'unknown',None,.1,None,None,'TimeoutError')]
            db.executemany('INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',rows)
            db.commit();db.close()
            save(root/'state.json',{'status':'running','pid':1})
            save(root/'rows'/'Direct'/'c1.json',{'case':'c1','method':'Direct','status':'complete'})
            save(root/'rows'/'ReAct'/'c4.json',{'case':'c4','method':'ReAct','status':'request_failure','error':'bad crop'})
            before=(root/'ledger.sqlite').read_bytes()
            report=snapshot(root)
            self.assertEqual(report['summary']['request_error_events'],2)
            self.assertEqual(report['summary']['recovered_via_30b_events'],1)
            self.assertEqual(report['summary']['failed_case_methods'],1)
            self.assertEqual(report['summary']['active_requests'],1)
            self.assertAlmostEqual(report['summary']['unresolved_error_hold_usd'],.1)
            self.assertEqual((root/'ledger.sqlite').read_bytes(),before)
            self.assertEqual(report['request_events'][0]['recovery']['attempt_id'],'good')
            self.assertTrue((root/'harness_errors.json').exists())

    def test_stop_reason_retained_without_counting_it_as_request_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            save(root/'state.json',{'status':'stopped'})
            save(root/'fatal_error-v2.json',{'type':'ValueError','message':'Selected crop is empty'})
            save(root/'stop_reason-v2.json',{'type':'BudgetStop','message':'cap reached'})
            report=snapshot(root)
            self.assertEqual(report['summary']['request_error_events'],0)
            self.assertEqual({e['category'] for e in report['runtime_events']},
                             {'crop_or_image_runtime','budget_or_user_stop'})

    def test_semantic_abstention_is_not_a_harness_fault(self):
        event=classify({'stage':'direct','status':'invalid'},None,
            {'choices':[{'message':{'content':'<answer>None of the above</answer>'}}]})
        self.assertEqual(event['category'],'model_no_valid_choice')
        self.assertEqual(event['origin'],'model_answer_quality_not_harness')

    def test_overlay_reclassifies_result_without_changing_original_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            original=root/'rows'/'Direct'/'c1.json'
            save(original,dict(case='c1',method='Direct',status='request_failure',error='old validation'))
            before=original.read_bytes()
            save(root/'result_overlays'/'Direct'/'c1.json',dict(case='c1',method='Direct',
                status='complete',prediction=None,grade='NO_ANSWER'))
            report=snapshot(root)
            self.assertEqual(report['summary']['failed_case_methods'],0)
            self.assertEqual(report['summary']['no_answer_results'],1)
            self.assertEqual(original.read_bytes(),before)


if __name__=='__main__':
    unittest.main()
