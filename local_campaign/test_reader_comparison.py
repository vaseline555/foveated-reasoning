import json
from pathlib import Path
import tempfile
import unittest
from reader_comparison import extract_planner, summarize_cases, build


class ComparisonTests(unittest.TestCase):
    def test_correctness_transitions_use_identical_paired_set(self):
        cases=[]
        for pc,rc in ((True,True),(True,False),(False,True),(False,False)):
            cases.append(dict(planner=dict(available=True,prediction='A'),planner_correct=pc,reader_correct=rc))
        cases.append(dict(planner=dict(available=False,reason='missing'),planner_correct=None,reader_correct=True))
        v=summarize_cases(cases)
        self.assertEqual(v['paired_count'],4)
        self.assertEqual(v['planner_accuracy'],.5)
        self.assertEqual(v['paired_reader_accuracy'],.5)
        self.assertEqual(v['reader_accuracy'],.6)
        for name in ('correct_to_correct','correct_to_wrong','wrong_to_correct','wrong_to_wrong'):
            self.assertEqual(v[name],1)
        self.assertEqual(v['unavailable_count'],1)

    def test_uses_last_valid_planner_including_format_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=root/'response.json';p.write_text(json.dumps({'choices':[{'message':{'content':'<answer>(B)</answer>'}}]}))
            row=dict(case='x',records=[dict(stage='planner_1',response_path='/missing'),
                dict(stage='planner_2_format_retry1',format_retry_of='planner_2',response_path=str(p))])
            self.assertEqual(extract_planner(root,'Greedy',row)['prediction'],'B')
            p.write_text(json.dumps({'choices':[{'message':{'content':'<answer>B</answer>','tool_calls':[{}]}}]}))
            self.assertFalse(extract_planner(root,'Greedy',row)['available'])

    def test_graph_evidence_is_not_invented_as_option_and_noanswer_is_incorrect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'response.json'
            p.write_text(json.dumps({'choices':[{'message':{'content':'Cannot determine.'}}]}))
            rows={'Graph':[dict(case='x',status='complete',prediction='A')],
                  'ReAct':[dict(case='x',status='complete',prediction='A',records=[dict(stage='planner_1',response_path=str(p))])]}
            v=build(root,rows,{'x':'A'})['methods']
            self.assertEqual(v['Graph']['paired_count'],0)
            self.assertIsNone(v['Graph']['planner_accuracy'])
            self.assertEqual(v['ReAct']['planner_NO_ANSWER'],1)
            self.assertEqual(v['ReAct']['wrong_to_correct'],1)

    def test_explicit_graph_gate_has_correct_to_wrong_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            out=root/'graph/cases/x';out.mkdir(parents=True)
            (out/'gate.json').write_text(json.dumps(dict(answer='A')))
            audit=build(root,{'Graph':[dict(case='x',status='complete',prediction='B',gate_prediction='A')]},{'x':'A'})
            g=audit['methods']['Graph']
            self.assertEqual(g['planner_accuracy'],1)
            self.assertEqual(g['paired_reader_accuracy'],0)
            self.assertEqual(g['correct_to_wrong'],1)
            self.assertEqual(g['gate_accuracy_all'],1)
            self.assertEqual(g['gate_generated'],1)


if __name__=='__main__':unittest.main()
