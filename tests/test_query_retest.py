"""Bounded blind auditing: adversarial status, mapping, source, and format cases."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from active_vlm.decision.retest import QueryRetest
from active_vlm.decision.partition import PartitionDAG
from active_vlm.decision.partition_prompts import bind_observations


def fixture(probability=1, modification=None, malformed=False):
    dag = PartitionDAG(['A','B'])
    dag.accept([dict(task='Which color is the marked object?',target='the marked object',
                     kind='measure',requires=[],box=[100,100,500,500])], lambda _:None)
    fact = dict(target_visible=True,identity_evidence='marked object',observation='red',
                source='crop',resolved=True,reason='visible surface')
    added = dag.observe_batch(bind_observations({'g0':fact},dag.nodes,'A',dag.letters),[0],'crop','first')
    calls = []
    def tool(*args,**kwargs):
        calls.append((args,kwargs))
        if malformed:
            raise ValueError('TRUNCATED_TOOL_OUTPUT')
        return dict(observations={'g0':dict(fact,**(modification or {}))},answer='B'), {'receipt_id':'audit'}
    run = SimpleNamespace(dag=dag,case={'allowed_letters':['A','B'],'original_size':[1000,1000]},
        question='Which color is the marked object?\n(A) Red\n(B) Blue',turn=1,repairs=0,
        validate_pixels=lambda _:None,tool=tool,rejections=[])
    crop = dict(id='crop',normalized=[100,100,500,500],original_xyxy=[100,100,500,500])
    return QueryRetest(dict(seed=42,retest_probability=probability),'case'),run,crop,added,calls


def test_conflict_does_not_overwrite_original_or_create_majority_truth():
    audit,run,crop,added,calls=fixture()
    old=deepcopy(run.dag.__dict__)
    out=audit.consider(run,crop,added)
    assert out['choice_relation']=='different' and not out['semantic_truth_verified']
    assert run.dag.__dict__==old
    assert audit.consider(run,crop,added) is None and len(calls)==1
    packet=calls[0][0][2]
    assert 'prior_observations' not in packet and 'native_relevance_report' not in packet
    assert calls[0][1]['include_original'] is False


@pytest.mark.parametrize('change',[{'source':'original'},{'source':'inference'},
                                  {'resolved':False},{'target_visible':False},
                                  {'identity_evidence':''},{'observation':''}])
def test_unsupported_audit_never_exports_positive_fact_or_answer(change):
    audit,run,crop,added,_=fixture(modification=change)
    out=audit.consider(run,crop,added)
    assert out['report']['fact'] is out['report']['identity'] is out['report']['answer_hypothesis'] is None
    assert not out['report']['observed']
    assert run.dag.possible=={'A'}


def test_malformed_paid_retest_is_consumed_once_without_model_retry():
    audit,run,crop,added,calls=fixture(malformed=True)
    out=audit.consider(run,crop,added)
    assert out['report'] is None and 'TRUNCATED' in out['format_error']
    assert audit.consider(run,crop,added) is None and len(calls)==1
    assert run.dag.possible=={'A'}


def test_missing_mapping_is_not_automatic_visual_failure():
    audit,run,crop,added,calls=fixture()
    oldtool=run.tool
    def unmapped(*args,**kwargs):
        payload,record=oldtool(*args,**kwargs);payload['answer']=None
        return payload,record
    run.tool=unmapped
    out=audit.consider(run,crop,added)
    assert out['report']['observed'] and out['report']['fact']=='red'
    assert out['report']['answer_hypothesis'] is None
    assert out['choice_relation']=='unmapped'


def test_unresolved_query_never_triggers_reassurance_and_control_never_dispatches():
    audit,run,crop,added,calls=fixture(0)
    added[0]['observed']=False
    assert audit.consider(run,crop,added) is None and audit.opportunity is None
    added[0]['observed']=True
    assert audit.consider(run,crop,added) is None and not calls
    assert audit.opportunity['selected'] is False


@pytest.mark.parametrize('p',[-.01,1.01,True,float('nan'),float('inf'),'0.5'])
def test_invalid_audit_probability_stops_before_dispatch(p):
    with pytest.raises(ValueError,match='PROBABILITY'):
        QueryRetest(dict(seed=1,retest_probability=p),'case')
