"""Graph-only categorical judgment, followed by independent Direct fusion."""
from pathlib import Path
import json
from backend import adaptive, official, Backend, _save_new, serializable, validate_schema, _json

GATE_SCHEMA={
 'type':'object','required':['answer','rationale','evidence','uncertainties'],
 'properties':{
  'answer':{'type':'string','enum':['A','B','C','D','NO_ANSWER']},
  'rationale':{'type':'string','minLength':1,'maxLength':2000},
  'evidence':{'type':'array','maxItems':12,'items':{
   'type':'object','required':['role_id','image_key','observation_index'],
   'properties':{'role_id':{'type':'string'},'image_key':{'type':'string'},
                 'observation_index':{'type':'integer','minimum':0}},'additionalProperties':False}},
  'uncertainties':{'type':'array','maxItems':8,'items':{'type':'string'}}},
 'additionalProperties':False}
GATE_SYSTEM='You are the independent Graph evidence judge. Return one submit_graph_gate tool call or equivalent JSON. You have no Direct analysis. Evidence statuses are model claims, not verified truth.'
GATE_INSTRUCTION='Using ONLY the question, original image, recorded Graph observations/crops and evidence state, select the most supported option. Return answer A/B/C/D or NO_ANSWER, a concise rationale, evidence citations, and uncertainties. Each citation must name a role_id, image_key and zero-based observation_index present together in the recorded observations. A categorical answer requires at least one evidence citation. Never invent observations or infer truth from an expected_evidence task. If support is insufficient, use NO_ANSWER. Do not request more crops.'
FUSION_SYSTEM='You are the final evidence judge. Compare the independent Graph gate verdict with the independent Direct analysis. Both are model judgments and may be wrong. Inspect their supporting observations and images; do not prefer either solely because it sounds confident. Give a concise justification, then exactly one <answer>A/B/C/D</answer> or <answer>NO_ANSWER</answer>. Do not call tools.'
FUSION_INSTRUCTION='Synthesize the Graph gate verdict and Direct analysis supplied below, with the recorded visual evidence. Resolve any disagreement using evidence and give your final answer. No new crop observations are available.'


def store(path,value):
    path=Path(path);value=serializable(value)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Conflicting immutable gate/final artifact')
    else:_save_new(path,value)


class GraphGateBackend(Backend):
    # Explicit ablations share the same exploration snapshot and validator.
    reader_mode = "common_direct"

    def reader_messages(self, state, gate):
        if self.reader_mode == "legacy_fusion":
            return self.fusion_messages(state, gate)
        if self.reader_mode not in {"common_direct", "common_no_direct"}:
            raise ValueError("Unknown Graph reader mode")
        direct = (self.direct_baseline['message'].get('content') or '') if self.reader_mode == "common_direct" else ''
        return official.aggregator_messages(self.case['question'], self.image_urls['original'],
                                            self._aggregator_history(state), direct)

    def gate_messages(self,state):
        # This builder intentionally does not access self.direct_baseline.
        messages=self._messages(GATE_INSTRUCTION,state)
        messages[0]={'role':'system','content':GATE_SYSTEM}
        return messages

    def validate_gate(self,value,state):
        validate_schema(value,GATE_SCHEMA)
        import re
        options=re.findall(r'^\(([A-D])\)',self.case['question'],re.M) or list('ABCD')
        if value['answer'] not in options+['NO_ANSWER']:
            raise ValueError('Gate option not present in this question')
        if value['answer']!='NO_ANSWER' and not value['evidence']:
            raise ValueError('A categorical gate answer requires recorded evidence')
        for ref in value['evidence']:
            i=ref['observation_index']
            current=state.evidence.get(ref['role_id'])
            if current is None or current['observation_index'] != i:
                raise ValueError('Citation references invalidated or superseded evidence')
            if i>=len(state.observations):raise ValueError('Unknown observation index')
            observation=state.observations[i]
            if observation.error or observation.image_key!=ref['image_key']:
                raise ValueError('Citation does not reference a successful recorded image observation')
            if not any(c.role_id==ref['role_id'] for c in observation.report.claims):
                raise ValueError('Citation role is absent from this observation')
        return value

    def fusion_messages(self,state,gate):
        messages=self._messages(FUSION_INSTRUCTION,state,{
            'graph_gate':gate,'direct_analysis':self.direct_baseline['message'].get('content') or ''})
        messages[0]={'role':'system','content':FUSION_SYSTEM}
        return messages

    async def finalize(self,state):
        self._event('state_before_gate',state=state,images=self.image_registry)
        store(self.outputdir/'composition.json', adaptive.evidence_summary(state))
        gate=self._call('graph_gate',self.gate_messages(state),
                        lambda value:self.validate_gate(value,state),'submit_graph_gate',GATE_SCHEMA)
        self.gate_result=gate
        store(self.outputdir/'gate.json',gate)  # Persist BEFORE the final aggregator request.
        self._event('gate_complete',gate=gate)
        def validate(message):
            if message.get('tool_calls'):raise ValueError('Final aggregator must not call tools')
            text=message.get('content')
            if not isinstance(text,str) or not text.strip():raise ValueError('Empty final aggregator response')
            import re
            match=re.search(r'<answer>\s*(A|B|C|D|NO_ANSWER)\s*</answer>',text,re.I)
            if not match:raise ValueError('Final answer requires an explicit option or NO_ANSWER tag')
            return text
        final=self._call('aggregator',self.reader_messages(state,gate),validate)
        store(self.outputdir/'aggregator.json',{'answer_text':final, 'reader_mode':self.reader_mode})
        return final
