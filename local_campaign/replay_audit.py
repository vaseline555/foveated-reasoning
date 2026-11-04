"""Reconstruct partial episodes against a strictly read-only response cache."""
import asyncio
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from contextlib import contextmanager
import campaign

class Boundary(Exception):pass
class ReadOnlyCache:
    def __init__(self,client,failed):self.models=client.models;self.db=client.db;self.hits=0;self.failed=failed
    @contextmanager
    def connect(self):
        db=sqlite3.connect('file:'+str(self.db)+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        try:yield db
        finally:db.close()
    def reserve_final(self,*args):pass
    def release_final(self,*args):pass
    def call(self,messages,stage,case,method,tools=None,max_tokens=2048,seed=42,validator=None,force_model=None):
        logical=dict(messages=messages,stage=stage,case=case,method=method,tools=tools,max_tokens=max_tokens,seed=seed,models=self.models,force_model=force_model)
        wire=json.dumps(logical,sort_keys=True,separators=(',',':'))
        key=hashlib.sha256(wire.encode()).hexdigest()
        with self.connect() as db:
            row=db.execute("SELECT * FROM attempts WHERE logical_key=? AND status='valid' ORDER BY finished DESC LIMIT 1",(key,)).fetchone()
            invalid=db.execute("SELECT 1 FROM attempts WHERE logical_key=? AND status='invalid'",(key,)).fetchone()
        if row:
            self.hits+=1
            value=json.loads(Path(row['response_path']).read_text())['choices'][0]['message']
            return value,dict(row)|{'cache_hit':True}
        if invalid:raise self.failed('Requested backbone output validation failed: primary: OUTPUT_VALIDATION_FAILED')
        raise Boundary(f'{method}/{case}/{stage}: first uncached request; no API dispatched')


def main():
    base,graph,client,gc=campaign.runtimes()
    cases={c['key']:c for c in campaign.read(campaign.RUN/'prepared/manifest.json')['cases']}
    result=[]
    for method,key in [('Greedy','vstar_153'),('Graph','vstar_153'),('ReAct','vstar_186')]:
        cache=ReadOnlyCache(client,base.RequestFailure)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                if method=='Graph':asyncio.run(graph.execute_case(cases[key],Path(tmp),cache,{'limits':{'max_steps':10,'max_model_calls':24}}))
                else:
                    baseline=base.effective_row(campaign.RUN,'Direct',cases[key])
                    base.trajectory(cache,Path(tmp),cases[key],'FOVEA-Greedy' if method=='Greedy' else method,baseline)
                boundary='already fully cached'
            except Boundary as exc:boundary=str(exc)
        result.append(dict(method=method,case=key,cached_responses_reused=cache.hits,boundary=boundary,api_dispatches=0))
    campaign.save(campaign.RUN/'scheduling/cache_replay_audit.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
