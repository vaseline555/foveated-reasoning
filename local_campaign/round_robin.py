"""One fair request slot per waiting method; persisted budget/cooldown stay authoritative."""
import threading
import time
import uuid

METHOD_ORDER=('Direct','ReAct','FOVEA-Greedy','ADAPTIVE_V4')


def next_method(order,last,pending):
    start=(order.index(last)+1)%len(order) if last in order else 0
    return next((order[(start+i)%len(order)] for i in range(len(order))
                 if order[(start+i)%len(order)] in pending),None)


class RoundRobinGate:
    def __init__(self,client,paused_error):
        self.client=client;self.paused_error=paused_error
        self.condition=threading.Condition();self.pending={}
        with client.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS rr_state (singleton INTEGER PRIMARY KEY,last_method TEXT)')
            db.execute('INSERT OR IGNORE INTO rr_state VALUES (1,NULL)')
            db.execute('''CREATE TABLE IF NOT EXISTS rr_requests (
                id TEXT PRIMARY KEY,method TEXT,case_id TEXT,stage TEXT,
                queued_at REAL,granted_at REAL,finished_at REAL,status TEXT)''')
            db.execute("UPDATE rr_requests SET status='abandoned_on_resume',finished_at=? WHERE status='waiting'",(time.time(),))
            self.last=db.execute('SELECT last_method FROM rr_state WHERE singleton=1').fetchone()[0]

    def wait(self,client,case=None,method=None,stage=None):
        if method not in METHOD_ORDER:raise ValueError('Unknown scheduling method: '+str(method))
        token=uuid.uuid4().hex;entered=time.time();granted=False
        with self.condition:
            if method in self.pending:raise RuntimeError('More than one pending worker for '+method)
            self.pending[method]=token
            with client.connect() as db:
                db.execute("INSERT INTO rr_requests VALUES (?,?,?,?,?,NULL,NULL,'waiting')",(token,method,case,stage,entered))
            self.condition.notify_all()
        try:
            while True:
                client._check_stop()
                with self.condition:
                    selected=next_method(METHOD_ORDER,self.last,self.pending)
                    if selected==method:
                        with client.connect() as db:
                            db.execute('BEGIN IMMEDIATE')
                            health=dict(db.execute('SELECT * FROM api_health WHERE singleton=1').fetchone())
                            if health['paused']:raise self.paused_error('Persistent HTTP429; paid dispatch is paused')
                            now=time.time();ready=max(health['cooldown_until'],health['next_dispatch_at'])
                            if now>=ready:
                                # Apply the existing 30/45/60s policy inside the transaction as well
                                # as the pacing helper; retain any longer server-directed cooldown.
                                effective_gap=max(client.min_request_interval,
                                    60. if health['total_429']>=3 else (45. if health['total_429'] else 30.))
                                db.execute('UPDATE api_health SET next_dispatch_at=? WHERE singleton=1',(now+effective_gap,))
                                db.execute('UPDATE rr_state SET last_method=? WHERE singleton=1',(method,))
                                db.execute("UPDATE rr_requests SET granted_at=?,finished_at=?,status='granted' WHERE id=?",(now,now,token))
                                self.last=method;del self.pending[method];granted=True
                                self.condition.notify_all()
                                return
                        self.condition.wait(timeout=min(.5,max(.01,ready-time.time())))
                    else:self.condition.wait(timeout=.5)
        finally:
            if not granted:
                with self.condition:
                    if self.pending.get(method)==token:del self.pending[method]
                    with client.connect() as db:
                        db.execute("UPDATE rr_requests SET status='cancelled',finished_at=? WHERE id=?",(time.time(),token))
                    self.condition.notify_all()


def install(base_class,client,paused_error):
    gate=RoundRobinGate(client,paused_error)
    def wait(instance,case=None,method=None,stage=None):
        return gate.wait(instance,case,method,stage)
    base_class._wait_for_dispatch=wait
    return gate
