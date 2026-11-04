import contextlib
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import round_robin as rr

class Cancelled(Exception):pass
class Paused(Exception):pass

class Client:
    min_request_interval=30.
    def __init__(self,path):self.path=path;self.stop=threading.Event()
    @contextlib.contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=10);db.row_factory=sqlite3.Row
        try:
            with db:yield db
        finally:db.close()
    def _check_stop(self):
        if self.stop.is_set():raise Cancelled()

class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.client=Client(Path(self.temp.name)/'ledger.sqlite')
        with self.client.connect() as d:
            d.execute('CREATE TABLE api_health(singleton INTEGER PRIMARY KEY,cooldown_until REAL,next_dispatch_at REAL,total_429 INTEGER,paused INTEGER)')
            d.execute('INSERT INTO api_health VALUES(1,0,1000000000000000,0,0)')
        self.gate=rr.RoundRobinGate(self.client,Paused)
    def count(self,status):
        with self.client.connect() as d:return d.execute('select count(*) from rr_requests where status=?',(status,)).fetchone()[0]
    def until(self,fn):
        deadline=time.monotonic()+3
        while not fn():
            if time.monotonic()>deadline:self.fail('Timed out waiting for scheduler')
            time.sleep(.01)
    def launch(self,method,errors):
        def job():
            try:self.gate.wait(self.client,'case',method,'stage')
            except Exception as e:errors.append(e)
        t=threading.Thread(target=job);t.start();self.addCleanup(lambda:t.join(timeout=2));return t

    def test_backlogged_methods_get_exact_round_robin_order(self):
        errors=[];threads=[self.launch(m,errors) for m in reversed(rr.METHOD_ORDER)]
        self.until(lambda:self.count('waiting')==4)
        self.assertEqual(self.count('granted'),0)
        # Advance an artificial clock once per request without sleeping 30s per slot.
        clock=iter(range(2000000000,2000010000,100))
        with patch.object(rr.time,'time',side_effect=lambda:next(clock)):
            with self.client.connect() as d:d.execute('UPDATE api_health SET next_dispatch_at=0')
            with self.gate.condition:self.gate.condition.notify_all()
            for t in threads:t.join(timeout=2)
        self.assertEqual(errors,[])
        with self.client.connect() as d:
            order=[x[0] for x in d.execute("select method from rr_requests where status='granted' order by granted_at")]
        self.assertEqual(order,list(rr.METHOD_ORDER))
        self.assertTrue(all(not t.is_alive() for t in threads))

    def test_pending_selection_skips_idle_methods_and_wraps(self):
        self.assertEqual(rr.next_method(rr.METHOD_ORDER,'Direct',{'Direct','ADAPTIVE_V4'}),'ADAPTIVE_V4')
        self.assertEqual(rr.next_method(rr.METHOD_ORDER,'ADAPTIVE_V4',{'Direct','ReAct'}),'Direct')
        self.assertIsNone(rr.next_method(rr.METHOD_ORDER,'ReAct',set()))

    def test_cooldown_and_adaptive_gap_are_not_shortened(self):
        errors=[]
        with patch.object(rr.time,'time',return_value=100.):
            with self.client.connect() as d:d.execute('UPDATE api_health SET cooldown_until=200,next_dispatch_at=150,total_429=5')
            t=self.launch('ReAct',errors);self.until(lambda:self.count('waiting')==1)
            self.assertEqual(self.count('granted'),0)
            with patch.object(rr.time,'time',return_value=201.):
                with self.gate.condition:self.gate.condition.notify_all()
                t.join(timeout=2)
        self.assertEqual(errors,[])
        with self.client.connect() as d:
            row=d.execute('select cooldown_until,next_dispatch_at from api_health').fetchone()
        self.assertEqual(tuple(row),(200.,261.))

    def test_stop_cancels_queue_entry_without_dispatch(self):
        errors=[];t=self.launch('ADAPTIVE_V4',errors)
        self.until(lambda:self.count('waiting')==1);self.client.stop.set()
        with self.gate.condition:self.gate.condition.notify_all()
        t.join(timeout=2)
        self.assertEqual(len(errors),1);self.assertIsInstance(errors[0],Cancelled)
        self.assertEqual(self.count('cancelled'),1);self.assertEqual(self.count('granted'),0)
        self.assertEqual(self.gate.pending,{})

if __name__=='__main__':unittest.main()
