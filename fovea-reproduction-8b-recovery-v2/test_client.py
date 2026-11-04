import json
import base64
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from client import BudgetStop, Client, RequestFailure, UnknownCharge, RateLimitPause


MODELS = {
    "primary":{"id":"qwen/qwen3-vl-8b-instruct","prompt_price":0.25,"completion_price":0.75,"context_length":262144},
    "fallback":{"id":"qwen/qwen3-vl-30b-a3b-instruct","prompt_price":0.15,"completion_price":0.60,"context_length":262144},
}


class Response:
    def __init__(self,content,cost=0.00001):
        self.body = json.dumps({"id":"gen-test","choices":[{"message":{"role":"assistant","content":content},
            "finish_reason":"stop"}],"usage":{"cost":cost}}).encode()
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self): return self.body


class MeteredClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ,{"OPENROUTER_API_KEY":"test-secret"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.messages=[{"role":"user","content":"Which answer?"}]

    def client(self,cap=15,**kwargs):
        return Client(self.temp.name,MODELS,cap=cap,min_request_interval=0,**kwargs)

    def call(self,client,**kwargs):
        return client.call(self.messages,"answer","case1","DIRECT",max_tokens=16,**kwargs)

    def test_valid_response_cache_never_rebills(self):
        client=self.client()
        with patch("urllib.request.urlopen",return_value=Response("A")) as network:
            message,record=self.call(client,validator=lambda m:m["content"]=="A")
            same,cached=self.call(client,validator=lambda m:m["content"]=="A")
        self.assertEqual(network.call_count,1)
        self.assertTrue(cached["cache_hit"])
        self.assertEqual(client.stats()["attempts"],1)
        self.assertAlmostEqual(client.stats()["confirmed_usd"],0.00001)
        self.assertEqual(client.stats()["held_usd"],0)
        self.assertEqual(message,same)
        for path in Path(self.temp.name).rglob("*.json"):
            self.assertNotIn("test-secret",path.read_text())

    def test_invalid_primary_falls_back_and_both_costs_count(self):
        client=self.client()
        with patch("urllib.request.urlopen",side_effect=[Response("bad"),Response("A")]) as network:
            message,record=self.call(client,validator=lambda m:m["content"]=="A")
        self.assertEqual(network.call_count,2)
        self.assertEqual(record["role"],"fallback")
        self.assertEqual(record["fallback_reason"],["primary: OUTPUT_VALIDATION_FAILED"])
        self.assertAlmostEqual(client.stats()["confirmed_usd"],0.00002)

    def test_cap_applies_before_any_network_call(self):
        client=self.client(cap=0.000001)
        with patch("urllib.request.urlopen") as network:
            with self.assertRaises(BudgetStop): self.call(client)
        network.assert_not_called()
        self.assertEqual(client.stats()["attempts"],0)

    def test_unknown_cost_keeps_reservation_and_blocks_new_calls(self):
        response=Response("A")
        body=json.loads(response.body)
        body.pop("id")
        body["usage"]={}
        response.body=json.dumps(body).encode()
        client=self.client()
        with patch("urllib.request.urlopen",return_value=response):
            with self.assertRaises(UnknownCharge): self.call(client)
        self.assertGreater(client.stats()["held_usd"],0)
        self.assertEqual(client.stats()["confirmed_usd"],0)
        with patch("urllib.request.urlopen") as network:
            with self.assertRaises(UnknownCharge):
                client.call(self.messages,"answer","new_case","DIRECT",max_tokens=16)
        network.assert_not_called()

    def test_network_failure_does_not_repeat_or_switch_model(self):
        client=self.client()
        with patch("urllib.request.urlopen",side_effect=TimeoutError) as network:
            with self.assertRaises(UnknownCharge): self.call(client)
        self.assertEqual(network.call_count,1)
        self.assertGreater(client.stats()["held_usd"],0)

    def test_forced_fallback_is_separate_scope(self):
        client=self.client()
        with patch("urllib.request.urlopen",return_value=Response("A")) as network:
            _,primary=self.call(client)
            _,fallback=self.call(client,force_model="fallback")
        self.assertEqual(network.call_count,2)
        self.assertEqual(primary["role"],"primary")
        self.assertEqual(fallback["role"],"fallback")

    def test_invalid_completed_attempts_not_rebilled_on_resume(self):
        client=self.client()
        with patch("urllib.request.urlopen",side_effect=[Response("bad"),Response("bad")]):
            with self.assertRaises(RequestFailure): self.call(client,validator=lambda m:False)
        resumed=self.client()
        with patch("urllib.request.urlopen") as network:
            with self.assertRaises(RequestFailure): self.call(resumed,validator=lambda m:False)
        network.assert_not_called()
        self.assertEqual(resumed.stats()["attempts"],2)

    def test_frozen_cap_cannot_be_reset(self):
        self.client(cap=15)
        with self.assertRaises(ValueError): self.client(cap=14)

    def test_image_request_artifacts_reuse_content_addressed_asset(self):
        from PIL import Image
        stream=io.BytesIO()
        Image.new("RGB",(16,16),"red").save(stream,format="JPEG")
        encoded=base64.b64encode(stream.getvalue()).decode()
        image={"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+encoded}}
        client=self.client()
        messages=[{"role":"user","content":[image,image]}]
        with patch("urllib.request.urlopen",return_value=Response("A")):
            client.call(messages,"image","case1","DIRECT",max_tokens=16)
        assets=list((Path(self.temp.name)/"api/assets").glob("*"))
        self.assertEqual(len(assets),1)
        request=next((Path(self.temp.name)/"api").glob("*.request.json")).read_text()
        self.assertNotIn(encoded,request)
        self.assertIn("asset_path",request)

    def test_auth_error_is_zero_cost_rejection_without_fallback(self):
        error=urllib.error.HTTPError("https://openrouter.ai",401,"Unauthorized",{},io.BytesIO(b'{"error":"auth"}'))
        client=self.client()
        with patch("urllib.request.urlopen",side_effect=error) as network:
            with self.assertRaises(RequestFailure): self.call(client)
        self.assertEqual(network.call_count,1)
        self.assertEqual(client.stats()["held_usd"],0)
        self.assertEqual(client.stats()["confirmed_usd"],0)

    def test_historical_charge_counts_toward_hard_cap_and_stats(self):
        client=self.client(cap=15,historical_usd=14.999999)
        with patch('urllib.request.urlopen') as network:
            with self.assertRaises(BudgetStop):
                self.call(client)
        network.assert_not_called()
        self.assertAlmostEqual(client.stats()['confirmed_usd'],14.999999)
        self.assertEqual(client.stats()['run_confirmed_usd'],0)
        with self.assertRaises(ValueError):
            self.client(cap=15,historical_usd=0)

    def test_historical_and_new_charge_reported_separately(self):
        client=self.client(historical_usd=.92410405)
        with patch('urllib.request.urlopen',return_value=Response('A')):
            self.call(client)
        stats=client.stats()
        self.assertAlmostEqual(stats['confirmed_usd'],.92411405)
        self.assertAlmostEqual(stats['run_confirmed_usd'],.00001)
        self.assertAlmostEqual(stats['historical_confirmed_usd'],.92410405)
        self.assertAlmostEqual(stats['remaining_usd'],15-.92411405)

    def test_rate_limit_preserves_attempt_and_recovers_same_request_after_global_wait(self):
        client=self.client()
        clock=FakeClock()
        dispatch_times=[]
        rate_error=urllib.error.HTTPError('https://openrouter.ai',429,'Limited',
            {'Retry-After':'2'},io.BytesIO(b'{"error":"rate limited"}'))
        replies=iter([rate_error,Response('A')])
        def request(*args,**kwargs):
            dispatch_times.append(clock.time())
            reply=next(replies)
            if isinstance(reply,Exception): raise reply
            return reply
        with patch('client.time.time',side_effect=clock.time),patch('client.time.sleep',side_effect=clock.sleep),patch('urllib.request.urlopen',side_effect=request):
            message,record=self.call(client,force_model='primary')
        self.assertEqual(message['content'],'A')
        self.assertGreaterEqual(dispatch_times[1]-dispatch_times[0],30)
        with client.connect() as connection:
            rows=connection.execute('SELECT logical_key,status,cost FROM attempts ORDER BY started').fetchall()
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['logical_key'],rows[1]['logical_key'])
        self.assertEqual(rows[0]['status'],'rejected')
        self.assertEqual(rows[0]['cost'],0)
        self.assertEqual(rows[1]['status'],'valid')
        self.assertEqual(client.stats()['held_usd'],0)

    def test_separate_client_instances_share_dispatch_gate_and_cooldown(self):
        first=self.client()
        second=self.client()
        clock=FakeClock()
        with patch('client.time.time',side_effect=clock.time),patch('client.time.sleep',side_effect=clock.sleep):
            first._rate_limited('600','case1','DIRECT','answer')
            second._wait_for_dispatch('case2','DIRECT','answer')
        self.assertGreaterEqual(clock.now,1600)
        health=json.loads((Path(self.temp.name)/'api_wait.json').read_text())
        self.assertEqual(health['reason'],'HTTP429_global_cooldown')

    def test_persistent_rate_limit_pauses_without_advancing_case_or_holding_charge(self):
        client=self.client(rate_retry_limit=2)
        clock=FakeClock()
        errors=[urllib.error.HTTPError('https://openrouter.ai',429,'Limited',{},
            io.BytesIO(b'{"error":"rate limited"}')) for _ in range(2)]
        with patch('client.time.time',side_effect=clock.time),patch('client.time.sleep',side_effect=clock.sleep),patch('urllib.request.urlopen',side_effect=errors) as network:
            with self.assertRaises(RateLimitPause): self.call(client)
            with self.assertRaises(RateLimitPause):
                client.call(self.messages,'answer','case2','DIRECT',max_tokens=16)
        self.assertEqual(network.call_count,2)
        self.assertEqual(client.stats()['held_usd'],0)
        self.assertEqual(client.stats()['confirmed_usd'],0)
        with client.connect() as connection:
            cases={row[0] for row in connection.execute('select case_id from attempts')}
        self.assertEqual(cases,{'case1'})
        self.assertEqual(json.loads((Path(self.temp.name)/'api_wait.json').read_text())['status'],'paused')

    def test_stop_interrupts_cooldown_without_dispatch(self):
        client=self.client()
        clock=FakeClock()
        with patch('client.time.time',side_effect=clock.time):
            client._rate_limited(None,'case1','DIRECT','answer')
        (Path(self.temp.name)/'STOP').touch()
        with patch('urllib.request.urlopen') as network:
            with self.assertRaises(BudgetStop): self.call(client)
        network.assert_not_called()


class FakeClock:
    def __init__(self): self.now=1000.0
    def time(self): return self.now
    def sleep(self,seconds): self.now+=seconds


if __name__=="__main__": unittest.main()
