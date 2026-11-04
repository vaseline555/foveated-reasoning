"""Reservation tests; all requests are local and no API is contacted."""
import copy
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import transport


def models():
    spec = {'id': transport.MODEL, 'prompt_price': 0.25, 'completion_price': 0.75,
            'context_length': 262144, 'provider': {'only': ['Parasail'],
            'allow_fallbacks': False, 'require_parameters': True}}
    return {'primary': copy.deepcopy(spec), 'fallback': copy.deepcopy(spec)}


class TransportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.client = transport.AdaptiveClient(self.tmp.name, models(), cap=0.20)
        self.old = transport.Client(self.tmp.name, models(), cap=0.20,
                                    min_request_interval=3.0)

    def rows(self):
        with self.client.connect() as connection:
            return [dict(row) for row in connection.execute('SELECT * FROM attempts')]

    def reserve(self, client=None, key='k', upper=0.01, stage='plan'):
        return (client or self.client)._reserve(
            key, models()['primary'], 'primary', 'case', transport.METHOD,
            stage, upper, Path(self.tmp.name) / (key + '.request.json'))

    def test_hold_is_idempotent_and_visible_to_old_client(self):
        attempt = self.client.reserve_final('case')
        self.assertEqual(attempt, self.client.reserve_final('case'))
        self.assertAlmostEqual(self.old.stats()['held_usd'], self.client.final_upper)
        self.reserve(self.old, upper=0.13)
        with self.assertRaises(transport.BudgetStop):
            self.reserve(self.old, key='over-cap', upper=0.01)
        self.assertEqual(len(self.rows()), 2)

    def test_atomic_shared_cap(self):
        self.client.reserve_final('case')
        barrier = threading.Barrier(10)

        def compete(i):
            barrier.wait()
            try:
                self.reserve(self.old, key=f'parallel-{i}', upper=0.02)
                return True
            except transport.BudgetStop:
                return False

        with ThreadPoolExecutor(max_workers=10) as pool:
            successes = sum(pool.map(compete, range(10)))
        self.assertEqual(successes, 6)
        self.assertLessEqual(self.old.stats()['held_usd'], 0.20)
        self.assertAlmostEqual(self.old.stats()['held_usd'], self.client.final_upper + 0.12)

    def test_final_conversion_is_one_row_and_retry_is_new_attempt(self):
        hold = self.client.reserve_final('case')
        final = self.reserve(key='final', upper=0.02, stage='aggregator')
        self.assertEqual(hold, final)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0]['stage'], 'aggregator')
        self.assertEqual(self.rows()[0]['logical_key'], 'final')
        with self.assertRaises(transport.UnknownCharge):
            self.reserve(key='final', upper=0.02, stage='aggregator')
        self.client._update(final, status='rejected', cost=0)
        retry = self.reserve(key='final', upper=0.02, stage='aggregator')
        self.assertNotEqual(final, retry)
        self.assertEqual(len(self.rows()), 2)

    def test_final_upper_violation_preserves_hold(self):
        hold = self.client.reserve_final('case')
        with self.assertRaises(transport.BudgetStop):
            self.reserve(stage='aggregator', upper=self.client.final_upper + 0.001)
        self.assertEqual(self.rows()[0]['id'], hold)
        self.assertEqual(self.rows()[0]['status'], 'reserved_final')

    def test_unknown_and_dispatched_are_never_released(self):
        self.client.reserve_final('case')
        final = self.reserve(stage='aggregator')
        self.client._update(final, status='dispatched')
        self.assertEqual(self.client.release_final('case'), 0)
        self.client._update(final, status='unknown')
        self.assertEqual(self.client.release_final('case'), 0)
        with self.assertRaises(transport.UnknownCharge):
            self.client.reserve_final('case')
        self.assertEqual(self.rows()[0]['status'], 'unknown')
        self.assertIsNone(self.rows()[0]['cost'])
        self.assertEqual(self.rows()[0]['reserved'], 0.01)

    def test_release_preserves_nonphysical_provenance(self):
        hold = self.client.reserve_final('case')
        artifact = Path(self.rows()[0]['request_path'])
        original = artifact.read_text()
        self.assertEqual(self.client.release_final('case'), 1)
        self.assertEqual(self.client.release_final('case'), 0)
        row = self.rows()[0]
        self.assertEqual(row['id'], hold)
        self.assertEqual(row['status'], 'cancelled_reservation')
        self.assertEqual(row['cost'], 0)
        self.assertIsNone(row['response_path'])
        self.assertEqual(self.old.stats()['held_usd'], 0)
        self.client.reserve_final('case')
        self.assertEqual(artifact.read_text(), original)
        stats = self.client.final_reservation_stats()
        self.assertEqual(stats['pending_final_reservations'], 1)
        self.assertEqual(stats['cancelled_final_reservations'], 1)
        self.assertEqual(stats['physical_attempt_rows'], 0)

    def test_cached_final_releases_unused_hold_without_api(self):
        self.client.reserve_final('case')
        with patch.object(transport.Client, 'call', return_value=(
                {'content': 'A'}, {'cache_hit': True})) as call:
            result = self.client.call([], 'aggregator', 'case', transport.METHOD)
        self.assertTrue(result[1]['cache_hit'])
        self.assertEqual(call.call_args.kwargs['force_model'], 'primary')
        self.assertEqual(self.rows()[0]['status'], 'cancelled_reservation')

    def test_8b_provider_and_primary_guards(self):
        for field, value in (('id', 'qwen/qwen3-vl-30b-a3b-instruct'),
                             ('prompt_price', 0.20)):
            changed = models()
            changed['fallback'][field] = value
            with self.assertRaises(ValueError):
                transport.AdaptiveClient(self.tmp.name, changed, cap=0.20)
        changed = models()
        changed['primary']['provider']['only'] = ['Other']
        with self.assertRaises(ValueError):
            transport.AdaptiveClient(self.tmp.name, changed, cap=0.20)
        with self.assertRaises(ValueError):
            self.client.call([], 'plan', 'case', transport.METHOD, force_model='fallback')
        with self.assertRaises(ValueError):
            self.client.call([], 'plan', 'case', 'OTHER')
        with self.assertRaises(ValueError):
            self.client.call([], 'aggregator', 'case', transport.METHOD, max_tokens=2049)
        with self.assertRaises(ValueError):
            transport.AdaptiveClient(self.tmp.name, models(), cap=0.20,
                                     min_request_interval=2)


if __name__ == '__main__':
    unittest.main()
