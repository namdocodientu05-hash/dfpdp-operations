"""Run: python -m unittest test_operations -v"""
import unittest
from operations import SimulationEngine, SAMPLE_LOCATIONS


def run_until_stop(engine, max_ticks=800, speed=120):
    engine.set_speed(speed)
    engine.start()
    for _ in range(max_ticks):
        if not engine.running:
            break
        engine.tick()
        st = engine.dispatcher.env.state
        for v in st.vehicle_list:
            assert -.001 <= v.load <= v.capacity + .001
    return engine.state()


class V8OperationsTests(unittest.TestCase):
    def test_limits(self):
        bad = [dict(vehicles=25), dict(orders=101), dict(days=0), dict(days=8),
               dict(traffic_region='FAKE'), dict(algorithm='fake')]
        for params in bad:
            with self.subTest(params=params), self.assertRaises(ValueError):
                SimulationEngine(**params)

    def test_embedded_traffic_profiles(self):
        expected = {'HCMC':17.0}
        for region, speed in expected.items():
            with self.subTest(region=region):
                e = SimulationEngine(orders=3, days=2, traffic_region=region)
                s = e.state()
                self.assertAlmostEqual(s['traffic_profile']['selected']['speed_kmh'], speed, places=1)
                self.assertEqual(len(s['traffic_profile']['selected']['profile']), 24)

    def test_multiday_carryover(self):
        e = SimulationEngine(vehicles=1, orders=100, dynamic_percent=20,
                             days=2, algorithm='nearest', seed=9)
        s = run_until_stop(e, max_ticks=300, speed=240)
        self.assertEqual(s['run']['status'], 'horizon_end')
        self.assertEqual(len(s['calendar']['daily_summaries']), 2)
        self.assertGreater(s['calendar']['daily_summaries'][0]['carryover_count'], 0)
        self.assertGreater(s['summary']['carryover'], 0)

    def test_dynamic_demand_spans_multiple_days(self):
        e = SimulationEngine(vehicles=2, orders=12, dynamic_percent=100, days=3, seed=123)
        releases = [x['release_time'] for x in e.auto_dynamic_schedule]
        self.assertTrue(any(1440 < x < 2880 for x in releases))
        self.assertTrue(any(x >= 2880 for x in releases))

    def test_drl_observability_trace(self):
        e = SimulationEngine(vehicles=4, orders=18, dynamic_percent=30,
                             algorithm='drl', days=2, seed=42)
        e.set_speed(30); e.start()
        for _ in range(20):
            e.tick()
            if e.decision_trace:
                break
        s = e.state(); t = s['intelligence']['latest']
        self.assertIsNotNone(t)
        self.assertEqual(t['algorithm'], 'drl')
        self.assertIn('Z', t['state_shape'])
        self.assertTrue(t['top_candidates'])
        self.assertIsNotNone(t['top_candidates'][0]['probability'])
        self.assertEqual(len(t['pipeline']), 5)

    def test_manual_order_still_supported(self):
        e = SimulationEngine(orders=10, dynamic_percent=0, days=2)
        before = len(e.state()['orders'])
        e.create_order(pickup=SAMPLE_LOCATIONS['HCMUT'], delivery=SAMPLE_LOCATIONS['BEN_THANH'],
                       weight=30, note='manual')
        self.assertEqual(len(e.state()['orders']), before + 1)

    def test_operating_window_ends_at_22(self):
        e = SimulationEngine(orders=3, days=2, traffic_region='HCMC')
        s = e.state()
        self.assertEqual(s['clock']['shift_start'], '08:00')
        self.assertEqual(s['clock']['shift_end'], '22:00')

    def test_playback_range(self):
        e = SimulationEngine()
        e.set_speed(240)
        self.assertEqual(e.sim_minutes_per_tick, 240)
        e.set_speed(1000)
        self.assertEqual(e.sim_minutes_per_tick, 240)


if __name__ == '__main__':
    unittest.main()
