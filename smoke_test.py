"""V8 end-to-end smoke test."""
from operations import SimulationEngine

e = SimulationEngine(vehicles=6, orders=36, dynamic_percent=50,
                     algorithm='drl', days=3, traffic_region='HCMC', seed=42)
s = e.state()
assert s['model']['checkpoint_loaded']
assert s['clock']['days_total'] == 3
assert s['traffic_profile']['selected']['speed_kmh'] == 17.0
assert len(s['traffic_profile']['selected']['profile']) == 24
assert any(x['release_time'] >= 1440 for x in e.auto_dynamic_schedule)

e.set_speed(120); e.start()
for _ in range(80):
    if not e.running:
        break
    e.tick()
    if e.decision_trace and e.state()['clock']['day_number'] >= 2:
        break
s = e.state()
assert s['intelligence']['latest'] is not None
assert s['intelligence']['latest']['state_shape']['Z'][1] == 12
assert s['clock']['day_number'] >= 1
assert 'calendar' in s and 'traffic_profile' in s
print('SMOKE TEST V8 OK')
print('day:', s['clock']['day_number'], '/', s['clock']['days_total'])
print('traffic:', s['traffic_profile']['selected']['region_label'], s['traffic_profile']['selected']['speed_kmh'], 'km/h')
print('latest decision:', s['intelligence']['latest']['vehicle_id'], '->', s['intelligence']['latest']['chosen_order'])
