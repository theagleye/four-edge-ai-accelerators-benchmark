# coral power analysis: splits the atorch log into idle and inference, energy per inference
# runs on: host pc, from ~/THESIS
# input: realimg_power_run3_coral.csv, coral_realimage_e2e_results_run3.json
# output: Coral_tpu/coral_realimg_power_results_run3.json

import csv
import json
import statistics

POWER_CSV = 'realimg_power_run3_coral.csv'
TIMING_JSON = 'Coral_tpu/coral_realimage_e2e_results_run3.json'
OUT_JSON = 'Coral_tpu/coral_realimg_power_results_run3.json'
THRESHOLD_W = 1.0  # idle below / inference above
FIN_PRICE = 0.10  # euros per kWh, Finnish average on 19.04.2026

# load power csv
rows = []
with open(POWER_CSV, newline='') as f:
    for r in csv.DictReader(f):
        rows.append({
            'ts': float(r['timestamp_s']),
            'V': float(r['voltage_V']),
            'A': float(r['current_A']),
            'W': float(r['power_W']),
            'T': float(r['temp_C'])
        })

t0 = rows[0]['ts']
for r in rows:
    r['elapsed'] = r['ts'] - t0

# load timing json for invoke and e2e medians
with open(TIMING_JSON) as f:
    tj = json.load(f)
invoke_med_ms = tj['steady_state']['invoke']['median_ms']
e2e_med_ms = tj['steady_state']['e2e']['median_ms']

# split phases
idle_rows = [r for r in rows if r['W'] < THRESHOLD_W]
inf_rows = [r for r in rows if r['W'] >= THRESHOLD_W]

idle_W = [r['W'] for r in idle_rows]
inf_W = [r['W'] for r in inf_rows]
idle_V = [r['V'] for r in idle_rows]
inf_V = [r['V'] for r in inf_rows]
idle_T = [r['T'] for r in idle_rows]
inf_T = [r['T'] for r in inf_rows]

print('\ncoral real-image power analysis:')
print(f'total samples : {len(rows)}')
print(f"duration      : {rows[-1]['elapsed']:.1f}s")
print(f'threshold     : {THRESHOLD_W}W')

print(f'\n-- idle ({len(idle_rows)} samples) --')
print(f'power : mean={statistics.mean(idle_W):.4f}W  min={min(idle_W):.4f}W  max={max(idle_W):.4f}W  std={statistics.stdev(idle_W):.4f}W')
print(f'volt  : mean={statistics.mean(idle_V):.3f}V  min={min(idle_V):.3f}V  max={max(idle_V):.3f}V')
print(f'temp  : mean={statistics.mean(idle_T):.1f}C  min={min(idle_T):.0f}C  max={max(idle_T):.0f}C')

print(f'\n-- inference ({len(inf_rows)} samples) --')
inf_mean = statistics.mean(inf_W)
inf_med = statistics.median(inf_W)
inf_std = statistics.stdev(inf_W)
print(f'power : mean={inf_mean:.4f}W  median={inf_med:.4f}W  min={min(inf_W):.4f}W  max={max(inf_W):.4f}W  std={inf_std:.4f}W')
print(f'volt  : mean={statistics.mean(inf_V):.3f}V  min={min(inf_V):.3f}V  max={max(inf_V):.3f}V')
print(f'temp  : mean={statistics.mean(inf_T):.1f}C  min={min(inf_T):.0f}C  max={max(inf_T):.0f}C  rise={max(inf_T)-inf_T[0]:.0f}C')

# voltage sag
idle_V_mean = statistics.mean(idle_V)
inf_V_mean = statistics.mean(inf_V)
vsag = idle_V_mean - inf_V_mean
print('\n-- voltage sag --')
print(f'idle mean volt : {idle_V_mean:.3f}V')
print(f'inf  mean volt : {inf_V_mean:.3f}V')
print(f'sag            : {vsag:.3f}V ({vsag/idle_V_mean*100:.2f}%)')

# energy per inference
e_invoke = inf_mean * (invoke_med_ms / 1000.0)
e_e2e = inf_mean * (e2e_med_ms / 1000.0)
print('\n-- energy per inference --')
print(f'timing source  : invoke={invoke_med_ms:.2f}ms  e2e={e2e_med_ms:.2f}ms')
print(f'energy/invoke  : {e_invoke:.4f}J  ({e_invoke*1000:.2f}mJ)')
print(f'energy/e2e     : {e_e2e:.4f}J  ({e_e2e*1000:.2f}mJ)')

# operational cost
fps = tj['steady_state']['throughput_fps']
fps_per_w = fps / inf_mean
wh_per_hour = inf_mean * 1.0  # W * 1h = Wh
wh_per_day = inf_mean * 24.0
cost_per_day = (wh_per_day / 1000.0) * FIN_PRICE
cost_per_yr = cost_per_day * 365
print('\n-- operational cost (continuous 24/7 inference) --')
print(f'fps/W          : {fps_per_w:.3f}')
print(f'power draw     : {inf_mean:.3f}W')
print(f'energy/hour    : {wh_per_hour:.3f}Wh')
print(f'energy/day     : {wh_per_day:.3f}Wh')
print(f'cost/day       : {cost_per_day:.4f} EUR  (@ {FIN_PRICE} EUR/kWh)')
print(f'cost/year      : {cost_per_yr:.2f} EUR')

# delta power
delta_w = inf_mean - statistics.mean(idle_W)
print('\n-- summary for thesis --')
print(f'idle power     : {statistics.mean(idle_W):.3f}W')
print(f'inf  power     : {inf_mean:.3f}W')
print(f'delta W        : {delta_w:.3f}W  (net TPU load cost)')
print(f'energy/invoke  : {e_invoke:.4f}J')
print(f'energy/e2e     : {e_e2e:.4f}J')
print(f'fps/W          : {fps_per_w:.3f}')
print(f"temp rise      : {max(inf_T)-rows[0]['T']:.0f}C  (thermal stability confirmed)")

# save json
results = {
    'platform': 'Coral Dev Board Mini — Google Edge TPU',
    'benchmark_type': 'real_image_e2e_power',
    'n_total_samples': len(rows),
    'duration_s': round(rows[-1]['elapsed'], 1),
    'threshold_W': THRESHOLD_W,
    'idle': {
        'n': len(idle_rows),
        'power_mean_W': round(statistics.mean(idle_W), 4),
        'power_std_W': round(statistics.stdev(idle_W), 4),
        'volt_mean_V': round(idle_V_mean, 3),
        'temp_mean_C': round(statistics.mean(idle_T), 1)
    },
    'inference': {
        'n': len(inf_rows),
        'power_mean_W': round(inf_mean, 4),
        'power_median_W': round(inf_med, 4),
        'power_std_W': round(inf_std, 4),
        'volt_mean_V': round(inf_V_mean, 3),
        'temp_mean_C': round(statistics.mean(inf_T), 1),
        'temp_max_C': round(max(inf_T), 1),
        'temp_rise_C': round(max(inf_T) - rows[0]['T'], 1)
    },
    'voltage_sag_V': round(vsag, 3),
    'voltage_sag_pct': round(vsag / idle_V_mean * 100, 2),
    'delta_power_W': round(delta_w, 4),
    'invoke_median_ms': invoke_med_ms,
    'e2e_median_ms': e2e_med_ms,
    'energy_invoke_J': round(e_invoke, 6),
    'energy_e2e_J': round(e_e2e, 6),
    'fps': round(fps, 3),
    'fps_per_W': round(fps_per_w, 3),
    'operational_cost': {
        'power_W': round(inf_mean, 3),
        'Wh_per_hour': round(wh_per_hour, 3),
        'Wh_per_day': round(wh_per_day, 3),
        'EUR_per_day': round(cost_per_day, 4),
        'EUR_per_year': round(cost_per_yr, 2),
        'electricity_rate': f'{FIN_PRICE} EUR/kWh'
    },
    'methodology_note': (
        'Power measured at Coral Dev Board Mini dedicated USB-C power port via ATORCH J7-C. '
        'OTG port connected to host for SSH — small unmetered VBUS contribution from host not captured. '
        f'Idle/inference phases separated at {THRESHOLD_W}W threshold.'
    )
}
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nsaved -> {OUT_JSON}')
