# vim3 power analysis: idle log plus the inference window of the real-image run, energy per inference
# runs on: host pc
# input: idle_power_run3_vim3.csv, realimg_e2e_power_run3_vim3.csv
# output: /home/eagle/THESIS/vim3_npu/realimg_power_results_run3_vim3.json

import csv
import json
import statistics

IDLE_CSV = '/home/eagle/THESIS/idle_power_run3_vim3.csv'
REALIMG_CSV = '/home/eagle/THESIS/realimg_e2e_power_run3_vim3.csv'
OUTPUT_JSON = '/home/eagle/THESIS/vim3_npu/realimg_power_results_run3_vim3.json'

# from benchmark output
invoke_ms = 1374.117
e2e_ms = 1407.354
fps = 0.711

idle_rows = []
with open(IDLE_CSV) as f:
    for row in csv.DictReader(f):
        idle_rows.append(row)

realimg_rows = []
with open(REALIMG_CSV) as f:
    for row in csv.DictReader(f):
        realimg_rows.append(row)

idle_w = [float(r['power_W']) for r in idle_rows]
warmup_w = [float(r['power_W']) for r in realimg_rows[95:103]]
steady_w = [float(r['power_W']) for r in realimg_rows[103:823]]

idle_mean = statistics.mean(idle_w)
idle_median = statistics.median(idle_w)
idle_std = statistics.stdev(idle_w)

steady_mean = statistics.mean(steady_w)
steady_median = statistics.median(steady_w)
steady_std = statistics.stdev(steady_w)

delta_w = steady_mean - idle_mean
e_invoke = steady_mean * (invoke_ms / 1000)
e_e2e = steady_mean * (e2e_ms / 1000)
fps_per_w = fps / steady_mean

v_idle = [float(r['voltage_V']) for r in realimg_rows[:95]]
v_inf = [float(r['voltage_V']) for r in realimg_rows[103:823]]
v_sag = statistics.mean(v_idle) - statistics.mean(v_inf)
v_sag_pct = (v_sag / statistics.mean(v_idle)) * 100

wh_per_day = steady_mean * 24
cost_per_yr = wh_per_day * 365 / 1000 * 0.10

print(f'idle power (n={len(idle_w)}):')
print(f'  mean={idle_mean:.4f}W  median={idle_median:.4f}W  std={idle_std:.4f}W')
print(f'\nwarmup power (n={len(warmup_w)}):')
print(f'  mean={statistics.mean(warmup_w):.4f}W  max={max(warmup_w):.4f}W')
print(f'\nsteady-state inference (n={len(steady_w)}):')
print(f'  mean={steady_mean:.4f}W  median={steady_median:.4f}W  std={steady_std:.4f}W')
print(f'\ndelta (inference - idle): {delta_w:.4f}W')
print(f'energy/invoke:  {e_invoke:.4f}J')
print(f'energy/e2e:     {e_e2e:.4f}J')
print(f'FPS/W:          {fps_per_w:.4f}')
print(f'\nvoltage sag: {v_sag:.4f}V ({v_sag_pct:.2f}%)')
print('\noperational cost (24/7 @ EUR0.10/kWh):')
print(f'  {wh_per_day:.2f} Wh/day  ->  EUR{cost_per_yr:.2f}/year')

results = {
    'platform': 'Khadas VIM3 Basic — Amlogic A311D NPU (TIM-VX)',
    'model': 'EfficientDet-Lite2 INT8 — model_run3.tflite',
    'benchmark_type': 'real_image_e2e',
    'idle_power': {
        'mean_W': round(idle_mean, 4),
        'median_W': round(idle_median, 4),
        'std_W': round(idle_std, 4),
        'n_samples': len(idle_w),
        'source': 'idle_power_run3_vim3.csv',
    },
    'warmup_power': {
        'mean_W': round(statistics.mean(warmup_w), 4),
        'max_W': round(max(warmup_w), 4),
        'n_samples': len(warmup_w),
        'note': 'cold-start first invoke spike + initial warmup invokes',
    },
    'inference_power': {
        'mean_W': round(steady_mean, 4),
        'median_W': round(steady_median, 4),
        'std_W': round(steady_std, 4),
        'n_samples': len(steady_w),
        'source': 'realimg_e2e_power_run3_vim3.csv rows 103-822',
    },
    'delta_W': round(delta_w, 4),
    'energy_per_invoke_J': round(e_invoke, 4),
    'energy_per_e2e_J': round(e_e2e, 4),
    'fps_per_watt': round(fps_per_w, 4),
    'voltage_sag_V': round(v_sag, 4),
    'voltage_sag_pct': round(v_sag_pct, 2),
    'operational_cost': {
        'wh_per_day': round(wh_per_day, 2),
        'eur_per_year': round(cost_per_yr, 2),
        'rate_eur_per_kwh': 0.10,
        'assumption': 'continuous 24/7 operation',
    },
    'timing_reference': {
        'invoke_median_ms': invoke_ms,
        'e2e_median_ms': e2e_ms,
        'fps': fps,
    },
    'coral_reference': {
        'idle_W': 0.796,
        'inference_W': 1.504,
        'energy_per_invoke_J': 0.492,
        'energy_per_e2e_J': 0.565,
        'fps_per_watt': 1.757,
    },
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\ndone: saved -> {OUTPUT_JSON}')
