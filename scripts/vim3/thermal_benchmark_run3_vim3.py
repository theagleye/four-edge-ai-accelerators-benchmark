# vim3 npu 500-run repeated-inference test: latency percentiles, cv, temperature
# runs on: khadas vim3 (tim-vx delegate)
# output: /home/khadas/thermal_results_run3_vim3.json

import time
import statistics
import json
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    print('fatal: tflite_runtime not found')
    exit(1)

MODEL = '/home/khadas/vx_tflite/mobilenet_v1/model_run3.tflite'
DELEGATE = '/home/khadas/vx_tflite/mobilenet_v1/libvx_delegate.so'
OUT_JSON = '/home/khadas/thermal_results_run3_vim3.json'
NUM_WARMUP = 5
NUM_RUNS = 500
INPUT_SEED = 42

print(f'thermal throttling test - {NUM_RUNS} runs, warmup={NUM_WARMUP}')

print('\nloading model ...')
t_load = time.perf_counter()
interp = Interpreter(
    model_path=MODEL,
    experimental_delegates=[load_delegate(DELEGATE)]
)
interp.allocate_tensors()
load_ms = (time.perf_counter() - t_load) * 1000
print(f'  load time: {load_ms:.1f}ms')

inp = interp.get_input_details()[0]
inp_idx = inp['index']
inp_shape = inp['shape']

rng = np.random.default_rng(INPUT_SEED)
dummy = rng.integers(0, 255, inp_shape, dtype=np.uint8)

print(f'\nwarm-up ({NUM_WARMUP} runs) ...')
warmup_invoke = []
warmup_e2e = []
for i in range(NUM_WARMUP):
    t0 = time.perf_counter()
    interp.set_tensor(inp_idx, dummy)
    t1 = time.perf_counter()
    interp.invoke()
    t2 = time.perf_counter()
    _ = [interp.get_tensor(o['index']) for o in interp.get_output_details()]
    t3 = time.perf_counter()
    inv_ms = (t2 - t1) * 1000
    e2e_ms = (t3 - t0) * 1000
    warmup_invoke.append(round(inv_ms, 3))
    warmup_e2e.append(round(e2e_ms, 3))
    print(f'  #{i+1}  invoke={inv_ms:.2f}ms  e2e={e2e_ms:.2f}ms')

print(f'\nrunning {NUM_RUNS} iterations ...')
t_start = time.perf_counter()
raw_invoke = []
raw_e2e = []
for i in range(NUM_RUNS):
    t0 = time.perf_counter()
    interp.set_tensor(inp_idx, dummy)
    t1 = time.perf_counter()
    interp.invoke()
    t2 = time.perf_counter()
    _ = [interp.get_tensor(o['index']) for o in interp.get_output_details()]
    t3 = time.perf_counter()
    raw_invoke.append(round((t2 - t1) * 1000, 3))
    raw_e2e.append(round((t3 - t0) * 1000, 3))
    if (i + 1) % 100 == 0:
        print(f'  [{i+1:3d}/{NUM_RUNS}]  invoke={raw_invoke[-1]:.1f}ms  e2e={raw_e2e[-1]:.1f}ms')

bench_s = time.perf_counter() - t_start

inv_sorted = sorted(raw_invoke)
p50 = statistics.median(raw_invoke)
p95 = inv_sorted[int(0.95 * len(inv_sorted))]
p99 = inv_sorted[int(0.99 * len(inv_sorted))]
mean = statistics.mean(raw_invoke)
std = statistics.stdev(raw_invoke)
cv = std / mean
fps = NUM_RUNS / bench_s

print('\nresults:')
print(f'p50  : {p50:.3f}ms')
print(f'p95  : {p95:.3f}ms')
print(f'p99  : {p99:.3f}ms')
print(f'mean : {mean:.3f}ms')
print(f'std  : {std:.3f}ms')
print(f'CV   : {cv:.4f} ({cv*100:.2f}%)')
print(f'min  : {min(raw_invoke):.3f}ms')
print(f'max  : {max(raw_invoke):.3f}ms')
print(f'fps  : {fps:.3f}')
print(f'time : {bench_s:.1f}s')

results = {
    'platform': 'Khadas VIM3 Basic — Amlogic A311D NPU (TIM-VX)',
    'model': 'EfficientDet-Lite2 INT8 — model_run3.tflite',
    'test_type': 'thermal_throttling_500runs',
    'num_warmup': NUM_WARMUP,
    'num_runs': NUM_RUNS,
    'load_time_ms': round(load_ms, 2),
    'warmup_invoke_ms': warmup_invoke,
    'warmup_e2e_ms': warmup_e2e,
    'steady_state': {
        'invoke': {
            'raw_ms': raw_invoke,
            'p50_ms': round(p50, 3),
            'p95_ms': round(p95, 3),
            'p99_ms': round(p99, 3),
            'mean_ms': round(mean, 3),
            'std_ms': round(std, 3),
            'cv': round(cv, 4),
            'min_ms': round(min(raw_invoke), 3),
            'max_ms': round(max(raw_invoke), 3),
            'n': NUM_RUNS
        },
        'e2e': {
            'raw_ms': raw_e2e,
            'p50_ms': round(statistics.median(raw_e2e), 3),
            'mean_ms': round(statistics.mean(raw_e2e), 3),
            'std_ms': round(statistics.stdev(raw_e2e), 3),
            'n': NUM_RUNS
        },
        'throughput_fps': round(fps, 3),
        'total_bench_time_s': round(bench_s, 3)
    }
}
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f'saved -> {OUT_JSON}')
