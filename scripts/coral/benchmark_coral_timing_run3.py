# coral edge tpu timing on synthetic input: model load, warmup, 100 steady-state runs, cpu and ram
# runs on: coral dev board mini
# output: /mnt/thesis/coral_timing_results_run3.json

import time
import threading
import statistics
import json
import sys
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:
    print('fatal: psutil not found. Install: pip3 install psutil')
    sys.exit(1)

try:
    from pycoral.utils.edgetpu import make_interpreter
except ImportError:
    print('fatal: pycoral not found.')
    sys.exit(1)


# config
MODEL_PATH = '/mnt/thesis/model_run3_edgetpu.tflite'
INPUT_H = 448
INPUT_W = 448
NUM_WARMUP = 5
NUM_RUNS = 100
MONITOR_INTERVAL_S = 0.1
OUTPUT_JSON = '/mnt/thesis/coral_timing_results_run3.json'


# background system monitor

class SystemMonitor:
    def __init__(self, interval=MONITOR_INTERVAL_S):
        self.interval = interval
        self.cpu_pct = []
        self.ram_mb = []
        self._running = False
        self._thread = None
        self._proc = psutil.Process()

    def start(self):
        # prime psutil, the first call returns 0.0
        psutil.cpu_percent(interval=None)
        self._proc.cpu_percent(interval=None)
        time.sleep(0.2)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        while self._running:
            self.cpu_pct.append(psutil.cpu_percent(interval=None))
            self.ram_mb.append(self._proc.memory_info().rss / (1024 ** 2))
            time.sleep(self.interval)

    def summary(self):
        def stats(lst, label):
            if len(lst) < 2:
                return {'note': f'too few samples ({len(lst)}) for {label}'}
            return {
                'mean': round(statistics.mean(lst), 4),
                'median': round(statistics.median(lst), 4),
                'std': round(statistics.stdev(lst), 4),
                'min': round(min(lst), 4),
                'max': round(max(lst), 4),
                'n_samples': len(lst),
            }
        return {
            'cpu_percent': stats(self.cpu_pct, 'CPU'),
            'ram_mb': stats(self.ram_mb, 'RAM'),
            'power_watts': 'not_available — use USB-C power meter externally',
        }


# timing stats helper

def timing_stats(times_ms, label=''):
    if len(times_ms) < 2:
        return {'note': f'too few samples for {label}'}
    return {
        'median_ms': round(statistics.median(times_ms), 3),
        'mean_ms': round(statistics.mean(times_ms), 3),
        'std_ms': round(statistics.stdev(times_ms), 3),
        'min_ms': round(min(times_ms), 3),
        'max_ms': round(max(times_ms), 3),
        'n': len(times_ms),
    }


# main

def main():
    print('  coral Dev Board Mini Benchmark - EfficientDet-Lite2 INT8')
    print(f'  model: {MODEL_PATH}')

    # preflight
    if not Path(MODEL_PATH).exists():
        print(f'fatal: model not found: {MODEL_PATH}')
        sys.exit(1)

    # step 1: load model
    print('\nstep 1: loading model + Edge TPU delegate ...')
    t_load_start = time.perf_counter()
    try:
        interpreter = make_interpreter(MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f'fatal: model load failed: {e}')
        raise
    load_time_ms = (time.perf_counter() - t_load_start) * 1000
    print(f'         -> {load_time_ms:.1f} ms')

    # inspect tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print('\ninput tensor:')
    for d in input_details:
        print(f"       idx={d['index']}  shape={list(d['shape'])}  dtype={d['dtype'].__name__}")

    print('output tensors:')
    for d in output_details:
        print(f"       idx={d['index']}  name={d['name']}  shape={list(d['shape'])}  dtype={d['dtype'].__name__}")

    input_idx = input_details[0]['index']
    input_dtype = input_details[0]['dtype']

    # build dummy input
    rng = np.random.RandomState(42)
    dummy = rng.randint(0, 256, (1, INPUT_H, INPUT_W, 3)).astype(input_dtype)

    # step 2: warm-up
    print(f'\nstep 2: warm-up ({NUM_WARMUP} runs) ...')
    warmup_invoke_ms = []
    warmup_e2e_ms = []

    for i in range(NUM_WARMUP):
        t_e2e0 = time.perf_counter()
        interpreter.set_tensor(input_idx, dummy)
        t_inv0 = time.perf_counter()
        interpreter.invoke()
        t_inv1 = time.perf_counter()
        for od in output_details:
            _ = interpreter.get_tensor(od['index'])
        t_e2e1 = time.perf_counter()

        inv_ms = (t_inv1 - t_inv0) * 1000
        e2e_ms = (t_e2e1 - t_e2e0) * 1000
        warmup_invoke_ms.append(round(inv_ms, 2))
        warmup_e2e_ms.append(round(e2e_ms, 2))
        print(f'  #{i+1:02d}  invoke={inv_ms:8.2f} ms   e2e={e2e_ms:8.2f} ms')

    # step 3: steady-state benchmark
    print(f'\nstep 3: steady-state benchmark ({NUM_RUNS} runs) ...')
    monitor = SystemMonitor()
    monitor.start()

    invoke_ms_list = []
    e2e_ms_list = []
    bench_t0 = time.perf_counter()

    for i in range(NUM_RUNS):
        t_e2e0 = time.perf_counter()
        interpreter.set_tensor(input_idx, dummy)
        t_inv0 = time.perf_counter()
        interpreter.invoke()
        t_inv1 = time.perf_counter()
        for od in output_details:
            _ = interpreter.get_tensor(od['index'])
        t_e2e1 = time.perf_counter()

        invoke_ms_list.append((t_inv1 - t_inv0) * 1000)
        e2e_ms_list.append((t_e2e1 - t_e2e0) * 1000)

        if (i + 1) % 25 == 0:
            print(f'  ... {i+1}/{NUM_RUNS} completed  '
                  f'(last invoke={invoke_ms_list[-1]:.1f} ms  e2e={e2e_ms_list[-1]:.1f} ms)')

    bench_elapsed = time.perf_counter() - bench_t0
    monitor.stop()
    throughput_fps = NUM_RUNS / bench_elapsed

    # step 4: compile results
    results = {
        'platform': 'Coral Dev Board Mini — Google Edge TPU',
        'model': 'EfficientDet-Lite2 INT8 — model_run3_edgetpu.tflite',
        'input_shape': [1, INPUT_H, INPUT_W, 3],
        'num_warmup': NUM_WARMUP,
        'num_runs': NUM_RUNS,

        'load_time_ms': round(load_time_ms, 2),

        'warmup': {
            'invoke_ms_per_run': warmup_invoke_ms,
            'e2e_ms_per_run': warmup_e2e_ms,
            'note': 'First run typically slowest (JIT/cache cold start)',
        },

        'steady_state': {
            'invoke': timing_stats(invoke_ms_list, 'invoke'),
            'e2e': timing_stats(e2e_ms_list, 'e2e'),
            'throughput_fps': round(throughput_fps, 3),
            'total_bench_time_s': round(bench_elapsed, 3),
        },

        'system': monitor.summary(),

        'notes': {
            'e2e_def': 'E2E = set_tensor + invoke + get_tensor (all outputs)',
            'invoke_def': 'invoke only = interpreter.invoke() call',
            'input_type': 'synthetic dummy input uint8 [0,255] seed=42',
            'power_note': 'Power not measurable via sysfs — use USB-C power meter',
            'vim3_reference': {
                'invoke_median_ms': 1374,
                'e2e_median_ms': 1374,
                'throughput_fps': 0.727,
            },
        },
    }

    # print summary
    print('  RESULTS SUMMARY')
    print(f'  load time:          {load_time_ms:.1f} ms')
    print(f'\n  warm-up invoke (ms): {warmup_invoke_ms}')
    print(f'  warm-up e2e    (ms): {warmup_e2e_ms}')

    s = results['steady_state']
    inv = s['invoke']
    print('\n  steady-state invoke (ms):')
    print(f"    median={inv['median_ms']}  mean={inv['mean_ms']}  "
          f"std={inv['std_ms']}  min={inv['min_ms']}  max={inv['max_ms']}")

    e2e = s['e2e']
    print('\n  steady-state E2E (ms):')
    print(f"    median={e2e['median_ms']}  mean={e2e['mean_ms']}  "
          f"std={e2e['std_ms']}  min={e2e['min_ms']}  max={e2e['max_ms']}")

    print(f'\n  throughput:         {throughput_fps:.3f} FPS')

    sys_s = results['system']
    cpu_s = sys_s.get('cpu_percent', {})
    if isinstance(cpu_s, dict) and 'mean' in cpu_s:
        print(f"\n  CPU usage:  mean={cpu_s['mean']}%  max={cpu_s['max']}%")
    ram_s = sys_s.get('ram_mb', {})
    if isinstance(ram_s, dict) and 'mean' in ram_s:
        print(f"  RAM usage:  mean={ram_s['mean']} MB  max={ram_s['max']} MB")
    print('  power:      measure externally with USB-C power meter')

    print('\n   VIM3 reference ')
    print(f"  VIM3 invoke median: 1374 ms  |  Coral invoke median: {inv['median_ms']} ms")

    # save json
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\ndone: results saved -> {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
