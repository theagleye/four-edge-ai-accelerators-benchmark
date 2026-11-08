# vim3 npu timing on synthetic input: model load, warmup, 100 steady-state runs, cpu and ram
# runs on: khadas vim3 (tim-vx delegate)
# output: /home/khadas/timing_results_run3_vim3.json

import time
import threading
import statistics
import json
import glob
import sys
from pathlib import Path

import numpy as np
import psutil

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    print('fatal: tflite_runtime not found.')
    sys.exit(1)

MODEL_PATH = '/home/khadas/vx_tflite/mobilenet_v1/model_run3.tflite'
DELEGATE_PATH = '/home/khadas/vx_tflite/mobilenet_v1/libvx_delegate.so'
INPUT_H = 448
INPUT_W = 448
NUM_WARMUP = 5
NUM_RUNS = 100
MONITOR_INTERVAL_S = 0.1
OUTPUT_JSON = '/home/khadas/timing_results_run3_vim3.json'


def find_power_sysfs():
    candidates = sorted(glob.glob('/sys/class/hwmon/hwmon*/power*_input'))
    for p in candidates:
        try:
            val = int(Path(p).read_text().strip())
            if val > 0:
                print(f'power sysfs found: {p}  ({val} uW = {val/1e6:.3f} W)')
                return p
        except Exception:
            pass
    return None


def read_power_watts(path):
    try:
        return int(Path(path).read_text().strip()) / 1_000_000.0
    except Exception:
        return None


class SystemMonitor:
    def __init__(self, power_path=None, interval=MONITOR_INTERVAL_S):
        self.power_path = power_path
        self.interval = interval
        self.cpu_pct = []
        self.ram_mb = []
        self.power_w = []
        self._running = False
        self._thread = None
        self._proc = psutil.Process()

    def start(self):
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
            if self.power_path:
                w = read_power_watts(self.power_path)
                if w is not None:
                    self.power_w.append(w)
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
        result = {
            'cpu_percent': stats(self.cpu_pct, 'CPU'),
            'ram_mb': stats(self.ram_mb, 'RAM'),
        }
        if self.power_w:
            result['power_watts'] = stats(self.power_w, 'power')
        else:
            result['power_watts'] = 'not_available — no sysfs power file found'
        return result


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


def identify_output_tensors(output_details):
    boxes_idx = classes_idx = scores_idx = num_det_idx = None
    for od in output_details:
        shape = tuple(od['shape'])
        name = od['name'].lower()
        idx = od['index']
        if len(shape) == 3 and shape[2] == 4:
            boxes_idx = idx
        elif shape == (1,):
            num_det_idx = idx
        elif len(shape) == 2:
            if any(k in name for k in ('score', 'confidence')):
                scores_idx = idx
            elif any(k in name for k in ('class', 'label', 'categor')):
                classes_idx = idx
            else:
                if scores_idx is None:
                    scores_idx = idx
                else:
                    classes_idx = idx
    missing = [k for k, v in {
        'boxes': boxes_idx, 'classes': classes_idx,
        'scores': scores_idx, 'num_det': num_det_idx
    }.items() if v is None]
    if missing:
        print(f'warn: could not identify tensors: {missing} - using positional fallback')
        if len(output_details) == 4:
            boxes_idx = output_details[1]['index']
            classes_idx = output_details[2]['index']
            scores_idx = output_details[0]['index']
            num_det_idx = output_details[3]['index']
    return {
        'boxes_idx': boxes_idx,
        'classes_idx': classes_idx,
        'scores_idx': scores_idx,
        'num_det_idx': num_det_idx,
    }


def main():
    print('VIM3 NPU Benchmark - EfficientDet-Lite2 INT8')
    print(f'model:    {MODEL_PATH}')
    print(f'delegate: {DELEGATE_PATH}')
    print()

    for p in [MODEL_PATH, DELEGATE_PATH]:
        if not Path(p).exists():
            print(f'fatal: not found: {p}')
            sys.exit(1)

    power_path = find_power_sysfs()
    if not power_path:
        print('warn: no power sysfs readable - power metric skipped.')

    print('\nstep 1: loading model + delegate ...')
    t_load_start = time.perf_counter()
    try:
        delegate = load_delegate(DELEGATE_PATH)
        interpreter = Interpreter(model_path=MODEL_PATH, experimental_delegates=[delegate])
        interpreter.allocate_tensors()
    except Exception as e:
        print(f'fatal: model load failed: {e}')
        raise
    load_time_ms = (time.perf_counter() - t_load_start) * 1000
    print(f'  -> {load_time_ms:.1f} ms')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print('\ninput tensor:')
    for d in input_details:
        print(f"  idx={d['index']}  shape={list(d['shape'])}  dtype={d['dtype'].__name__}")
    print('output tensors:')
    for d in output_details:
        print(f"  idx={d['index']}  name={d['name']}  shape={list(d['shape'])}  dtype={d['dtype'].__name__}")

    input_idx = input_details[0]['index']
    input_dtype = input_details[0]['dtype']
    out_map = identify_output_tensors(output_details)
    print(f'tensor map: {out_map}')

    rng = np.random.default_rng(42)
    dummy = rng.integers(0, 256, (1, INPUT_H, INPUT_W, 3), dtype=input_dtype)

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

    print(f'\nstep 3: steady-state benchmark ({NUM_RUNS} runs) ...')
    monitor = SystemMonitor(power_path=power_path)
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
            print(f'  ... {i+1}/{NUM_RUNS}  last invoke={invoke_ms_list[-1]:.1f} ms  e2e={e2e_ms_list[-1]:.1f} ms')

    bench_elapsed = time.perf_counter() - bench_t0
    monitor.stop()
    throughput_fps = NUM_RUNS / bench_elapsed

    results = {
        'platform': 'Khadas VIM3 Basic — Amlogic A311D NPU (TIM-VX)',
        'model': 'EfficientDet-Lite2 INT8 — model_run3.tflite',
        'input_shape': [1, INPUT_H, INPUT_W, 3],
        'num_warmup': NUM_WARMUP,
        'num_runs': NUM_RUNS,
        'load_time_ms': round(load_time_ms, 2),
        'warmup': {
            'invoke_ms_per_run': warmup_invoke_ms,
            'e2e_ms_per_run': warmup_e2e_ms,
            'note': 'first run typically slowest (JIT/cache cold start)',
        },
        'steady_state': {
            'invoke': timing_stats(invoke_ms_list, 'invoke'),
            'e2e': timing_stats(e2e_ms_list, 'e2e'),
            'throughput_fps': round(throughput_fps, 3),
            'total_bench_time_s': round(bench_elapsed, 3),
        },
        'system': monitor.summary(),
        'notes': {
            'nms_op': 'CUSTOM/NMS op falls back to CPU — expected for EfficientDet',
            'e2e_def': 'E2E = set_tensor + invoke + get_tensor (all outputs)',
            'invoke_def': 'invoke only = interpreter.invoke() call',
            'coral_reference': {
                'invoke_median_ms': 317.82,
                'e2e_median_ms': 375.61,
                'mAP_50': 0.2347,
            },
        },
    }

    print('\nRESULTS SUMMARY')
    inv = results['steady_state']['invoke']
    e2e = results['steady_state']['e2e']
    print(f'  load time:           {load_time_ms:.1f} ms')
    print(f'  warm-up invoke (ms): {warmup_invoke_ms}')
    print(f'  warm-up e2e    (ms): {warmup_e2e_ms}')
    print('\n  steady-state invoke (ms):')
    print(f"    median={inv['median_ms']}  mean={inv['mean_ms']}  std={inv['std_ms']}  min={inv['min_ms']}  max={inv['max_ms']}")
    print('\n  steady-state E2E (ms):')
    print(f"    median={e2e['median_ms']}  mean={e2e['mean_ms']}  std={e2e['std_ms']}  min={e2e['min_ms']}  max={e2e['max_ms']}")
    print(f'\n  throughput:          {throughput_fps:.3f} FPS')
    sys_s = results['system']
    cpu_s = sys_s.get('cpu_percent', {})
    if isinstance(cpu_s, dict) and 'mean' in cpu_s:
        print(f"  CPU usage:  mean={cpu_s['mean']}%  max={cpu_s['max']}%  (n={cpu_s['n_samples']} samples)")
    ram_s = sys_s.get('ram_mb', {})
    if isinstance(ram_s, dict) and 'mean' in ram_s:
        print(f"  RAM usage:  mean={ram_s['mean']} MB  max={ram_s['max']} MB")
    pw_s = sys_s.get('power_watts', 'not_available')
    if isinstance(pw_s, dict) and 'mean' in pw_s:
        print(f"  power:      mean={pw_s['mean']} W  max={pw_s['max']} W")
    else:
        print(f'  power:      {pw_s}')
    print(f'\n  coral ref:  invoke={317.82} ms  e2e={375.61} ms  mAP={0.2347}')
    print(f"  VIM3:       invoke={inv['median_ms']} ms  e2e={e2e['median_ms']} ms")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\ndone: results saved -> {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
