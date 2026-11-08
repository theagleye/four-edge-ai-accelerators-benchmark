# coral edge tpu end-to-end timing on 500 real validation images (seed 42)
# runs on: coral dev board mini
# output: /mnt/thesis/coral_realimage_e2e_results_run3.json

import time
import threading
import statistics
import json
import random
import sys
import os
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    print('fatal: pillow not found. Install: pip3 install Pillow')
    sys.exit(1)

try:
    from pycoral.utils.edgetpu import make_interpreter
except ImportError:
    print('fatal: pycoral not found.')
    sys.exit(1)

try:
    import psutil
except ImportError:
    print('fatal: psutil not found.')
    sys.exit(1)

# config
MODEL_PATH = '/mnt/thesis/model_run3_edgetpu.tflite'
VAL_IMAGES = '/mnt/thesis/run3/val/images'
INPUT_H = 448
INPUT_W = 448
NUM_IMAGES = 500
RANDOM_SEED = 42
MONITOR_INTERVAL_S = 0.1
OUTPUT_JSON = '/mnt/thesis/coral_realimage_e2e_results_run3.json'


class SystemMonitor:
    def __init__(self, interval=MONITOR_INTERVAL_S):
        self.interval = interval
        self.cpu_pct = []
        self.ram_mb = []
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
            time.sleep(self.interval)

    def summary(self):
        def stats(lst):
            if len(lst) < 2:
                return {'note': f'too few samples ({len(lst)})'}
            return {
                'mean': round(statistics.mean(lst), 4),
                'median': round(statistics.median(lst), 4),
                'std': round(statistics.stdev(lst), 4),
                'min': round(min(lst), 4),
                'max': round(max(lst), 4),
                'n_samples': len(lst),
            }
        return {
            'cpu_percent': stats(self.cpu_pct),
            'ram_mb': stats(self.ram_mb),
            'power_watts': 'not_available — measure with USB-C power meter',
        }


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


def main():
    print('  coral Dev Board Mini Real-Image E2E Benchmark')
    print(f'  model:  {MODEL_PATH}')
    print(f'  images: {VAL_IMAGES}')

    # preflight
    for p in [MODEL_PATH, VAL_IMAGES]:
        if not Path(p).exists():
            print(f'fatal: not found: {p}')
            sys.exit(1)

    # sample 500 images
    all_images = sorted([
        f for f in os.listdir(VAL_IMAGES)
        if f.lower().endswith('.jpg')
    ])
    if len(all_images) < NUM_IMAGES:
        print(f'fatal: only {len(all_images)} images found, need {NUM_IMAGES}')
        sys.exit(1)

    random.seed(RANDOM_SEED)
    selected = random.sample(all_images, NUM_IMAGES)
    print(f'\nselected {NUM_IMAGES} images (seed={RANDOM_SEED})')

    # load model
    print('\nstep 1: loading model + Edge TPU delegate ...')
    t_load_start = time.perf_counter()
    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    load_time_ms = (time.perf_counter() - t_load_start) * 1000
    print(f'  load time: {load_time_ms:.1f} ms')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_idx = input_details[0]['index']
    input_dtype = input_details[0]['dtype']

    # warm-up: 5 runs with first image
    print('\nstep 2: warm-up (5 runs with first image) ...')
    first_img = np.array(
        Image.open(os.path.join(VAL_IMAGES, selected[0]))
             .resize((INPUT_W, INPUT_H))
             .convert('RGB'),
        dtype=input_dtype
    )[np.newaxis, ...]

    warmup_invoke_ms = []
    warmup_e2e_ms = []
    for i in range(5):
        t_e2e0 = time.perf_counter()
        interpreter.set_tensor(input_idx, first_img)
        t_inv0 = time.perf_counter()
        interpreter.invoke()
        t_inv1 = time.perf_counter()
        for od in output_details:
            _ = interpreter.get_tensor(od['index'])
        t_e2e1 = time.perf_counter()
        warmup_invoke_ms.append(round((t_inv1 - t_inv0) * 1000, 2))
        warmup_e2e_ms.append(round((t_e2e1 - t_e2e0) * 1000, 2))
        print(f'  #{i+1}  invoke={warmup_invoke_ms[-1]:.1f} ms  e2e={warmup_e2e_ms[-1]:.1f} ms')

    # real-image e2e benchmark
    print(f'\nstep 3: real-image e2e benchmark ({NUM_IMAGES} images) ...')
    monitor = SystemMonitor()
    monitor.start()

    e2e_ms_list = []
    invoke_ms_list = []
    skipped = 0
    bench_t0 = time.perf_counter()

    for i, fname in enumerate(selected):
        try:
            t_e2e0 = time.perf_counter()

            img = np.array(
                Image.open(os.path.join(VAL_IMAGES, fname))
                     .resize((INPUT_W, INPUT_H))
                     .convert('RGB'),
                dtype=input_dtype
            )[np.newaxis, ...]

            interpreter.set_tensor(input_idx, img)

            t_inv0 = time.perf_counter()
            interpreter.invoke()
            t_inv1 = time.perf_counter()

            for od in output_details:
                _ = interpreter.get_tensor(od['index'])

            t_e2e1 = time.perf_counter()

            e2e_ms_list.append((t_e2e1 - t_e2e0) * 1000)
            invoke_ms_list.append((t_inv1 - t_inv0) * 1000)

        except Exception as ex:
            print(f'  warn: skipped {fname}: {ex}')
            skipped += 1

        if (i + 1) % 100 == 0:
            print(f'  [{i+1:4d}/{NUM_IMAGES}]  '
                  f'last invoke={invoke_ms_list[-1]:.1f} ms  '
                  f'last e2e={e2e_ms_list[-1]:.1f} ms')

    bench_elapsed = time.perf_counter() - bench_t0
    monitor.stop()
    throughput_fps = len(e2e_ms_list) / bench_elapsed

    # results
    results = {
        'platform': 'Coral Dev Board Mini — Google Edge TPU',
        'model': 'EfficientDet-Lite2 INT8 — model_run3_edgetpu.tflite',
        'benchmark_type': 'real_image_e2e',
        'input_shape': [1, INPUT_H, INPUT_W, 3],
        'num_images': NUM_IMAGES,
        'random_seed': RANDOM_SEED,
        'images_processed': len(e2e_ms_list),
        'images_skipped': skipped,
        'load_time_ms': round(load_time_ms, 2),
        'warmup': {
            'invoke_ms_per_run': warmup_invoke_ms,
            'e2e_ms_per_run': warmup_e2e_ms,
        },
        'steady_state': {
            'invoke': timing_stats(invoke_ms_list, 'invoke'),
            'e2e': timing_stats(e2e_ms_list, 'e2e'),
            'throughput_fps': round(throughput_fps, 3),
            'total_bench_time_s': round(bench_elapsed, 3),
        },
        'system': monitor.summary(),
        'notes': {
            'e2e_def': 'E2E = disk load + PIL resize + set_tensor + invoke + get_tensor',
            'invoke_def': 'invoke only = interpreter.invoke() call',
            'input_type': 'real BDD100K val images, resized to 448x448',
            'power_note': 'measure with USB-C power meter inline',
            'vim3_reference': {
                'invoke_median_ms': 1374.255,
                'e2e_median_ms': 1407.106,
                'throughput_fps': 0.711,
            },
        },
    }

    # print summary
    print('  RESULTS SUMMARY')
    inv = results['steady_state']['invoke']
    e2e = results['steady_state']['e2e']
    print(f'  load time:           {load_time_ms:.1f} ms')
    print(f'  warm-up invoke (ms): {warmup_invoke_ms}')
    print(f'  warm-up e2e    (ms): {warmup_e2e_ms}')
    print('\n  steady-state invoke (ms):')
    print(f"    median={inv['median_ms']}  mean={inv['mean_ms']}  "
          f"std={inv['std_ms']}  min={inv['min_ms']}  max={inv['max_ms']}")
    print('\n  steady-state E2E (ms):')
    print(f"    median={e2e['median_ms']}  mean={e2e['mean_ms']}  "
          f"std={e2e['std_ms']}  min={e2e['min_ms']}  max={e2e['max_ms']}")
    print(f'\n  throughput:          {throughput_fps:.3f} FPS')
    print(f'  images processed:    {len(e2e_ms_list)}/{NUM_IMAGES}')
    print(f'  skipped:             {skipped}')

    sys_s = results['system']
    cpu_s = sys_s.get('cpu_percent', {})
    if isinstance(cpu_s, dict) and 'mean' in cpu_s:
        print(f"\n  CPU usage:  mean={cpu_s['mean']}%  max={cpu_s['max']}%")
    ram_s = sys_s.get('ram_mb', {})
    if isinstance(ram_s, dict) and 'mean' in ram_s:
        print(f"  RAM usage:  mean={ram_s['mean']} MB  max={ram_s['max']} MB")
    print('\n   VIM3 reference ')
    print(f"  VIM3 invoke median: 1374.3 ms  |  Coral invoke median: {inv['median_ms']} ms")
    print(f"  VIM3 e2e median:    1407.1 ms  |  Coral e2e median:    {e2e['median_ms']} ms")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\ndone: results saved -> {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
