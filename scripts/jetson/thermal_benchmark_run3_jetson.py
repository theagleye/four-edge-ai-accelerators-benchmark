# jetson nano 500-run repeated-inference test: latency percentiles, cv, power, temperature
# runs on: jetson nano, from the tensorrt_demos folder
# output: ~/THESIS/jetson_nano/thermal_results_run3_jetson.json

import time
import json
import subprocess
import threading
import statistics
import re
import numpy as np
import pycuda.autoinit  # creates the cuda context
import sys
import os
sys.path.insert(0, os.path.expanduser('~/tensorrt_demos'))
from utils.yolo_with_plugins import TrtYOLO

MODEL = 'yolov4-416'
CATEGORY_NUM = 80
NUM_WARMUP = 10
NUM_RUNS = 500
OUT_JSON = os.path.expanduser('~/THESIS/jetson_nano/thermal_results_run3_jetson.json')
TEGRASTATS_INTERVAL = 500  # ms

teg_records = []
teg_stop = threading.Event()


def parse_tegrastats(line):
    rec = {}
    m = re.search(r'POM_5V_IN (\d+)/(\d+)', line)
    if m:
        rec['pom_5v_in_mw'] = int(m.group(1))
    m = re.search(r'POM_5V_GPU (\d+)/(\d+)', line)
    if m:
        rec['pom_5v_gpu_mw'] = int(m.group(1))
    m = re.search(r'CPU@([\d.]+)C', line)
    if m:
        rec['cpu_temp_c'] = float(m.group(1))
    m = re.search(r'GPU@([\d.]+)C', line)
    if m:
        rec['gpu_temp_c'] = float(m.group(1))
    m = re.search(r'thermal@([\d.]+)C', line)
    if m:
        rec['thermal_c'] = float(m.group(1))
    m = re.search(r'GR3D_FREQ (\d+)%', line)
    if m:
        rec['gpu_util_pct'] = int(m.group(1))
    rec['ts'] = time.perf_counter()
    return rec


def teg_reader(proc):
    for line in proc.stdout:
        if teg_stop.is_set():
            break
        rec = parse_tegrastats(line.strip())
        if rec:
            teg_records.append(rec)


def main():
    os.chdir(os.path.expanduser('~/tensorrt_demos'))
    print(f'thermal benchmark - {NUM_RUNS} runs, warmup={NUM_WARMUP}')

    trt_yolo = TrtYOLO(MODEL, CATEGORY_NUM)
    dummy = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)

    print(f'\nwarm-up ({NUM_WARMUP} runs) ...')
    warmup_ms = []
    for i in range(NUM_WARMUP):
        t0 = time.perf_counter()
        trt_yolo.detect(dummy)
        t1 = time.perf_counter()
        warmup_ms.append(round((t1 - t0) * 1000, 3))
        print(f'  #{i+1}  {warmup_ms[-1]:.2f}ms')

    print('\nstarting tegrastats ...')
    proc = subprocess.Popen(
        ['sudo', 'tegrastats', '--interval', str(TEGRASTATS_INTERVAL)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        universal_newlines=True, bufsize=1
    )
    teg_thread = threading.Thread(target=teg_reader, args=(proc,), daemon=True)
    teg_thread.start()
    time.sleep(1.0)

    print(f'\nrunning {NUM_RUNS} iterations ...')
    raw_ms = []
    t_bench_start = time.perf_counter()
    for i in range(NUM_RUNS):
        t0 = time.perf_counter()
        trt_yolo.detect(dummy)
        t1 = time.perf_counter()
        raw_ms.append(round((t1 - t0) * 1000, 3))
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{NUM_RUNS}]  last={raw_ms[-1]:.1f}ms')

    bench_s = time.perf_counter() - t_bench_start
    teg_stop.set()
    os.system('sudo pkill tegrastats')
    teg_thread.join(timeout=3)

    raw_sorted = sorted(raw_ms)
    p50 = statistics.median(raw_ms)
    p95 = raw_sorted[int(0.95 * len(raw_sorted))]
    p99 = raw_sorted[int(0.99 * len(raw_sorted))]
    mean = statistics.mean(raw_ms)
    std = statistics.stdev(raw_ms)
    cv = std / mean
    fps = NUM_RUNS / bench_s

    print('\nresults:')
    print(f'p50 : {p50:.3f}ms')
    print(f'p95 : {p95:.3f}ms')
    print(f'p99 : {p99:.3f}ms')
    print(f'mean: {mean:.3f}ms')
    print(f'std : {std:.3f}ms')
    print(f'CV  : {cv:.4f} ({cv*100:.2f}%)')
    print(f'fps : {fps:.3f}')

    pom_in = [r['pom_5v_in_mw'] for r in teg_records if 'pom_5v_in_mw' in r]
    pom_gpu = [r['pom_5v_gpu_mw'] for r in teg_records if 'pom_5v_gpu_mw' in r]
    cpu_temps = [r['cpu_temp_c'] for r in teg_records if 'cpu_temp_c' in r]
    gpu_temps = [r['gpu_temp_c'] for r in teg_records if 'gpu_temp_c' in r]

    print(f'\n power (n={len(pom_in)} samples) ')
    if pom_in:
        print(f'total  median={statistics.median(pom_in):.0f}mW  mean={statistics.mean(pom_in):.0f}mW')
        print(f'gpu    median={statistics.median(pom_gpu):.0f}mW  mean={statistics.mean(pom_gpu):.0f}mW')
    if cpu_temps:
        print(f'cpu temp: {min(cpu_temps):.1f}C - {max(cpu_temps):.1f}C  rise={max(cpu_temps)-cpu_temps[0]:.1f}C')
    if gpu_temps:
        print(f'gpu temp: {min(gpu_temps):.1f}C - {max(gpu_temps):.1f}C  rise={max(gpu_temps)-gpu_temps[0]:.1f}C')

    # latency histogram
    bin_min = int(min(raw_ms) // 10) * 10
    bin_max = int(max(raw_ms) // 10) * 10 + 10
    bins = list(range(bin_min, bin_max + 10, 10))
    hist = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        c = sum(1 for v in raw_ms if lo <= v < hi)
        if c > 0:
            hist[f'{lo}-{hi}ms'] = c

    results = {
        'platform': 'Jetson Nano — NVIDIA Maxwell GPU',
        'model': 'YOLOv4-416 TensorRT FP16',
        'test_type': 'thermal_throttling_500runs',
        'num_warmup': NUM_WARMUP,
        'num_runs': NUM_RUNS,
        'warmup_ms': warmup_ms,
        'steady_state': {
            'invoke': {
                'raw_ms': raw_ms,
                'p50_ms': round(p50, 3),
                'p95_ms': round(p95, 3),
                'p99_ms': round(p99, 3),
                'mean_ms': round(mean, 3),
                'std_ms': round(std, 3),
                'cv': round(cv, 4),
                'min_ms': round(min(raw_ms), 3),
                'max_ms': round(max(raw_ms), 3),
                'n': NUM_RUNS
            },
            'throughput_fps': round(fps, 3),
            'total_bench_time_s': round(bench_s, 3)
        },
        'power': {
            'pom_5v_in_median_mw': round(statistics.median(pom_in), 1) if pom_in else None,
            'pom_5v_in_mean_mw': round(statistics.mean(pom_in), 1) if pom_in else None,
            'pom_5v_gpu_median_mw': round(statistics.median(pom_gpu), 1) if pom_gpu else None,
            'n_samples': len(pom_in)
        },
        'temperature': {
            'cpu_min_c': round(min(cpu_temps), 1) if cpu_temps else None,
            'cpu_max_c': round(max(cpu_temps), 1) if cpu_temps else None,
            'cpu_rise_c': round(max(cpu_temps) - cpu_temps[0], 1) if cpu_temps else None,
            'gpu_min_c': round(min(gpu_temps), 1) if gpu_temps else None,
            'gpu_max_c': round(max(gpu_temps), 1) if gpu_temps else None,
            'gpu_rise_c': round(max(gpu_temps) - gpu_temps[0], 1) if gpu_temps else None,
        },
        'histogram_ms': hist
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nsaved -> {OUT_JSON}')


if __name__ == '__main__':
    main()
