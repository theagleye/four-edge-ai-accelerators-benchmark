# jetson nano end-to-end timing and tegrastats power on 500 coco val2017 images (seed 42)
# runs on: jetson nano, from the tensorrt_demos folder
# output: ~/THESIS/jetson_nano/realimg_e2e_results_run3_jetson.json

import time
import json
import subprocess
import threading
import statistics
import random
import glob
import re
import numpy as np
import cv2
import pycuda.autoinit  # creates the cuda context
import sys
import os
sys.path.insert(0, os.path.expanduser('~/tensorrt_demos'))
from utils.yolo_with_plugins import TrtYOLO

MODEL = 'yolov4-416'
CATEGORY_NUM = 80
NUM_WARMUP = 10
NUM_IMAGES = 500
RANDOM_SEED = 42
IMG_DIR = '/mnt/usb/coco/images/val2017'
OUT_JSON = os.path.expanduser('~/THESIS/jetson_nano/realimg_e2e_results_run3_jetson.json')
TEGRASTATS_INTERVAL = 500

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

    random.seed(RANDOM_SEED)
    all_imgs = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))
    if len(all_imgs) < NUM_IMAGES:
        print(f'fatal: only {len(all_imgs)} images found')
        return
    selected = random.sample(all_imgs, NUM_IMAGES)
    print(f'selected {NUM_IMAGES} images (seed={RANDOM_SEED})')

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

    print(f'\nreal-image e2e benchmark ({NUM_IMAGES} images) ...')
    e2e_ms = []
    invoke_ms = []
    skipped = 0
    t_bench_start = time.perf_counter()

    for i, img_path in enumerate(selected):
        try:
            t_e2e0 = time.perf_counter()
            img = cv2.imread(img_path)
            img = cv2.resize(img, (416, 416))
            t_inv0 = time.perf_counter()
            boxes, confs, clss = trt_yolo.detect(img)
            t_inv1 = time.perf_counter()
            t_e2e1 = time.perf_counter()
            e2e_ms.append((t_e2e1 - t_e2e0) * 1000)
            invoke_ms.append((t_inv1 - t_inv0) * 1000)
        except Exception as ex:
            print(f'  warn: skipped {img_path}: {ex}')
            skipped += 1

        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{NUM_IMAGES}]  last invoke={invoke_ms[-1]:.1f}ms  e2e={e2e_ms[-1]:.1f}ms')

    bench_s = time.perf_counter() - t_bench_start
    teg_stop.set()
    os.system('sudo pkill tegrastats')
    teg_thread.join(timeout=3)

    def stats(arr, label):
        s = sorted(arr)
        return {
            'median_ms': round(statistics.median(arr), 3),
            'mean_ms': round(statistics.mean(arr), 3),
            'std_ms': round(statistics.stdev(arr), 3),
            'p95_ms': round(s[int(0.95 * len(s))], 3),
            'p99_ms': round(s[int(0.99 * len(s))], 3),
            'min_ms': round(min(arr), 3),
            'max_ms': round(max(arr), 3),
            'cv': round(statistics.stdev(arr) / statistics.mean(arr), 4),
            'n': len(arr)
        }

    print('\nresults:')
    inv_s = stats(invoke_ms, 'invoke')
    e2e_s = stats(e2e_ms, 'e2e')
    print(f"invoke median={inv_s['median_ms']}ms  p95={inv_s['p95_ms']}ms  p99={inv_s['p99_ms']}ms")
    print(f"e2e    median={e2e_s['median_ms']}ms  p95={e2e_s['p95_ms']}ms  p99={e2e_s['p99_ms']}ms")
    print(f'fps={len(e2e_ms)/bench_s:.3f}  skipped={skipped}')

    pom_in = [r['pom_5v_in_mw'] for r in teg_records if 'pom_5v_in_mw' in r]
    pom_gpu = [r['pom_5v_gpu_mw'] for r in teg_records if 'pom_5v_gpu_mw' in r]
    cpu_temps = [r['cpu_temp_c'] for r in teg_records if 'cpu_temp_c' in r]
    gpu_temps = [r['gpu_temp_c'] for r in teg_records if 'gpu_temp_c' in r]

    results = {
        'platform': 'Jetson Nano — NVIDIA Maxwell GPU',
        'model': 'YOLOv4-416 TensorRT FP16',
        'benchmark_type': 'real_image_e2e',
        'num_images': NUM_IMAGES,
        'random_seed': RANDOM_SEED,
        'images_processed': len(e2e_ms),
        'images_skipped': skipped,
        'warmup_ms': warmup_ms,
        'invoke': inv_s,
        'e2e': e2e_s,
        'throughput_fps': round(len(e2e_ms) / bench_s, 3),
        'total_bench_time_s': round(bench_s, 3),
        'power': {
            'pom_5v_in_median_mw': round(statistics.median(pom_in), 1) if pom_in else None,
            'pom_5v_in_mean_mw': round(statistics.mean(pom_in), 1) if pom_in else None,
            'pom_5v_gpu_median_mw': round(statistics.median(pom_gpu), 1) if pom_gpu else None,
            'n_samples': len(pom_in)
        },
        'temperature': {
            'cpu_min_c': round(min(cpu_temps), 1) if cpu_temps else None,
            'cpu_max_c': round(max(cpu_temps), 1) if cpu_temps else None,
            'gpu_min_c': round(min(gpu_temps), 1) if gpu_temps else None,
            'gpu_max_c': round(max(gpu_temps), 1) if gpu_temps else None,
        },
        'notes': {
            'e2e_def': 'E2E = cv2.imread + cv2.resize + trt_yolo.detect',
            'invoke_def': 'invoke = trt_yolo.detect only',
            'power_tool': 'tegrastats (INA3221 onboard sensor)',
            'dataset': 'COCO val2017, 500 images, seed=42'
        }
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nsaved -> {OUT_JSON}')


if __name__ == '__main__':
    main()
