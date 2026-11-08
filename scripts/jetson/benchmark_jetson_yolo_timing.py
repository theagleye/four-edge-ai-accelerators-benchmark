# jetson nano timing of the yolov4-416 tensorrt fp16 engine: 10 warmup, 500 runs
# runs on: jetson nano, from the tensorrt_demos folder
# output: ~/THESIS/jetson_nano/jetson_yolo_timing_results.json

import time
import numpy as np
import pycuda.autoinit  # creates the cuda context
import sys
import os
sys.path.insert(0, os.path.expanduser('~/tensorrt_demos'))
from utils.yolo_with_plugins import TrtYOLO

MODEL = 'yolov4-416'
WARMUP = 10
ITERATIONS = 500
CATEGORY_NUM = 80


def main():
    os.chdir(os.path.expanduser('~/tensorrt_demos'))
    trt_yolo = TrtYOLO(MODEL, CATEGORY_NUM)
    dummy = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)

    print(f'warming up {WARMUP} iterations...')
    for _ in range(WARMUP):
        trt_yolo.detect(dummy)

    print(f'benchmarking {ITERATIONS} iterations...')
    times = []
    for i in range(ITERATIONS):
        t0 = time.perf_counter()
        trt_yolo.detect(dummy)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{ITERATIONS}')

    times = np.array(times)
    print('\nYOLOv4-416 TensorRT FP16 Synthetic Timing:')
    print(f'median:  {np.median(times):.3f} ms')
    print(f'mean:    {np.mean(times):.3f} ms')
    print(f'std:     {np.std(times):.3f} ms')
    print(f'min:     {np.min(times):.3f} ms')
    print(f'max:     {np.max(times):.3f} ms')
    print(f'FPS:     {1000/np.median(times):.3f}')

    import json
    results = {
        'model': MODEL,
        'backend': 'TensorRT FP16',
        'platform': 'Jetson Nano',
        'warmup': WARMUP,
        'iterations': ITERATIONS,
        'median_ms': float(np.median(times)),
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'fps': float(1000 / np.median(times))
    }
    out = os.path.expanduser('~/THESIS/jetson_nano/jetson_yolo_timing_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'saved to {out}')


if __name__ == '__main__':
    main()
