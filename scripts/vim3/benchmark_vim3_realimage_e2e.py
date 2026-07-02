
# vim3 npu - real-image end-to-end timing benchmark
# measures per-image: disk load + resize + set_tensor + invoke + get_tensor
# 500 images sampled, seed=42 
# always run inside screen while on ssh (not nohup - npu driver needs active tty)

import time
import threading
import statistics
import json
import random
import os
from pathlib import Path

import numpy as np
from PIL import Image
import psutil
from tflite_runtime.interpreter import Interpreter, load_delegate

MODEL_PATH = "/home/khadas/vx_tflite/mobilenet_v1/model_run3.tflite"
DELEGATE_PATH = "/home/khadas/vx_tflite/mobilenet_v1/libvx_delegate.so"
VAL_IMAGES = "/home/khadas/bdd100k/val_images"
INPUT_H = 448
INPUT_W = 448
NUM_IMAGES = 500
RANDOM_SEED = 42
MONITOR_INTERVAL_S = 0.1
OUTPUT_JSON = "/home/khadas/realimg_e2e_results_run3_vim3.json"


# Background thread samples psutil while the bench runs
class SystemMonitor:
    def __init__(self, interval=MONITOR_INTERVAL_S):
        self.interval = interval
        self.cpu_pct = []
        self.ram_mb = []
        self._running = False
        self._thread = None
        self._proc = psutil.Process()

    def start(self):
        # Prime psutil - first call always returns 0.0
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
                return {"note": f"too few samples ({len(lst)})"}
            return {
                "mean": round(statistics.mean(lst), 4),
                "median": round(statistics.median(lst), 4),
                "std": round(statistics.stdev(lst), 4),
                "min": round(min(lst), 4),
                "max": round(max(lst), 4),
                "n_samples": len(lst),
            }
        return {
            "cpu_percent": stats(self.cpu_pct),
            "ram_mb": stats(self.ram_mb),
            "power_watts": "not_available - measured externally via ATORCH J7-C",
        }


def timing_stats(times_ms, label=""):
    if len(times_ms) < 2:
        return {"note": f"too few samples for {label}"}
    return {
        "median_ms": round(statistics.median(times_ms), 3),
        "mean_ms": round(statistics.mean(times_ms), 3),
        "std_ms": round(statistics.stdev(times_ms), 3),
        "min_ms": round(min(times_ms), 3),
        "max_ms": round(max(times_ms), 3),
        "n": len(times_ms),
    }


print("VIM3 NPU Real-Image E2E Benchmark - EfficientDet-Lite2 INT8")
print(f"Model:  {MODEL_PATH}")
print(f"Images: {VAL_IMAGES}")
print()

for p in [MODEL_PATH, DELEGATE_PATH, VAL_IMAGES]:
    if not Path(p).exists():
        print(f"fatal: not found: {p}")
        raise SystemExit(1)

all_images = sorted([f for f in os.listdir(VAL_IMAGES) if f.lower().endswith(".jpg")])
if len(all_images) < NUM_IMAGES:
    print(f"fatal: only {len(all_images)} images found, need {NUM_IMAGES}")
    raise SystemExit(1)

random.seed(RANDOM_SEED)
selected = random.sample(all_images, NUM_IMAGES)
print(f"selected {NUM_IMAGES} images (seed={RANDOM_SEED})")

print("\nloading model + TIM-VX delegate ...")
t_load_start = time.perf_counter()
delegate = load_delegate(DELEGATE_PATH)
interpreter = Interpreter(model_path=MODEL_PATH, experimental_delegates=[delegate])
interpreter.allocate_tensors()
load_time_ms = (time.perf_counter() - t_load_start) * 1000
print(f"  load time: {load_time_ms:.1f} ms")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_idx = input_details[0]["index"]
input_dtype = input_details[0]["dtype"]

print("\nwarm-up (5 runs with first image) ...")
first_img = np.array(
    Image.open(os.path.join(VAL_IMAGES, selected[0]))
         .resize((INPUT_W, INPUT_H))
         .convert("RGB"),
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
        _ = interpreter.get_tensor(od["index"])
    t_e2e1 = time.perf_counter()
    warmup_invoke_ms.append(round((t_inv1 - t_inv0) * 1000, 2))
    warmup_e2e_ms.append(round((t_e2e1 - t_e2e0) * 1000, 2))
    print(f"  #{i+1}  invoke={warmup_invoke_ms[-1]:.1f} ms  e2e={warmup_e2e_ms[-1]:.1f} ms")

print(f"\nreal-image e2e benchmark ({NUM_IMAGES} images) ...")
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
                 .convert("RGB"),
            dtype=input_dtype
        )[np.newaxis, ...]
        interpreter.set_tensor(input_idx, img)
        t_inv0 = time.perf_counter()
        interpreter.invoke()
        t_inv1 = time.perf_counter()
        for od in output_details:
            _ = interpreter.get_tensor(od["index"])
        t_e2e1 = time.perf_counter()
        e2e_ms_list.append((t_e2e1 - t_e2e0) * 1000)
        invoke_ms_list.append((t_inv1 - t_inv0) * 1000)
    except Exception as ex:
        print(f"  warn: skipped {fname}: {ex}")
        skipped += 1

    if (i + 1) % 100 == 0:
        print(f"  [{i+1:4d}/{NUM_IMAGES}]  last invoke={invoke_ms_list[-1]:.1f} ms  last e2e={e2e_ms_list[-1]:.1f} ms")

bench_elapsed = time.perf_counter() - bench_t0
monitor.stop()
throughput_fps = len(e2e_ms_list) / bench_elapsed

results = {
    "platform": "Khadas VIM3 Basic - Amlogic A311D NPU (TIM-VX delegate)",
    "model": "EfficientDet-Lite2 INT8 - model_run3.tflite",
    "benchmark_type": "real_image_e2e",
    "input_shape": [1, INPUT_H, INPUT_W, 3],
    "num_images": NUM_IMAGES,
    "random_seed": RANDOM_SEED,
    "images_processed": len(e2e_ms_list),
    "images_skipped": skipped,
    "load_time_ms": round(load_time_ms, 2),
    "warmup": {
        "invoke_ms_per_run": warmup_invoke_ms,
        "e2e_ms_per_run": warmup_e2e_ms,
    },
    "steady_state": {
        "invoke": timing_stats(invoke_ms_list, "invoke"),
        "e2e": timing_stats(e2e_ms_list, "e2e"),
        "throughput_fps": round(throughput_fps, 3),
        "total_bench_time_s": round(bench_elapsed, 3),
    },
    "system": monitor.summary(),
    "notes": {
        "e2e_def": "E2E = disk load + PIL resize + set_tensor + invoke + get_tensor",
        "invoke_def": "invoke only = interpreter.invoke() call",
        "input_type": "real BDD100K val images, resized to 448x448",
        "power_note": "measured externally via ATORCH J7-C inline on USB-C power rail",
    },
}

print("\nRESULTS SUMMARY")
inv = results["steady_state"]["invoke"]
e2e = results["steady_state"]["e2e"]
print(f"  Load time:           {load_time_ms:.1f} ms")
print(f"  Warm-up invoke (ms): {warmup_invoke_ms}")
print(f"  Warm-up e2e    (ms): {warmup_e2e_ms}")
print(f"\n  Steady-state invoke (ms):")
print(f"    median={inv['median_ms']}  mean={inv['mean_ms']}  std={inv['std_ms']}  min={inv['min_ms']}  max={inv['max_ms']}")
print(f"\n  Steady-state E2E (ms):")
print(f"    median={e2e['median_ms']}  mean={e2e['mean_ms']}  std={e2e['std_ms']}  min={e2e['min_ms']}  max={e2e['max_ms']}")
print(f"\n  Throughput:          {throughput_fps:.3f} FPS")
print(f"  Images processed:    {len(e2e_ms_list)}/{NUM_IMAGES}")
print(f"  Skipped:             {skipped}")
sys_s = results["system"]
cpu_s = sys_s.get("cpu_percent", {})
if isinstance(cpu_s, dict) and "mean" in cpu_s:
    print(f"  CPU usage:  mean={cpu_s['mean']}%  max={cpu_s['max']}%")
ram_s = sys_s.get("ram_mb", {})
if isinstance(ram_s, dict) and "mean" in ram_s:
    print(f"  RAM usage:  mean={ram_s['mean']} MB  max={ram_s['max']} MB")

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved -> {OUTPUT_JSON}")