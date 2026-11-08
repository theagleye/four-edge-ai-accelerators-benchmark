# jetson nano idle power baseline: 90 s of tegrastats
# runs on: jetson nano
# output: ~/THESIS/jetson_nano/idle_power_results_run3_jetson.json

import time
import json
import subprocess
import threading
import statistics
import re
import os

OUT_JSON = os.path.expanduser('~/THESIS/jetson_nano/idle_power_results_run3_jetson.json')
DURATION_S = 90
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
    return rec


def teg_reader(proc):
    for line in proc.stdout:
        if teg_stop.is_set():
            break
        rec = parse_tegrastats(line.strip())
        if rec:
            teg_records.append(rec)


def main():
    print(f'idle power baseline - {DURATION_S}s logging ...')

    proc = subprocess.Popen(
        ['sudo', 'tegrastats', '--interval', str(TEGRASTATS_INTERVAL)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        universal_newlines=True, bufsize=1
    )
    teg_thread = threading.Thread(target=teg_reader, args=(proc,), daemon=True)
    teg_thread.start()

    for i in range(DURATION_S):
        time.sleep(1.0)
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{DURATION_S}s ...')

    teg_stop.set()
    os.system('sudo pkill tegrastats')
    teg_thread.join(timeout=3)

    pom_in = [r['pom_5v_in_mw'] for r in teg_records if 'pom_5v_in_mw' in r]
    pom_gpu = [r['pom_5v_gpu_mw'] for r in teg_records if 'pom_5v_gpu_mw' in r]
    cpu_temps = [r['cpu_temp_c'] for r in teg_records if 'cpu_temp_c' in r]
    gpu_temps = [r['gpu_temp_c'] for r in teg_records if 'gpu_temp_c' in r]

    print(f'\n idle power (n={len(pom_in)} samples) ')
    if pom_in:
        print(f'total  median={statistics.median(pom_in):.0f}mW  mean={statistics.mean(pom_in):.0f}mW  std={statistics.stdev(pom_in):.0f}mW')
        print(f'gpu    median={statistics.median(pom_gpu):.0f}mW')
    if cpu_temps:
        print(f'cpu temp: mean={statistics.mean(cpu_temps):.1f}C')
    if gpu_temps:
        print(f'gpu temp: mean={statistics.mean(gpu_temps):.1f}C')

    results = {
        'platform': 'Jetson Nano — NVIDIA Maxwell GPU',
        'benchmark_type': 'idle_power_baseline',
        'duration_s': DURATION_S,
        'n_samples': len(pom_in),
        'pom_5v_in_median_mw': round(statistics.median(pom_in), 1) if pom_in else None,
        'pom_5v_in_mean_mw': round(statistics.mean(pom_in), 1) if pom_in else None,
        'pom_5v_in_std_mw': round(statistics.stdev(pom_in), 1) if len(pom_in) > 1 else None,
        'pom_5v_gpu_median_mw': round(statistics.median(pom_gpu), 1) if pom_gpu else None,
        'cpu_temp_mean_c': round(statistics.mean(cpu_temps), 1) if cpu_temps else None,
        'gpu_temp_mean_c': round(statistics.mean(gpu_temps), 1) if gpu_temps else None,
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nsaved -> {OUT_JSON}')


if __name__ == '__main__':
    main()
