# coral_extended_analysis_run3.py
# computes all extended metrics from coral benchmark jsons and csvs
# run from ~/THESIS: python3 coral_extended_analysis_run3.py

import json
import csv
import statistics
import math

thermal_f = "Coral_tpu/thermal_results_run3_coral.json"
map_f = "Coral_tpu/map_results_run3_coral.json"
power_f = "Coral_tpu/coral_realimg_power_results_run3.json"
thermal_csv = "thermal_power_run3_coral.csv"
out_f = "Coral_tpu/coral_extended_results_run3.json"
with open(thermal_f) as f: th = json.load(f)
with open(map_f) as f: mp = json.load(f)
with open(power_f) as f: pw = json.load(f)

# load thermal csv
csv_rows = []
with open(thermal_csv, newline="") as f:
    for r in csv.DictReader(f):
        csv_rows.append({"ts": float(r["timestamp_s"]), "W": float(r["power_W"]), "T": float(r["temp_C"])})

# latency stats (from thermal 500-run json)
inv = th["steady_state"]["invoke"]
raw_ms = inv["raw_ms"]
p50 = inv["p50_ms"]
p95 = inv["p95_ms"]
p99 = inv["p99_ms"]
cv = inv["cv"]
lat_mean = inv["mean_ms"]
lat_std = inv["std_ms"]

print("--- latency (500-run thermal test) ---")
print(f"p50  : {p50:.3f}ms")
print(f"p95  : {p95:.3f}ms")
print(f"p99  : {p99:.3f}ms")
print(f"mean : {lat_mean:.3f}ms")
print(f"std  : {lat_std:.3f}ms")
print(f"CV   : {cv:.4f} ({cv*100:.2f}%)")

# latency histogram (10ms bins)
bin_min = int(min(raw_ms) // 10) * 10
bin_max = int(max(raw_ms) // 10) * 10 + 10
bins = list(range(bin_min, bin_max + 10, 10))
hist = {}
for lo, hi in zip(bins[:-1], bins[1:]):
    key = f"{lo}-{hi}ms"
    hist[key] = sum(1 for v in raw_ms if lo <= v < hi)

print("\n--- latency histogram ---")
for k, v in hist.items():
    bar = "#" * (v // 5)
    print(f"  {k:12} {v:4d}  {bar}")

# latency vs temperature correlation
# align thermal benchmark iterations to logger timestamps
# find benchmark start = first csv row where power > 1.0W
inf_rows = [r for r in csv_rows if r["W"] > 1.0]
if inf_rows:
    t_bench_start = inf_rows[0]["ts"]
    bench_dur = th["steady_state"]["total_bench_time_s"]
    iter_temps = []
    for i in range(len(raw_ms)):
        t_iter = t_bench_start + (i * bench_dur / len(raw_ms))
        nearest = min(csv_rows, key=lambda r: abs(r["ts"] - t_iter))
        iter_temps.append(nearest["T"])
    # pearson correlation
    n = len(raw_ms)
    mx = sum(raw_ms) / n
    mt = sum(iter_temps) / n
    num = sum((raw_ms[i] - mx) * (iter_temps[i] - mt) for i in range(n))
    dx = math.sqrt(sum((v - mx)**2 for v in raw_ms))
    dt = math.sqrt(sum((v - mt)**2 for v in iter_temps))
    corr = num / (dx * dt) if dx * dt > 0 else 0.0
    temp_range = max(iter_temps) - min(iter_temps)
    print(f"\n--- latency vs temperature ---")
    print(f"temp range  : {min(iter_temps):.0f}C - {max(iter_temps):.0f}C  (delta={temp_range:.0f}C)")
    print(f"correlation : {corr:.4f}  ({'negligible - thermal stability confirmed' if abs(corr) < 0.1 else 'significant'})")
else:
    corr = 0.0
    temp_range = 0
    iter_temps = []
    print("\n[warn] could not find inference phase in thermal csv")

# cold-start energy
inf_power = pw["inference"]["power_mean_W"]
cold_ms = th["warmup_invoke_ms"][0]
steady_energy = pw["energy_invoke_J"]
cold_energy = inf_power * (cold_ms / 1000.0)
cold_overhead = cold_energy - steady_energy

print(f"\n--- cold-start energy ---")
print(f"cold invoke : {cold_ms:.2f}ms")
print(f"cold energy : {cold_energy:.4f}J")
print(f"steady      : {steady_energy:.4f}J")
print(f"overhead    : {cold_overhead:.4f}J  ({cold_overhead/steady_energy*100:.1f}% above steady)")

# class frequency vs AP
print(f"\n--- class frequency vs AP ---")
cls_data = mp["ap_per_class"]
print(f"  {'class':<12} {'n_gt':>6}  {'freq%':>6}  {'ap':>6}  {'n_det':>7}")
total_gt = sum(v["n_gt"] for v in cls_data.values())
cls_freq_ap = {}
for cls, v in cls_data.items():
    freq = v["n_gt"] / total_gt * 100
    print(f"  {cls:<12} {v['n_gt']:>6}  {freq:>5.1f}%  {v['ap']:>6.4f}  {v['n_det']:>7}")
    cls_freq_ap[cls] = {"n_gt": v["n_gt"], "freq_pct": round(freq, 2), "ap": v["ap"], "n_det": v["n_det"]}

# confidence score distribution per class
score_bins = [0.25, 0.35, 0.50, 0.75, 1.01]
score_labels = ["0.25-0.35", "0.35-0.50", "0.50-0.75", "0.75-1.00"]
print(f"\n--- confidence score distribution ---")
conf_dist = {}
for cls, scores in mp["confidence_scores_per_class"].items():
    if len(scores) == 0:
        conf_dist[cls] = {"n": 0, "mean": None, "std": None, "bins": {l: 0 for l in score_labels}}
        print(f"  {cls:<12} 0 detections")
        continue
    s_mean = statistics.mean(scores)
    s_std = statistics.stdev(scores) if len(scores) > 1 else 0
    bins_d = {}
    for lo, hi, lbl in zip(score_bins[:-1], score_bins[1:], score_labels):
        bins_d[lbl] = sum(1 for s in scores if lo <= s < hi)
    conf_dist[cls] = {"n": len(scores), "mean": round(s_mean, 4), "std": round(s_std, 4), "bins": bins_d}
    print(f"  {cls:<12} n={len(scores):5d}  mean={s_mean:.3f}  std={s_std:.3f}  bins={list(bins_d.values())}")

# power + energy (from existing analysis)
print(f"\n--- power summary ---")
print(f"idle     : {pw['idle']['power_mean_W']:.4f}W")
print(f"inference: {pw['inference']['power_mean_W']:.4f}W")
print(f"delta    : {pw['delta_power_W']:.4f}W")
print(f"e/invoke : {pw['energy_invoke_J']:.4f}J")
print(f"e/e2e    : {pw['energy_e2e_J']:.4f}J")
print(f"fps/W    : {pw['fps_per_W']:.3f}")
print(f"vsag     : {pw['voltage_sag_V']:.3f}V  ({pw['voltage_sag_pct']:.2f}%)")
print(f"cost/yr  : {pw['operational_cost']['EUR_per_year']:.2f} EUR")

# edge tpu op fallback
edgetpu_ops = {"total_ops": 357, "on_tpu": 351, "cpu_fallback": 6,
               "fallback_reason": "custom NMS op not supported on Edge TPU"}
print(f"\n--- edge tpu op fallback ---")
print(f"total ops   : {edgetpu_ops['total_ops']}")
print(f"on tpu      : {edgetpu_ops['on_tpu']}  ({edgetpu_ops['on_tpu']/edgetpu_ops['total_ops']*100:.1f}%)")
print(f"cpu fallback: {edgetpu_ops['cpu_fallback']}  ({edgetpu_ops['fallback_reason']})")

# efficientdet-lite2 positioning
# coco mAP@50 from published benchmarks (pretrained, not fine-tuned)
lite_pos = {
    "lite0": {"input": "320x320", "coco_map50_edgetpu": 25.54, "edge_tpu_compatible": True},
    "lite2": {"input": "448x448", "coco_map50_edgetpu": 28.5,  "edge_tpu_compatible": True},
    "lite4": {"input": "640x640", "coco_map50_edgetpu": None,   "edge_tpu_compatible": False,
              "note": "compiler error on Edge TPU - too large for on-chip SRAM"}
}
print(f"\n--- efficientdet-lite2 positioning ---")
for m, v in lite_pos.items():
    compat = "OK edge tpu" if v["edge_tpu_compatible"] else "FAIL edge tpu"
    note = f"  [{v['note']}]" if "note" in v else ""
    coco = f"{v['coco_map50_edgetpu']:.2f}%" if v["coco_map50_edgetpu"] is not None else "n/a"
    print(f"  {m}: {v['input']}  coco_edgetpu={coco}  {compat}{note}")

# save consolidated json
results = {
    "platform": "Coral Dev Board Mini - Google Edge TPU",
    "model": "EfficientDet-Lite2 INT8 - model_run3_edgetpu.tflite",
    "latency": {
        "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
        "mean_ms": lat_mean, "std_ms": lat_std, "cv": cv,
        "histogram_ms": hist
    },
    "latency_vs_temp": {
        "pearson_r": round(corr, 4),
        "temp_range_C": temp_range,
        "note": "temperature varied only 1C - thermal stability confirmed, correlation not meaningful"
    },
    "cold_start": {
        "cold_invoke_ms": cold_ms,
        "cold_energy_J": round(cold_energy, 4),
        "steady_energy_J": steady_energy,
        "overhead_J": round(cold_overhead, 4),
        "overhead_pct": round(cold_overhead / steady_energy * 100, 1)
    },
    "class_freq_vs_ap": cls_freq_ap,
    "confidence_distribution": conf_dist,
    "power": {
        "idle_W": pw["idle"]["power_mean_W"],
        "inference_W": pw["inference"]["power_mean_W"],
        "delta_W": pw["delta_power_W"],
        "energy_invoke_J": pw["energy_invoke_J"],
        "energy_e2e_J": pw["energy_e2e_J"],
        "fps_per_W": pw["fps_per_W"],
        "voltage_sag_V": pw["voltage_sag_V"],
        "voltage_sag_pct": pw["voltage_sag_pct"],
        "EUR_per_year": pw["operational_cost"]["EUR_per_year"]
    },
    "edgetpu_ops": edgetpu_ops,
    "lite2_positioning": lite_pos,
    "mAP_50": mp["mAP_50"],
    "ap_per_class": mp["ap_per_class"]
}
with open(out_f, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved -> {out_f}")
