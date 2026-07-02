# analyzes realimg_power_run3_coral.csv alongside coral_realimage_e2e_results_run3.json
# separates idle vs inference phases by power threshold
# computes all thesis metrics: power, energy, voltage sag, temp, operational cost

import csv
import json
import statistics

POWER_CSV = "realimg_power_run3_coral.csv"
TIMING_JSON = "Coral_tpu/coral_realimage_e2e_results_run3.json"
OUT_JSON = "Coral_tpu/coral_realimg_power_results_run3.json"
THRESHOLD_W = 1.0  # idle below / inference above
FIN_PRICE = 0.10  # euros per kWh, Finnish average on 19.04.2026

# load power csv
rows = []
with open(POWER_CSV, newline="") as f:
    for r in csv.DictReader(f):
        rows.append({
            "ts": float(r["timestamp_s"]),
            "V": float(r["voltage_V"]),
            "A": float(r["current_A"]),
            "W": float(r["power_W"]),
            "T": float(r["temp_C"])
        })

t0 = rows[0]["ts"]
for r in rows:
    r["elapsed"] = r["ts"] - t0

# load timing json for invoke and e2e medians
with open(TIMING_JSON) as f:
    tj = json.load(f)
invoke_med_ms = tj["steady_state"]["invoke"]["median_ms"]
e2e_med_ms = tj["steady_state"]["e2e"]["median_ms"]

# split phases
idle_rows = [r for r in rows if r["W"] < THRESHOLD_W]
inf_rows = [r for r in rows if r["W"] >= THRESHOLD_W]

idle_W = [r["W"] for r in idle_rows]
inf_W = [r["W"] for r in inf_rows]
idle_V = [r["V"] for r in idle_rows]
inf_V = [r["V"] for r in inf_rows]
idle_T = [r["T"] for r in idle_rows]
inf_T = [r["T"] for r in inf_rows]

print(f"\n--- coral real-image power analysis ---")
print(f"total samples : {len(rows)}")
print(f"duration      : {rows[-1]['elapsed']:.1f}s")
print(f"threshold     : {THRESHOLD_W}W")

print(f"\n-- idle ({len(idle_rows)} samples) --")
print(f"power : mean={statistics.mean(idle_W):.3f}W  std={statistics.stdev(idle_W):.3f}W  min={min(idle_W):.3f}W  max={max(idle_W):.3f}W")
print(f"volt  : mean={statistics.mean(idle_V):.3f}V")
print(f"temp  : mean={statistics.mean(idle_T):.1f}C  max={max(idle_T):.1f}C")

print(f"\n-- inference ({len(inf_rows)} samples) --")
print(f"power : mean={statistics.mean(inf_W):.3f}W  std={statistics.stdev(inf_W):.3f}W  min={min(inf_W):.3f}W  max={max(inf_W):.3f}W")
print(f"volt  : mean={statistics.mean(inf_V):.3f}V  min={min(inf_V):.3f}V")
print(f"temp  : mean={statistics.mean(inf_T):.1f}C  max={max(inf_T):.1f}C")

# thesis metrics
idle_mean = statistics.mean(idle_W)
idle_std = statistics.stdev(idle_W) if len(idle_W) > 1 else 0
inf_mean = statistics.mean(inf_W)
inf_std = statistics.stdev(inf_W) if len(inf_W) > 1 else 0
v_idle = statistics.mean(idle_V)
v_inf = statistics.mean(inf_V)
voltage_sag_V = v_idle - v_inf
voltage_sag_pct = (voltage_sag_V / v_idle) * 100 if v_idle > 0 else 0

# energy per inference: mean inference power × invoke median time
e_invoke = (inf_mean * invoke_med_ms) / 1000
e_e2e = (inf_mean * e2e_med_ms) / 1000
fps = tj["steady_state"]["throughput_fps"]
fps_per_w = fps / inf_mean

# operational cost: assume continuous inference at measured inference power
wh_per_hour = inf_mean  # 1W for 1h = 1 Wh
wh_per_day = wh_per_hour * 24
cost_per_day = (wh_per_day / 1000) * FIN_PRICE
cost_per_yr = cost_per_day * 365

results = {
    "platform": "Coral Dev Board Mini - Google Edge TPU",
    "model": "EfficientDet-Lite2 INT8 - model_run3_edgetpu.tflite",
    "threshold_W": THRESHOLD_W,
    "n_samples_total": len(rows),
    "duration_s": round(rows[-1]["elapsed"], 3),
    "idle": {
        "n_samples": len(idle_rows),
        "power_mean_W": round(idle_mean, 4),
        "power_std_W": round(idle_std, 4),
        "power_min_W": round(min(idle_W), 4),
        "power_max_W": round(max(idle_W), 4),
        "voltage_mean_V": round(v_idle, 4),
        "temp_mean_C": round(statistics.mean(idle_T), 3),
        "temp_max_C": round(max(idle_T), 3),
    },
    "inference": {
        "n_samples": len(inf_rows),
        "power_mean_W": round(inf_mean, 4),
        "power_std_W": round(inf_std, 4),
        "power_min_W": round(min(inf_W), 4),
        "power_max_W": round(max(inf_W), 4),
        "voltage_mean_V": round(v_inf, 4),
        "temp_mean_C": round(statistics.mean(inf_T), 3),
        "temp_max_C": round(max(inf_T), 3),
    },
    "delta_power_W": round(inf_mean - idle_mean, 4),
    "voltage_sag_V": round(voltage_sag_V, 4),
    "voltage_sag_pct": round(voltage_sag_pct, 3),
    "invoke_median_ms": invoke_med_ms,
    "e2e_median_ms": e2e_med_ms,
    "energy_invoke_J": round(e_invoke, 6),
    "energy_e2e_J": round(e_e2e, 6),
    "fps": round(fps, 3),
    "fps_per_W": round(fps_per_w, 3),
    "operational_cost": {
        "power_W": round(inf_mean, 3),
        "Wh_per_hour": round(wh_per_hour, 3),
        "Wh_per_day": round(wh_per_day, 3),
        "EUR_per_day": round(cost_per_day, 4),
        "EUR_per_year": round(cost_per_yr, 2),
        "electricity_rate": f"{FIN_PRICE} EUR/kWh"
    },
    "methodology_note": (
        "Power measured at Coral Dev Board Mini dedicated USB-C power port via ATORCH J7-C. "
        "OTG port connected to host for SSH - small unmetered VBUS contribution from host not captured. "
        f"Idle/inference phases separated at {THRESHOLD_W}W threshold."
    )
}
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved -> {OUT_JSON}")