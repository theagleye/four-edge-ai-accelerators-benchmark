
# kria kv260 fpga/dpu - efficientdet-lite2 INT8 benchmark (voe/vitisai EP)
# prerequisites (run before this script):
#   sudo su
#   export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
#   load DPU fabric:
#     sudo /usr/local/share/pynq-venv/bin/python3 -c "
#     import os; os.environ['XILINX_XRT']='/usr'
#     from pynq_dpu import DpuOverlay
#     DpuOverlay('/usr/local/share/pynq-venv/lib/python3.10/site-packages/pynq_dpu/dpu.bit')
#     print('DPU loaded')"
#   /etc/vart.conf must contain: firmware: /usr/lib/dpu.xclbin
#
# invoke def:  sess.run() call (DPU inference + I/O copy, no NMS)
# e2e def:     numpy alloc + sess.run() [synthetic]
#              PIL load + resize + normalize + sess.run() + NMS decode [real-image]
#
# note on comparability:
#   kv260 onnx model outputs 10 raw FPN tensors (no NMS).
#   coral/vim3 tflite models include NMS in the model graph.
#   => invoke is NOT directly comparable across platforms (disclosed in thesis).
#   => real-image e2e IS comparable: NMS decode included in kv260 e2e timing.

import glob
import time
import threading
import statistics
import json
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL_PATH = "/home/ubuntu/model_run3_quark_200.onnx"
VAIP_CONFIG = "/usr/bin/vaip_config.json"
VAL_DIR = "/home/ubuntu/val_images"
INPUT_H = 448
INPUT_W = 448
NUM_WARMUP = 5
NUM_RUNS = 500
NUM_REAL = 500
MONITOR_INTERVAL_S = 0.1
OUTPUT_JSON = "/home/ubuntu/benchmark_kv260_results_200calib_topk.json"

# efficientdet-lite2 anchor params (verified for 448x448, 5 fpn levels)
# strides: level 3..7 -> 8,16,32,64,112
ANCHOR_STRIDES = [8, 16, 32, 64, 112]
ANCHOR_SCALES = [2**0, 2**(1/3), 2**(2/3)]
ANCHOR_RATIOS = [1.0, 2.0, 0.5]  # h/w
ANCHOR_BASE_SCALE = 4.0
SCORE_THRESH = 0.05
TOP_K = 500
NMS_IOU_THRESH = 0.5
NUM_CLASSES = 7


def build_anchors(img_h=INPUT_H, img_w=INPUT_W):
    anchors = []
    for stride in ANCHOR_STRIDES:
        fh = int(np.ceil(img_h / stride))
        fw = int(np.ceil(img_w / stride))
        for yi in range(fh):
            for xi in range(fw):
                cy = (yi + 0.5) * stride / img_h
                cx = (xi + 0.5) * stride / img_w
                for s in ANCHOR_SCALES:
                    base = ANCHOR_BASE_SCALE * s * stride
                    for r in ANCHOR_RATIOS:
                        h = base * np.sqrt(r) / img_h
                        w = base / np.sqrt(r) / img_w
                        anchors.append([cy, cx, h, w])
    return np.array(anchors, dtype=np.float32)


def decode_boxes(raw_cls, raw_box, anchors):
    cls_flat = []
    box_flat = []
    for i in range(5):
        cls_flat.append(raw_cls[i][0].reshape(-1, NUM_CLASSES))
        box_flat.append(raw_box[i][0].reshape(-1, 4))
    cls_all = np.concatenate(cls_flat, axis=0)
    box_all = np.concatenate(box_flat, axis=0)

    scores_all = 1.0 / (1.0 + np.exp(-cls_all))  # sigmoid
    max_scores = scores_all.max(axis=1)
    max_classes = scores_all.argmax(axis=1)

    mask = max_scores >= SCORE_THRESH
    if mask.sum() == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32)
    if mask.sum() > TOP_K:
        topk_idx = np.argpartition(max_scores[mask], -TOP_K)[-TOP_K:]
        idx = np.where(mask)[0][topk_idx]
        mask = np.zeros(len(max_scores), dtype=bool)
        mask[idx] = True

    a = anchors[mask]  # (cy,cx,h,w)
    b = box_all[mask]  # (dy,dx,dh,dw)
    sc = max_scores[mask]
    cl = max_classes[mask]

    cy = a[:, 0] + b[:, 0] * a[:, 2]
    cx = a[:, 1] + b[:, 1] * a[:, 3]
    h = a[:, 2] * np.exp(np.clip(b[:, 2], -4, 4))
    w = a[:, 3] * np.exp(np.clip(b[:, 3], -4, 4))

    boxes = np.stack([cy - h/2, cx - w/2, cy + h/2, cx + w/2], axis=1)
    boxes = np.clip(boxes, 0.0, 1.0)
    return boxes, sc, cl


def nms(boxes, scores):
    # pure numpy NMS, boxes: [N,4] (y1,x1,y2,x2)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        iy1, ix1, iy2, ix2 = boxes[i]
        oy1 = np.maximum(iy1, boxes[order[1:], 0])
        ox1 = np.maximum(ix1, boxes[order[1:], 1])
        oy2 = np.minimum(iy2, boxes[order[1:], 2])
        ox2 = np.minimum(ix2, boxes[order[1:], 3])
        inter = np.maximum(0, oy2 - oy1) * np.maximum(0, ox2 - ox1)
        area_i = (iy2 - iy1) * (ix2 - ix1)
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                 (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-6)
        order = order[1:][iou <= NMS_IOU_THRESH]
    return keep


class SystemMonitor:
    def __init__(self, interval=MONITOR_INTERVAL_S):
        self.interval = interval
        self.cpu_pct = []
        self.ram_mb = []
        self._running = False
        self._thread = None
        try:
            import psutil
            self._psutil = psutil
            self._proc = psutil.Process()
        except ImportError:
            self._psutil = None

    def start(self):
        if not self._psutil:
            return
        self._psutil.cpu_percent(interval=None)
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
            if self._psutil:
                self.cpu_pct.append(self._psutil.cpu_percent(interval=None))
                self.ram_mb.append(self._proc.memory_info().rss / (1024**2))
            time.sleep(self.interval)

    def summary(self):
        def stats(lst, label):
            if len(lst) < 2:
                return {"note": f"too few samples ({len(lst)}) for {label}"}
            return {
                "mean": round(statistics.mean(lst), 4),
                "median": round(statistics.median(lst), 4),
                "std": round(statistics.stdev(lst), 4),
                "min": round(min(lst), 4),
                "max": round(max(lst), 4),
                "n_samples": len(lst),
            }
        return {
            "cpu_percent": stats(self.cpu_pct, "CPU") if self.cpu_pct else "psutil not available",
            "ram_mb": stats(self.ram_mb, "RAM") if self.ram_mb else "psutil not available",
            "power_watts": "measured via INA260 onboard sensor (sysfs)",
        }


def timing_stats(times_ms, label=""):
    if len(times_ms) < 2:
        return {"note": f"too few samples for {label}"}
    s = sorted(times_ms)
    n = len(s)
    p95 = s[int(0.95 * n)]
    p99 = s[int(0.99 * n)]
    mean = statistics.mean(times_ms)
    std = statistics.stdev(times_ms)
    return {
        "median_ms": round(statistics.median(times_ms), 3),
        "mean_ms": round(mean, 3),
        "std_ms": round(std, 3),
        "cv_pct": round(100 * std / mean, 4) if mean > 0 else None,
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "min_ms": round(min(times_ms), 3),
        "max_ms": round(max(times_ms), 3),
        "n": n,
    }


def load_session():
    opts = ort.SessionOptions()
    sess = ort.InferenceSession(
        MODEL_PATH,
        sess_options=opts,
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"config_file": VAIP_CONFIG}, {}],
    )
    return sess


def get_output_names(sess):
    return [o.name for o in sess.get_outputs()]


def parse_outputs(raw, out_names):
    # split 10 outputs into 5 cls + 5 box lists
    cls_outs = []
    box_outs = []
    for name, arr in zip(out_names, raw):
        if "output_1" in name:
            cls_outs.append(arr)
        else:
            box_outs.append(arr)
    return cls_outs, box_outs


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((INPUT_W, INPUT_H), Image.BILINEAR)
    return np.array(img, dtype=np.float32)[np.newaxis] / 255.0


print("kv260 fpga/dpu benchmark - efficientdet-lite2 INT8 (voe)")
print(f"model:   {MODEL_PATH}")
print(f"config:  {VAIP_CONFIG}")
print()

if not Path(MODEL_PATH).exists():
    print(f"fatal: model not found: {MODEL_PATH}")
    raise SystemExit(1)
if not Path(VAL_DIR).exists():
    print(f"fatal: val dir not found: {VAL_DIR}")
    raise SystemExit(1)

val_imgs = sorted(glob.glob(f"{VAL_DIR}/*.jpg"))[:NUM_REAL]
if len(val_imgs) < NUM_REAL:
    print(f"warn: only {len(val_imgs)} val images found (need {NUM_REAL})")

print("building anchors ...")
t0 = time.perf_counter()
anchors = build_anchors()
print(f"  anchors shape: {anchors.shape}  ({(time.perf_counter()-t0)*1000:.1f} ms)")

print("\nloading session ...")
t_load = time.perf_counter()
sess = load_session()
load_ms = (time.perf_counter() - t_load) * 1000
print(f"  load time: {load_ms:.1f} ms")
print(f"  providers: {sess.get_providers()}")

inp_name = sess.get_inputs()[0].name
out_names = get_output_names(sess)
print(f"  input:  {inp_name}  {sess.get_inputs()[0].shape}")
print(f"  outputs: {len(out_names)}")

print(f"\nwarm-up ({NUM_WARMUP} runs) ...")
warmup_invoke_ms = []
warmup_e2e_ms = []
for i in range(NUM_WARMUP):
    t_e2e0 = time.perf_counter()
    inp = np.random.rand(1, INPUT_H, INPUT_W, 3).astype(np.float32)
    t_inv0 = time.perf_counter()
    raw = sess.run(None, {inp_name: inp})
    t_inv1 = time.perf_counter()
    for _ in raw:
        pass
    t_e2e1 = time.perf_counter()
    inv_ms = (t_inv1 - t_inv0) * 1000
    e2e_ms = (t_e2e1 - t_e2e0) * 1000
    warmup_invoke_ms.append(round(inv_ms, 2))
    warmup_e2e_ms.append(round(e2e_ms, 2))
    print(f"  #{i+1:02d}  invoke={inv_ms:8.2f} ms   e2e={e2e_ms:8.2f} ms")

print(f"\nsynthetic invoke benchmark ({NUM_RUNS} runs) ...")
monitor_syn = SystemMonitor()
monitor_syn.start()
invoke_ms_list = []
e2e_ms_list = []
bench_t0 = time.perf_counter()
for i in range(NUM_RUNS):
    t_e2e0 = time.perf_counter()
    inp = np.empty((1, INPUT_H, INPUT_W, 3), dtype=np.float32)
    t_inv0 = time.perf_counter()
    raw = sess.run(None, {inp_name: inp})
    t_inv1 = time.perf_counter()
    for _ in raw:
        pass
    t_e2e1 = time.perf_counter()
    invoke_ms_list.append((t_inv1 - t_inv0) * 1000)
    e2e_ms_list.append((t_e2e1 - t_e2e0) * 1000)
    if (i + 1) % 100 == 0:
        print(f"  ... {i+1}/{NUM_RUNS}  last invoke={invoke_ms_list[-1]:.1f} ms  e2e={e2e_ms_list[-1]:.1f} ms")
bench_elapsed = time.perf_counter() - bench_t0
monitor_syn.stop()
throughput_fps = NUM_RUNS / bench_elapsed

print(f"\nreal-image e2e benchmark ({len(val_imgs)} images) ...")
monitor_real = SystemMonitor()
monitor_real.start()
real_e2e_ms = []
real_preproc_ms = []
real_invoke_ms2 = []
real_postproc_ms = []
real_t0 = time.perf_counter()
for i, path in enumerate(val_imgs):
    t0 = time.perf_counter()

    t_pre0 = time.perf_counter()
    inp = preprocess(path)
    t_pre1 = time.perf_counter()

    t_inv0 = time.perf_counter()
    raw = sess.run(None, {inp_name: inp})
    t_inv1 = time.perf_counter()

    t_post0 = time.perf_counter()
    cls_outs, box_outs = parse_outputs(raw, out_names)
    boxes, scores, classes = decode_boxes(cls_outs, box_outs, anchors)
    if len(boxes) > 0:
        keep = nms(boxes, scores)
    t_post1 = time.perf_counter()

    t1 = time.perf_counter()
    real_preproc_ms.append((t_pre1 - t_pre0) * 1000)
    real_invoke_ms2.append((t_inv1 - t_inv0) * 1000)
    real_postproc_ms.append((t_post1 - t_post0) * 1000)
    real_e2e_ms.append((t1 - t0) * 1000)
    if (i + 1) % 100 == 0:
        print(f"  ... {i+1}/{len(val_imgs)}  e2e={real_e2e_ms[-1]:.1f} ms  "
              f"(pre={real_preproc_ms[-1]:.1f} inv={real_invoke_ms2[-1]:.1f} "
              f"post={real_postproc_ms[-1]:.1f})")
real_elapsed = time.perf_counter() - real_t0
monitor_real.stop()
real_fps = len(val_imgs) / real_elapsed

results = {
    "platform": "AMD Kria KV260 Vision AI Starter Kit - DPUCZDX8G FPGA DPU",
    "model": "EfficientDet-Lite2 INT8 - model_run3_quark_200.onnx (Quark 0.11.1, PowerOfTwoMethod.MinMSE, 50 calib imgs)",
    "quantization_note": "PRELIMINARY - 50 calib images. Final run uses 200 calib images.",
    "input_shape": [1, INPUT_H, INPUT_W, 3],
    "input_dtype": "float32",
    "num_warmup": NUM_WARMUP,
    "num_runs_synthetic": NUM_RUNS,
    "num_runs_real": len(val_imgs),
    "load_time_ms": round(load_ms, 2),
    "anchor_count": int(anchors.shape[0]),
    "warmup": {
        "invoke_ms_per_run": warmup_invoke_ms,
        "e2e_ms_per_run": warmup_e2e_ms,
    },
    "synthetic": {
        "invoke": timing_stats(invoke_ms_list, "invoke"),
        "e2e": timing_stats(e2e_ms_list, "e2e_synthetic"),
        "throughput_fps": round(throughput_fps, 3),
        "total_bench_time_s": round(bench_elapsed, 3),
        "system": monitor_syn.summary(),
        "e2e_def": "np.empty alloc + sess.run() + iterate outputs (no NMS)",
        "invoke_def": "sess.run() call only (DPU inference + I/O copy)",
    },
    "real_image": {
        "invoke": timing_stats(real_invoke_ms2, "invoke_real"),
        "e2e": timing_stats(real_e2e_ms, "e2e_real"),
        "preproc": timing_stats(real_preproc_ms, "preproc"),
        "postproc_nms": timing_stats(real_postproc_ms, "postproc_nms"),
        "throughput_fps": round(real_fps, 3),
        "total_time_s": round(real_elapsed, 3),
        "system": monitor_real.summary(),
        "e2e_def": "PIL load+resize+normalize + sess.run() + anchor decode + NMS",
        "invoke_def": "sess.run() only (within real-image pipeline)",
        "comparability_note": (
            "KV260 invoke excludes NMS (baked into Coral/VIM3 TFLite). "
            "Real-image e2e includes NMS and is comparable to Coral/VIM3 e2e."
        ),
    },
    "notes": {
        "dpu_fingerprint": "0x101000056010407",
        "voe_version": "2023.1-r3.5.0",
        "quark_version": "0.11.1",
        "calib_method": "PowerOfTwoMethod.MinMSE",
        "nms_impl": "pure numpy, iou_thresh=0.5, score_thresh=0.05",
        "anchor_config": {
            "strides": ANCHOR_STRIDES,
            "scales": [round(s, 4) for s in ANCHOR_SCALES],
            "ratios": ANCHOR_RATIOS,
            "anchor_base_scale": ANCHOR_BASE_SCALE,
        },
        "power": "measured via INA260 onboard sensor (sysfs)",
        "coral_ref": {"invoke_ms": 317.82, "e2e_ms": 375.61, "mAP50": 0.2347},
        "vim3_ref": {"invoke_ms": 1374.14, "e2e_ms": 1407.11, "mAP50": 0.2278},
    },
}

inv = results["synthetic"]["invoke"]
e2e_s = results["synthetic"]["e2e"]
e2e_r = results["real_image"]["e2e"]
inv_r = results["real_image"]["invoke"]
pre = results["real_image"]["preproc"]
post = results["real_image"]["postproc_nms"]

print("\nresults summary")
print(f"  load time:              {load_ms:.1f} ms")
print(f"  anchor count:           {anchors.shape[0]}")
print(f"\n  synthetic invoke (ms):  median={inv['median_ms']}  mean={inv['mean_ms']}  "
      f"std={inv['std_ms']}  cv={inv['cv_pct']}%  p95={inv['p95_ms']}  p99={inv['p99_ms']}")
print(f"  synthetic e2e (ms):     median={e2e_s['median_ms']}  mean={e2e_s['mean_ms']}")
print(f"  synthetic fps:          {throughput_fps:.3f}")
print(f"\n  real invoke (ms):       median={inv_r['median_ms']}  mean={inv_r['mean_ms']}  "
      f"std={inv_r['std_ms']}  cv={inv_r['cv_pct']}%")
print(f"  real preproc (ms):      median={pre['median_ms']}")
print(f"  real postproc+nms (ms): median={post['median_ms']}")
print(f"  real e2e (ms):          median={e2e_r['median_ms']}  mean={e2e_r['mean_ms']}  "
      f"p95={e2e_r['p95_ms']}  p99={e2e_r['p99_ms']}")
print(f"  real fps:               {real_fps:.3f}")
print(f"\n  coral ref:  invoke={317.82} ms  e2e={375.61} ms")
print(f"  vim3 ref:   invoke={1374.14} ms  e2e={1407.11} ms")
print(f"  kv260 syn:  invoke={inv['median_ms']} ms  e2e={e2e_s['median_ms']} ms")
print(f"  kv260 real: invoke={inv_r['median_ms']} ms  e2e={e2e_r['median_ms']} ms")

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved -> {OUTPUT_JSON}")