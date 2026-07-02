
# vim3 npu - per-class mAP@50 on bdd100k val set
# efficientdet-lite2 INT8 (tim-vx delegate)
# some details:
#   - GT source:       pascal voc XML (pixel coords)
#   - class ID:        int(classes[j]) + 1 (model outputs 0-indexed)
#   - score threshold: 0.25
#   - IoU threshold:   0.50
# on ssh, always run inside screen (not nohup - npu driver needs active tty)

import os
import time
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter, load_delegate

MODEL_PATH = "/home/khadas/vx_tflite/mobilenet_v1/model_run3.tflite"
DELEGATE_PATH = "/home/khadas/vx_tflite/mobilenet_v1/libvx_delegate.so"
VAL_IMAGES = "/home/khadas/bdd100k/val_images"
VAL_ANNOTS = "/home/khadas/bdd100k/val_annotations"
OUTPUT_JSON = "/home/khadas/map_results_run3_vim3.json"
LOG_EVERY = 100
MAX_IMAGES = None

SCORE_THRESH = 0.25
IOU_THRESH = 0.50

CLASS_MAP = {
    1: "tl_red",
    2: "tl_green",
    3: "tl_yellow",
    4: "pedes",
    5: "car",
    6: "truck",
    7: "t_sign",
}
CLASS_NAMES = list(CLASS_MAP.values())


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        if name in CLASS_MAP.values():
            boxes.append({"name": name, "box": [xmin, ymin, xmax, ymax]})
    return boxes


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


def compute_ap(preds, n_gt):
    if n_gt == 0:
        return 0.0
    preds_sorted = sorted(preds, key=lambda x: x[0], reverse=True)
    tp = np.array([1 if p[1] else 0 for p in preds_sorted])
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_gt
    precision = np.concatenate([[1], precision, [0]])
    recall = np.concatenate([[0], recall, [1]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


# Some TIM-VX outputs aren't named, so we identify them by shape and index
def identify_output_tensors(output_details):
    boxes_idx = scores_idx = count_idx = classes_idx = None
    for od in output_details:
        shape = tuple(od["shape"])
        name = od["name"].lower()
        idx = od["index"]
        if len(shape) == 3 and shape[2] == 4:
            boxes_idx = idx
        elif shape == (1,):
            count_idx = idx
        elif len(shape) == 2:
            if any(k in name for k in ("score", "confidence")):
                scores_idx = idx
            elif any(k in name for k in ("class", "label", "categor")):
                classes_idx = idx
            else:
                if scores_idx is None:
                    scores_idx = idx
                else:
                    classes_idx = idx
    if None in (boxes_idx, scores_idx, count_idx, classes_idx):
        print("warn: name heuristic failed - using positional fallback.")
        scores_idx = output_details[0]["index"]
        boxes_idx = output_details[1]["index"]
        count_idx = output_details[2]["index"]
        classes_idx = output_details[3]["index"]
    return boxes_idx, scores_idx, count_idx, classes_idx


print("VIM3 mAP@50 Benchmark - EfficientDet-Lite2 INT8")
print(f"Score thresh: {SCORE_THRESH}  IoU thresh: {IOU_THRESH}")
print(f"Val images:  {VAL_IMAGES}")
print(f"Val annots:  {VAL_ANNOTS}")
print()

for p in [MODEL_PATH, DELEGATE_PATH, VAL_IMAGES, VAL_ANNOTS]:
    if not Path(p).exists():
        print(f"fatal: not found: {p}")
        raise SystemExit(1)

print("loading model + delegate ...")
t0 = time.perf_counter()
delegate = load_delegate(DELEGATE_PATH)
interpreter = Interpreter(model_path=MODEL_PATH, experimental_delegates=[delegate])
interpreter.allocate_tensors()
print(f"  → {(time.perf_counter()-t0)*1000:.1f} ms")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
inp = input_details[0]
in_h = inp["shape"][1]
in_w = inp["shape"][2]

print("\noutput tensors:")
for od in output_details:
    print(f"  idx={od['index']}  name={od['name']}  shape={list(od['shape'])}  dtype={od['dtype'].__name__}")

boxes_idx, scores_idx, count_idx, classes_idx = identify_output_tensors(output_details)
print(f"tensor map: scores={scores_idx}  boxes={boxes_idx}  count={count_idx}  classes={classes_idx}")

all_preds = defaultdict(list)
all_gt_counts = defaultdict(int)

xml_files = sorted([f for f in os.listdir(VAL_ANNOTS) if f.endswith(".xml")])
if MAX_IMAGES:
    xml_files = xml_files[:MAX_IMAGES]
    print(f"MAX_IMAGES={MAX_IMAGES} - sanity-check run")
print(f"\nevaluating {len(xml_files)} images ...")

t_run_start = time.perf_counter()
invoke_times_ms = []
skipped = 0

for i, xml_fn in enumerate(xml_files):
    img_fn = xml_fn.replace(".xml", ".jpg")
    img_path = os.path.join(VAL_IMAGES, img_fn)
    xml_path = os.path.join(VAL_ANNOTS, xml_fn)

    if not os.path.exists(img_path):
        skipped += 1
        continue

    gt_boxes = parse_voc_xml(xml_path)
    for gt in gt_boxes:
        all_gt_counts[gt["name"]] += 1

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    img_rs = img.resize((in_w, in_h), Image.BILINEAR)
    arr = np.array(img_rs, dtype=np.uint8)[np.newaxis]

    interpreter.set_tensor(inp["index"], arr)
    t_inv0 = time.perf_counter()
    interpreter.invoke()
    t_inv1 = time.perf_counter()
    invoke_times_ms.append((t_inv1 - t_inv0) * 1000)

    scores = interpreter.get_tensor(scores_idx)[0]
    boxes = interpreter.get_tensor(boxes_idx)[0]
    count = int(interpreter.get_tensor(count_idx)[0])
    classes = interpreter.get_tensor(classes_idx)[0]

    preds = []
    for j in range(count):
        if scores[j] < SCORE_THRESH:
            continue
        class_id = int(classes[j]) + 1
        if class_id not in CLASS_MAP:
            continue
        ymin, xmin, ymax, xmax = boxes[j]
        preds.append({
            "class": CLASS_MAP[class_id],
            "score": float(scores[j]),
            "box": [xmin*orig_w, ymin*orig_h, xmax*orig_w, ymax*orig_h],
        })

    gt_used = [False] * len(gt_boxes)
    preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
    for pred in preds_sorted:
        best_iou = 0
        best_idx = -1
        for k, gt in enumerate(gt_boxes):
            if gt_used[k] or gt["name"] != pred["class"]:
                continue
            iou_val = iou(pred["box"], gt["box"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = k
        is_tp = best_iou >= IOU_THRESH
        if is_tp:
            gt_used[best_idx] = True
        all_preds[pred["class"]].append((pred["score"], is_tp))

    if (i + 1) % LOG_EVERY == 0 or (i + 1) == len(xml_files):
        elapsed = time.perf_counter() - t_run_start
        fps = (i + 1) / elapsed
        eta = (len(xml_files) - i - 1) / fps if fps > 0 else 0
        print(f"  [{i+1:4d}/{len(xml_files)}]  elapsed={elapsed/60:.1f}min  fps={fps:.2f}  ETA={eta/60:.1f}min  last_invoke={invoke_times_ms[-1]:.1f}ms")

total_elapsed = time.perf_counter() - t_run_start
n_processed = len(xml_files) - skipped
print(f"\ndone: {n_processed} images in {total_elapsed/60:.1f} min  ({skipped} skipped)")

print("\ncomputing per-class AP@50 ...")
print("\nPer-class AP@50:")
aps = []
for cls in CLASS_NAMES:
    ap = compute_ap(all_preds[cls], all_gt_counts[cls])
    aps.append(ap)
    print(f"  {cls:<12}  AP={ap:.4f}  (gt={all_gt_counts[cls]}, preds={len(all_preds[cls])})")

mAP = float(np.mean(aps))

if len(invoke_times_ms) >= 2:
    inv_stats = {
        "median_ms": round(statistics.median(invoke_times_ms), 3),
        "mean_ms": round(statistics.mean(invoke_times_ms), 3),
        "std_ms": round(statistics.stdev(invoke_times_ms), 3),
        "min_ms": round(min(invoke_times_ms), 3),
        "max_ms": round(max(invoke_times_ms), 3),
        "n": len(invoke_times_ms),
    }
else:
    inv_stats = {"note": "too few samples"}

# confidence score distribution per class - matches Coral mAP JSON format
conf_scores = {}
for cls in CLASS_NAMES:
    scores_list = sorted([p[0] for p in all_preds[cls]], reverse=True)
    if scores_list:
        conf_scores[cls] = {
            "scores": scores_list,
            "mean": round(float(np.mean(scores_list)), 4),
            "median": round(float(np.median(scores_list)), 4),
            "std": round(float(np.std(scores_list)), 4),
            "n": len(scores_list),
        }
    else:
        conf_scores[cls] = {"scores": [], "n": 0}

results = {
    "platform": "Khadas VIM3 Basic - Amlogic A311D NPU (TIM-VX)",
    "model": "EfficientDet-Lite2 INT8 - model_run3.tflite",
    "dataset": f"BDD100K val - {n_processed} images",
    "methodology": "matches Coral mAP benchmark exactly",
    "score_threshold": SCORE_THRESH,
    "iou_threshold": IOU_THRESH,
    "mAP_50": round(mAP, 4),
    "ap_per_class": {
        cls: {
            "ap": round(compute_ap(all_preds[cls], all_gt_counts[cls]), 4),
            "n_gt": all_gt_counts[cls],
            "n_det": len(all_preds[cls]),
        }
        for cls in CLASS_NAMES
    },
    "confidence_scores_per_class": conf_scores,
    "invoke_timing": inv_stats,
    "total_inference_time_min": round(total_elapsed / 60, 2),
    "throughput_fps": round(n_processed / total_elapsed, 3),
    "skipped": skipped,
    "class_mapping": CLASS_MAP,
}

print(f"\nmAP@50 = {mAP:.4f}")

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nsaved -> {OUTPUT_JSON}")