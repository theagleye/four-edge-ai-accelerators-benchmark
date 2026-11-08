# kv260 dpu per-class AP and mAP@0.50 on the 4,993-image validation set
# runs on: kv260 (vitis ai execution provider)
# output: /home/ubuntu/map_results_quark_200calib_kr260.json

import os
import sys
import time
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from PIL import Image
except ImportError:
    print('fatal: pillow not found.')
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print('fatal: onnxruntime not found.')
    sys.exit(1)

MODEL_PATH = '/home/ubuntu/model_run3_quark_200.onnx'
VAIP_CONFIG = '/usr/bin/vaip_config.json'
ANCHOR_PATH = '/home/ubuntu/anchors_run3.npy'
VAL_IMAGES = '/home/ubuntu/val_full'
VAL_ANNOTS = '/home/ubuntu/val_annotations'
OUTPUT_JSON = '/home/ubuntu/map_results_quark_200calib_kr260.json'
LOG_EVERY = 100
MAX_IMAGES = None

SCORE_THRESH = 0.25
IOU_THRESH = 0.50
INPUT_H = 448
INPUT_W = 448
NUM_CLASSES = 7
TOP_K = 500
NMS_IOU_THRESH = 0.5

CLASS_NAMES = ['tl_red', 'tl_green', 'tl_yellow', 'pedes', 'car', 'truck', 't_sign']


def decode_boxes(raw_cls, raw_box, anchors):
    cls_flat = []
    box_flat = []
    for i in range(5):
        cls_flat.append(raw_cls[i][0].reshape(-1, NUM_CLASSES))
        box_flat.append(raw_box[i][0].reshape(-1, 4))
    cls_all = np.concatenate(cls_flat, axis=0)
    box_all = np.concatenate(box_flat, axis=0)

    scores_all = 1.0 / (1.0 + np.exp(-cls_all))
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

    b = box_all[mask]
    sc = max_scores[mask]
    cl = max_classes[mask]
    a = anchors[mask]  # [cy, cx, h, w]

    cy = a[:, 0] + b[:, 0] * a[:, 2]
    cx = a[:, 1] + b[:, 1] * a[:, 3]
    h = a[:, 2] * np.exp(np.clip(b[:, 2], -4, 4))
    w = a[:, 3] * np.exp(np.clip(b[:, 3], -4, 4))

    boxes = np.stack([cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2], axis=1)
    boxes = np.clip(boxes, 0.0, 1.0)
    return boxes, sc, cl


def nms(boxes, scores):
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


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bb = obj.find('bndbox')
        xmin = float(bb.find('xmin').text)
        ymin = float(bb.find('ymin').text)
        xmax = float(bb.find('xmax').text)
        ymax = float(bb.find('ymax').text)
        boxes.append({'name': name, 'box': [xmin, ymin, xmax, ymax]})
    return boxes


def iou_boxes(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
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


def main():
    print('KR260 mAP@50 Benchmark - EfficientDet-Lite2 INT8 (VOE)')
    print(f'score thresh: {SCORE_THRESH}  IoU thresh: {IOU_THRESH}')
    print(f'val images:  {VAL_IMAGES}')
    print(f'val annots:  {VAL_ANNOTS}')
    print()

    for p in [MODEL_PATH, VAIP_CONFIG, ANCHOR_PATH, VAL_IMAGES, VAL_ANNOTS]:
        if not Path(p).exists():
            print(f'fatal: not found: {p}')
            sys.exit(1)

    print('step 1: loading anchors ...')
    anchors = np.load(ANCHOR_PATH)  # [cy, cx, h, w] - from TFLite model
    print(f'  anchors: {anchors.shape}  format: [cy, cx, h, w]')

    print('step 2: loading session ...')
    t0 = time.perf_counter()
    opts = ort.SessionOptions()
    sess = ort.InferenceSession(
        MODEL_PATH,
        sess_options=opts,
        providers=['VitisAIExecutionProvider', 'CPUExecutionProvider'],
        provider_options=[{'config_file': VAIP_CONFIG, }, {}],
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f'  load: {load_ms:.1f} ms')
    print(f'  providers: {sess.get_providers()}')

    inp_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    xml_files = sorted([f for f in os.listdir(VAL_ANNOTS) if f.endswith('.xml')])
    if MAX_IMAGES:
        xml_files = xml_files[:MAX_IMAGES]
        print(f'MAX_IMAGES={MAX_IMAGES}')
    print(f'\nstep 3: evaluating {len(xml_files)} images ...')

    all_preds = defaultdict(list)
    all_gt_counts = defaultdict(int)
    invoke_times_ms = []
    skipped = 0
    t_run = time.perf_counter()

    for i, xml_fn in enumerate(xml_files):
        img_fn = xml_fn.replace('.xml', '.jpg')
        img_path = os.path.join(VAL_IMAGES, img_fn)
        xml_path = os.path.join(VAL_ANNOTS, xml_fn)

        if not os.path.exists(img_path):
            skipped += 1
            continue

        gt_boxes = parse_voc_xml(xml_path)
        for gt in gt_boxes:
            all_gt_counts[gt['name']] += 1

        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        inp = np.array(img.resize((INPUT_W, INPUT_H), Image.BILINEAR), dtype=np.float32)[np.newaxis] / 255.0

        t_inv0 = time.perf_counter()
        raw = sess.run(None, {inp_name: inp})
        invoke_times_ms.append((time.perf_counter() - t_inv0) * 1000)

        cls_outs = [raw[j] for j, n in enumerate(out_names) if 'output_1' in n]
        box_outs = [raw[j] for j, n in enumerate(out_names) if 'output_2' in n]
        if len(cls_outs) != 5 or len(box_outs) != 5:
            cls_outs = raw[:5]
            box_outs = raw[5:]

        boxes, scores, classes = decode_boxes(cls_outs, box_outs, anchors)
        if len(boxes) == 0:
            continue

        keep = nms(boxes, scores)
        keep = keep[:25]  # match TFLite max_detections=25
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

        preds_img = []
        for k in range(len(boxes)):
            y1, x1, y2, x2 = boxes[k]
            preds_img.append({
                'class': CLASS_NAMES[classes[k]],
                'score': float(scores[k]),
                'box': [x1 * orig_w, y1 * orig_h, x2 * orig_w, y2 * orig_h],
            })

        gt_used = [False] * len(gt_boxes)
        preds_sorted = sorted(preds_img, key=lambda x: x['score'], reverse=True)
        for pred in preds_sorted:
            best_iou = 0
            best_idx = -1
            for k, gt in enumerate(gt_boxes):
                if gt_used[k] or gt['name'] != pred['class']:
                    continue
                v = iou_boxes(pred['box'], gt['box'])
                if v > best_iou:
                    best_iou = v
                    best_idx = k
            is_tp = best_iou >= IOU_THRESH
            if is_tp:
                gt_used[best_idx] = True
            all_preds[pred['class']].append((pred['score'], is_tp))

        if (i + 1) % LOG_EVERY == 0 or (i + 1) == len(xml_files):
            elapsed = time.perf_counter() - t_run
            fps = (i + 1) / elapsed
            eta = (len(xml_files) - i - 1) / fps if fps > 0 else 0
            print(f'  [{i+1:4d}/{len(xml_files)}]  elapsed={elapsed/60:.1f}min  fps={fps:.2f}  ETA={eta/60:.1f}min  last_invoke={invoke_times_ms[-1]:.1f}ms')

    total_elapsed = time.perf_counter() - t_run
    n_processed = len(xml_files) - skipped
    print(f'\ndone: {n_processed} images in {total_elapsed/60:.1f} min  ({skipped} skipped)')

    print('\nstep 4: computing per-class AP@50 ...')
    aps = []
    for cls in CLASS_NAMES:
        ap = compute_ap(all_preds[cls], all_gt_counts[cls])
        aps.append(ap)
        print(f'  {cls:<12}  AP={ap:.4f}  (gt={all_gt_counts[cls]}, preds={len(all_preds[cls])})')

    mAP = float(np.mean(aps))

    if len(invoke_times_ms) >= 2:
        inv_stats = {
            'median_ms': round(statistics.median(invoke_times_ms), 3),
            'mean_ms': round(statistics.mean(invoke_times_ms), 3),
            'std_ms': round(statistics.stdev(invoke_times_ms), 3),
            'min_ms': round(min(invoke_times_ms), 3),
            'max_ms': round(max(invoke_times_ms), 3),
            'n': len(invoke_times_ms),
        }
    else:
        inv_stats = {'note': 'too few samples'}

    conf_scores = {}
    for cls in CLASS_NAMES:
        scores_list = sorted([p[0] for p in all_preds[cls]], reverse=True)
        if scores_list:
            conf_scores[cls] = {
                'scores': scores_list,
                'mean': round(float(np.mean(scores_list)), 4),
                'median': round(float(np.median(scores_list)), 4),
                'std': round(float(np.std(scores_list)), 4),
                'n': len(scores_list),
            }
        else:
            conf_scores[cls] = {'scores': [], 'n': 0}

    results = {
        'platform': 'AMD Kria KR260 Robotics Starter Kit — DPUCZDX8G FPGA DPU',
        'model': 'EfficientDet-Lite2 INT8 — model_run3_quark_200.onnx (Quark 0.11.1, PowerOfTwoMethod.MinMSE, 200 calib imgs)',
        'dataset': f'BDD100K val — {n_processed} images',
        'methodology': 'matches Coral/VIM3 mAP benchmark exactly',
        'score_threshold': SCORE_THRESH,
        'iou_threshold': IOU_THRESH,
        'top_k_pre_nms': TOP_K,
        'max_detections': 25,
        'anchor_source': 'extracted from TFLite model stack tensor, format [cy,cx,h,w], verified pixel-perfect vs TFLite FP32',
        'mAP_50': round(mAP, 4),
        'ap_per_class': {
            cls: {
                'ap': round(compute_ap(all_preds[cls], all_gt_counts[cls]), 4),
                'n_gt': all_gt_counts[cls],
                'n_det': len(all_preds[cls]),
            }
            for cls in CLASS_NAMES
        },
        'confidence_scores_per_class': conf_scores,
        'invoke_timing': inv_stats,
        'total_inference_time_min': round(total_elapsed / 60, 2),
        'throughput_fps': round(n_processed / total_elapsed, 3),
        'skipped': skipped,
        'coral_reference': {'mAP_50': 0.2347, 'invoke_median_ms': 317.82},
        'vim3_reference': {'mAP_50': 0.2278, 'invoke_median_ms': 1374.14},
        'dpu_config': {
            'fingerprint': '0x101000016010407',
            'architecture': 'DPUCZDX8G_ISA1_B4096',
            'core_clock_mhz': 300,
            'dsp_clock_mhz': 600,
        },
    }

    print(f'\nmAP@50 = {mAP:.4f}')
    print('coral ref mAP@50 = 0.2347')
    print('VIM3  ref mAP@50 = 0.2278')

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'done: results saved -> {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
