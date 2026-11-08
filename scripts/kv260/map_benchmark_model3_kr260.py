# kv260 accuracy reference: the integer-equivalent onnx graph converted from model_run3.tflite
# runs on: kv260
# output: /home/ubuntu/map_results_model3_orig.json

import os
import json
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

MODEL_PATH = '/home/ubuntu/model_run3.onnx'
VAIP_CONFIG = '/usr/bin/vaip_config.json'
ANCHOR_PATH = '/home/ubuntu/anchors_run3.npy'
VAL_IMAGES = '/home/ubuntu/val_full'
VAL_ANNOTS = '/home/ubuntu/val_annotations'
OUTPUT_JSON = '/home/ubuntu/map_results_model3_orig.json'
MAX_IMAGES = None
SCORE_THRESH = 0.25
IOU_THRESH = 0.50
INPUT_H = 448
INPUT_W = 448
NUM_CLASSES = 7
CLASS_NAMES = ['tl_red', 'tl_green', 'tl_yellow', 'pedes', 'car', 'truck', 't_sign']


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    boxes = []
    for obj in tree.iter('object'):
        n = obj.find('name').text
        if n not in CLASS_NAMES:
            continue
        bb = obj.find('bndbox')
        boxes.append({'name': n,
                      'box': [float(bb.find(x).text) for x in ('xmin', 'ymin', 'xmax', 'ymax')]})
    return boxes


def iou(a, b):
    xi = max(a[0], b[0])
    yi = max(a[1], b[1])
    xa = min(a[2], b[2])
    ya = min(a[3], b[3])
    inter = max(0, xa - xi) * max(0, ya - yi)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter) if (areaA + areaB - inter) > 0 else 0


def compute_ap(preds, n_gt):
    if n_gt == 0:
        return 0.0
    preds = sorted(preds, key=lambda x: x[0], reverse=True)
    tp = np.array([1 if p[1] else 0 for p in preds])
    fp = 1 - tp
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    prec = tp_c / (tp_c + fp_c)
    rec = tp_c / n_gt
    prec = np.concatenate([[1], prec, [0]])
    rec = np.concatenate([[0], rec, [1]])
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


import onnxruntime as ort

print('loading session...')
sess = ort.InferenceSession(
    MODEL_PATH,
    providers=['VitisAIExecutionProvider', 'CPUExecutionProvider'],
    provider_options=[{'config_file': VAIP_CONFIG}, {}]
)
inp_name = sess.get_inputs()[0].name
print(f'providers: {sess.get_providers()}')
print(f'input: {inp_name} {sess.get_inputs()[0].shape} {sess.get_inputs()[0].type}')

xml_files = sorted([f for f in os.listdir(VAL_ANNOTS) if f.endswith('.xml')])
if MAX_IMAGES:
    xml_files = xml_files[:MAX_IMAGES]
print(f'\nevaluating {len(xml_files)} images...')

preds_per = {c: [] for c in CLASS_NAMES}
gt_counts = {c: 0 for c in CLASS_NAMES}

for i, xf in enumerate(xml_files):
    img_path = os.path.join(VAL_IMAGES, xf.replace('.xml', '.jpg'))
    if not os.path.exists(img_path):
        continue
    gt = parse_voc_xml(os.path.join(VAL_ANNOTS, xf))
    for g in gt:
        gt_counts[g['name']] += 1

    img = np.array(Image.open(img_path).convert('RGB').resize((INPUT_W, INPUT_H)),
                   dtype=np.uint8)[np.newaxis]
    raw = sess.run(None, {inp_name: img})
    scores = raw[0][0]
    boxes = raw[1][0]
    n_det = int(raw[2][0])
    classes = raw[3][0].astype(np.int32)

    for j in range(min(n_det, 25)):
        if scores[j] < SCORE_THRESH:
            continue
        cls_idx = int(classes[j])
        if cls_idx >= NUM_CLASSES:
            continue
        cls_name = CLASS_NAMES[cls_idx]
        ymin, xmin, ymax, xmax = boxes[j]
        det = [xmin * 1280, ymin * 720, xmax * 1280, ymax * 720]
        best_iou = max((iou(det, g['box']) for g in gt if g['name'] == cls_name), default=0)
        preds_per[cls_name].append((float(scores[j]), best_iou >= IOU_THRESH))

    if (i + 1) % 100 == 0:
        print(f'  [{i+1}/{len(xml_files)}]')

print('\nstep 4: computing per-class AP@50 ...')
aps = {}
for c in CLASS_NAMES:
    ap = compute_ap(preds_per[c], gt_counts[c])
    aps[c] = ap
    print(f'  {c:<12} AP={ap:.4f}  (gt={gt_counts[c]}, preds={len(preds_per[c])})')
mAP = np.mean(list(aps.values()))
print(f'\nmAP@50 = {mAP:.4f}')
print('coral ref = 0.2347   VIM3 ref = 0.2278')
with open(OUTPUT_JSON, 'w') as f:
    json.dump({'mAP_50': float(mAP), 'ap_per_class': aps, 'n_images': len(xml_files)}, f)
print(f'\nresults saved -> {OUTPUT_JSON}')
