# jetson nano coco evaluation of yolov4-416 on the 5,000 val2017 images, per-class AP@0.50
# runs on: jetson nano, from the tensorrt_demos folder
# output: ~/THESIS/jetson_nano/map_results_run3_jetson.json, yolo_coco_results_run3.json

import os
import sys
import json
import statistics
import numpy as np
import cv2
import pycuda.autoinit  # creates the cuda context
sys.path.insert(0, os.path.expanduser('~/tensorrt_demos'))
from utils.yolo_with_plugins import TrtYOLO
from utils.yolo_classes import yolo_cls_to_ssd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from progressbar import progressbar

MODEL = 'yolov4-416'
CATEGORY_NUM = 80
IMGS_DIR = '/mnt/usb/coco/images/val2017'
ANNOTATIONS = '/mnt/usb/coco/annotations/instances_val2017.json'
RESULTS_FILE = os.path.expanduser('~/THESIS/jetson_nano/yolo_coco_results_run3.json')
OUT_JSON = os.path.expanduser('~/THESIS/jetson_nano/map_results_run3_jetson.json')


def main():
    os.chdir(os.path.expanduser('~/tensorrt_demos'))

    if not os.path.isfile('yolo/%s.trt' % MODEL):
        print(f'fatal: yolo/{MODEL}.trt not found')
        sys.exit(1)

    trt_yolo = TrtYOLO(MODEL, CATEGORY_NUM)
    jpgs = sorted([j for j in os.listdir(IMGS_DIR) if j.endswith('.jpg')])
    print(f'running inference on {len(jpgs)} images ...')

    results = []
    class_scores = {}

    for jpg in progressbar(jpgs):
        img = cv2.imread(os.path.join(IMGS_DIR, jpg))
        image_id = int(jpg.split('.')[0].split('_')[-1])
        boxes, confs, clss = trt_yolo.detect(img, conf_th=1e-2)
        for box, conf, cls in zip(boxes, confs, clss):
            x = float(box[0])
            y = float(box[1])
            w = float(box[2] - box[0] + 1)
            h = float(box[3] - box[1] + 1)
            cls_id = yolo_cls_to_ssd[int(cls)]
            results.append({
                'image_id': image_id,
                'category_id': cls_id,
                'bbox': [x, y, w, h],
                'score': float(conf)
            })
            if cls_id not in class_scores:
                class_scores[cls_id] = []
            class_scores[cls_id].append(float(conf))

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f)
    print(f'detections saved -> {RESULTS_FILE}')

    # COCO eval
    print('\nrunning COCO eval ...')
    coco_gt = COCO(ANNOTATIONS)
    coco_dt = coco_gt.loadRes(RESULTS_FILE)
    img_ids = sorted(coco_gt.getImgIds())
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # extract per-class AP@50
    # precision shape: [T=10, R=101, K=num_cats, A=4, M=3]
    # AP@50 = T index 0, all areas = A index 0, maxDets=100 = M index 2
    cat_ids = coco_eval.params.catIds
    cat_info = {c['id']: c['name'] for c in coco_gt.loadCats(cat_ids)}

    print('\nper-class AP@50:')
    per_class = {}
    for k, cat_id in enumerate(cat_ids):
        prec = coco_eval.eval['precision'][0, :, k, 0, 2]
        prec = prec[prec > -1]
        ap50 = float(np.mean(prec)) if len(prec) > 0 else 0.0
        name = cat_info.get(cat_id, str(cat_id))
        n_det = len(class_scores.get(cat_id, []))
        per_class[name] = {
            'cat_id': int(cat_id),
            'ap_50': round(ap50, 4),
            'n_det': int(n_det)
        }
        if ap50 > 0.01:
            print(f'  {name:<20} AP@50={ap50:.4f}  n_det={n_det}')

    # confidence score distribution
    print('\nconfidence distribution (top classes):')
    conf_dist = {}
    score_bins = [0.0, 0.25, 0.50, 0.75, 1.01]
    score_labels = ['0.00-0.25', '0.25-0.50', '0.50-0.75', '0.75-1.00']
    for cat_id, scores in class_scores.items():
        name = cat_info.get(cat_id, str(cat_id))
        if len(scores) < 10:
            continue
        bins_d = {}
        for lo, hi, lbl in zip(score_bins[:-1], score_bins[1:], score_labels):
            bins_d[lbl] = sum(1 for s in scores if lo <= s < hi)
        conf_dist[name] = {
            'n': len(scores),
            'mean': round(statistics.mean(scores), 4),
            'median': round(statistics.median(scores), 4),
            'std': round(statistics.stdev(scores), 4),
            'bins': bins_d
        }
    for name, d in sorted(conf_dist.items(), key=lambda x: -x[1]['n'])[:10]:
        print(f"  {name:<20} n={d['n']:6d}  mean={d['mean']:.3f}  bins={list(d['bins'].values())}")

    # overall metrics from summarize
    stats = coco_eval.stats
    out = {
        'platform': 'Jetson Nano — NVIDIA Maxwell GPU',
        'model': 'YOLOv4-416 TensorRT FP16',
        'dataset': 'COCO val2017',
        'num_images': len(jpgs),
        'AP_50_95': round(float(stats[0]), 4),
        'AP_50': round(float(stats[1]), 4),
        'AP_75': round(float(stats[2]), 4),
        'AP_small': round(float(stats[3]), 4),
        'AP_medium': round(float(stats[4]), 4),
        'AP_large': round(float(stats[5]), 4),
        'per_class_AP50': per_class,
        'confidence_distribution': conf_dist,
        'notes': {
            'model_note': 'YOLOv4-416 used — EfficientDet-Lite2 TRT conversion blocked on JetPack 4.6.1/TRT 8.0',
            'dataset_note': 'COCO val2017, 80 classes — not BDD100K',
            'conf_threshold_detection': '1e-2 (same as original eval_yolo.py)'
        }
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2, cls=NpEncoder)
    print(f'\nsaved -> {OUT_JSON}')


if __name__ == '__main__':
    main()
