# run3 dataset split: bdd100k train labels to pascal voc xml, 7 classes
# samples 50,000 labeled images (seed 42), class-stratified 90/10 split, keeps up to 45,000 train / 5,000 val
# runs on: host pc
# output: run3/train and run3/val (images + annotations)

import json
import os
import shutil
import random
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString


# paths
labels_file = os.path.expanduser(
    '/workspace/bdd100k/labels/det_v2_train_release.json'
)
images_src = os.path.expanduser(
    '/workspace/bdd100k/bdd100k/bdd100k/images/100k/train'
)
run_dir = os.path.expanduser('/home/eagle/THESIS/final_training_vastai/run3')


# class mapping
wanted = {'traffic light', 'pedestrian', 'car', 'truck', 'traffic sign'}


def get_label(category, attributes):
    if category == 'traffic light':
        color = (attributes or {}).get('trafficLightColor', [])
        if isinstance(color, list) and len(color) >= 2:
            c = str(color[1]).upper()
            if c == 'G':
                return 'tl_green'
            if c == 'R':
                return 'tl_red'
            if c == 'Y':
                return 'tl_yellow'
        return None  # skip if color is missing or unrecognized
    mapping = {
        'pedestrian': 'pedes',
        'car': 'car',
        'truck': 'truck',
        'traffic sign': 't_sign',
    }
    return mapping.get(category)


labels = ['tl_red', 'tl_green', 'tl_yellow', 'pedes', 'car', 'truck', 't_sign']


# load annotations
print('loading BDD100K annotations ...')
with open(labels_file) as f:
    entries = json.load(f)
print(f'  total entries: {len(entries)}')


# filter to images that have at least one usable box
print('filtering ...')
usable = []
for entry in entries:
    raw_labels = entry.get('labels') or []
    boxes = []
    for lbl in raw_labels:
        cat = lbl.get('category', '')
        if cat not in wanted:
            continue
        box = lbl.get('box2d')
        if box is None:
            continue
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        if x2 <= x1 or y2 <= y1:
            continue
        label = get_label(cat, lbl.get('attributes'))
        if label is None:
            continue
        boxes.append((label, x1, y1, x2, y2))
    if boxes:
        img_path = os.path.join(images_src, entry['name'])
        if os.path.exists(img_path):
            usable.append((entry['name'], boxes))

print(f'  images with usable boxes: {len(usable)}')


# sample 50,000 and split (stratified by class)
random.seed(42)
picked = random.sample(usable, min(50000, len(usable)))

# group images by their class set so every class appears in val
from collections import defaultdict
class_buckets = defaultdict(list)
for img in picked:
    classes_in_img = tuple(sorted(set(b[0] for b in img[1])))
    class_buckets[classes_in_img].append(img)

# take 10% of each bucket for val
train_set, val_set = [], []
for bucket in class_buckets.values():
    random.shuffle(bucket)
    n_val = max(1, len(bucket) // 10)
    val_set.extend(bucket[:n_val])
    train_set.extend(bucket[n_val:])

# trim to target sizes
random.shuffle(train_set)
random.shuffle(val_set)
train_set = train_set[:45000]
val_set = val_set[:5000]
print(f'  train: {len(train_set)}  val: {len(val_set)}')


# write pascal voc xml
def write_xml(filename, width, height, boxes, out_path):
    ann = Element('annotation')
    SubElement(ann, 'filename').text = filename
    sz = SubElement(ann, 'size')
    SubElement(sz, 'width').text = str(width)
    SubElement(sz, 'height').text = str(height)
    SubElement(sz, 'depth').text = '3'
    for (label, x1, y1, x2, y2) in boxes:
        obj = SubElement(ann, 'object')
        SubElement(obj, 'name').text = label
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'difficult').text = '0'
        bb = SubElement(obj, 'bndbox')
        SubElement(bb, 'xmin').text = str(int(x1))
        SubElement(bb, 'ymin').text = str(int(y1))
        SubElement(bb, 'xmax').text = str(int(x2))
        SubElement(bb, 'ymax').text = str(int(y2))
    pretty = parseString(tostring(ann)).toprettyxml(indent='  ')
    with open(out_path, 'w') as f:
        f.write(pretty)


# copy images and write xmls
img_w, img_h = 1280, 720

for split, data in [('train', train_set), ('val', val_set)]:
    imgs_out = os.path.join(run_dir, split, 'images')
    anns_out = os.path.join(run_dir, split, 'annotations')
    os.makedirs(imgs_out, exist_ok=True)
    os.makedirs(anns_out, exist_ok=True)

    print(f'  writing [{split}] ...')
    for i, (fname, boxes) in enumerate(data):
        shutil.copy2(os.path.join(images_src, fname),
                     os.path.join(imgs_out, fname))
        xml_name = os.path.splitext(fname)[0] + '.xml'
        write_xml(fname, img_w, img_h, boxes,
                  os.path.join(anns_out, xml_name))
        if (i + 1) % 5000 == 0:
            print(f'    {i + 1} / {len(data)} done ...')

    print(f'  [{split}] done - {len(data)} images + XMLs -> {run_dir}/{split}/')

print('\nall done. Ready to run train_full.py')
