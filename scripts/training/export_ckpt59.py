# exports checkpoint ckpt-59 of run3 to the int8 tflite model
# runs on: host pc
# output: run3/output/model_ckpt59.tflite (deployed as model_run3.tflite)

import os
from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import EfficientDetSpec

run_dir = '/home/eagle/THESIS/final_training_vastai/run3'
train_img = os.path.join(run_dir, 'train', 'images')
train_ann = os.path.join(run_dir, 'train', 'annotations')
out_dir = os.path.join(run_dir, 'output')
ckpt_path = os.path.join(out_dir, 'ckpts', 'ckpt-59')

labels = ['tl_red', 'tl_green', 'tl_yellow', 'pedes', 'car', 'truck', 't_sign']

train_data = object_detector.DataLoader.from_pascal_voc(train_img, train_ann, labels)

kaggle_uri = ('https://www.kaggle.com/models/tensorflow/efficientdet/'
              'tensorFlow2/lite2-feature-vector/1')
spec = EfficientDetSpec(
    model_name='efficientdet-lite2',
    uri=kaggle_uri,
    model_dir=os.path.join(out_dir, 'ckpts'),
    hparams={'max_instances_per_image': 100},
)

model = object_detector.create(
    train_data,
    model_spec=spec,
    batch_size=24,
    epochs=75,
    train_whole_model=True,
    validation_data=None,
    do_train=False,
)

model.model.load_weights(ckpt_path).expect_partial()
model.export(export_dir=out_dir, tflite_filename='model_ckpt59.tflite')

tflite_path = os.path.join(out_dir, 'model_ckpt59.tflite')
assert os.path.exists(tflite_path), 'export failed: no tflite'
size_mb = os.path.getsize(tflite_path) / 1e6
assert size_mb > 1, 'export failed: file too small'
print('model_ckpt59.tflite', round(size_mb, 1), 'MB')
