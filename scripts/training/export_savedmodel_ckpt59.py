# exports checkpoint ckpt-59 of run3 to a keras savedmodel, the input for tf2onnx on the kv260 path
# runs on: host pc
# output: run3/output/saved_model_ckpt59

import os
from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import EfficientDetSpec

run_dir = '/home/eagle/THESIS/final_training_vastai/run3'
train_img = os.path.join(run_dir, 'train', 'images')
train_ann = os.path.join(run_dir, 'train', 'annotations')
out_dir = os.path.join(run_dir, 'output')
ckpt_path = os.path.join(out_dir, 'ckpts', 'ckpt-59')
saved_model_dir = os.path.join(out_dir, 'saved_model_ckpt59')

labels = ['tl_red', 'tl_green', 'tl_yellow', 'pedes', 'car', 'truck', 't_sign']

kaggle_uri = (
    'https://www.kaggle.com/models/tensorflow/efficientdet/'
    'tensorFlow2/lite2-feature-vector/1'
)
spec = EfficientDetSpec(
    model_name='efficientdet-lite2',
    uri=kaggle_uri,
    model_dir=os.path.join(out_dir, 'ckpts'),
    hparams={'max_instances_per_image': 100},
)

print('loading train data...')
train_data = object_detector.DataLoader.from_pascal_voc(
    train_img, train_ann, labels
)
print(f'  train: {len(train_data)} samples')

print('building model structure (do_train=False)...')
model = object_detector.create(
    train_data,
    model_spec=spec,
    batch_size=24,
    epochs=75,
    train_whole_model=True,
    validation_data=None,
    do_train=False,
)

print(f'loading weights from {ckpt_path}...')
model.model.load_weights(ckpt_path).expect_partial()
print('weights loaded.')

print('building model graph with dummy input...')
import numpy as np
_ = model.model(np.zeros((1, 448, 448, 3), dtype=np.float), training=False)
print('graph built.')

print(f'saving Keras SavedModel to {saved_model_dir}...')
model.model.save(saved_model_dir, save_format='tf')

if os.path.exists(saved_model_dir):
    size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(saved_model_dir)
        for f in fn
    ) / 1e6
    print(f'\ndone - saved_model_ckpt59/ is {size:.1f} MB')
    print(f'path: {saved_model_dir}')
else:
    print('export failed - directory not found')
