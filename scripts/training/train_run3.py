# trains efficientdet-lite2 run3 with tflite-model-maker: 75 epochs, batch 24, whole model
# runs on: rented cloud gpu
# output: run3/output (checkpoints ckpts/ckpt-N, model.tflite)

import os
import tensorflow as tf
from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import EfficientDetSpec

# paths
run_dir = '/home/eagle/THESIS/final_training_vastai/run3'
train_img = os.path.join(run_dir, 'train', 'images')
train_ann = os.path.join(run_dir, 'train', 'annotations')
val_img = os.path.join(run_dir, 'val', 'images')
val_ann = os.path.join(run_dir, 'val', 'annotations')
out_dir = os.path.join(run_dir, 'output')
os.makedirs(out_dir, exist_ok=True)

# class names
labels = ['tl_red', 'tl_green', 'tl_yellow', 'pedes', 'car', 'truck', 't_sign']

# confirm gpu
gpus = tf.config.list_physical_devices('GPU')
print('TF version:', tf.__version__)
print('GPUs available:', gpus)
if not gpus:
    print('WARNING: No GPU found - training will be very slow on CPU')
else:
    print('GPU confirmed - proceeding with training')

# load dataset
print('\nloading dataset ...')
train_data = object_detector.DataLoader.from_pascal_voc(
    train_img, train_ann, labels
)
val_data = object_detector.DataLoader.from_pascal_voc(
    val_img, val_ann, labels
)
print(f'  train: {len(train_data)} samples')
print(f'  val:   {len(val_data)} samples')

# model spec
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

# train
print('\nstarting training (75 epochs, batch 24, whole model) ...')
model = object_detector.create(
    train_data,
    model_spec=spec,
    batch_size=24,
    epochs=75,
    train_whole_model=True,
    validation_data=None,
)

# export
print('\nexporting .tflite ...')
model.export(export_dir=out_dir)

tflite_path = os.path.join(out_dir, 'model.tflite')
if os.path.exists(tflite_path):
    size_mb = os.path.getsize(tflite_path) / 1e6
    print(f'\n done - model.tflite is {size_mb:.1f} MB')
    print(f'   {tflite_path}')
else:
    print(f'\n export failed - no .tflite found in {out_dir}')
