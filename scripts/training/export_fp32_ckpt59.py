# exports checkpoint ckpt-59 of run3 to a float32 tflite model
# runs on: host pc
# output: run3/output/model_run3_fp32.tflite

import os
from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import EfficientDetSpec

run_dir = '/home/eagle/THESIS/final_training_vastai/run3'
train_img = os.path.join(run_dir, 'train', 'images')
train_ann = os.path.join(run_dir, 'train', 'annotations')
out_dir = os.path.join(run_dir, 'output')
ckpt_path = os.path.join(out_dir, 'ckpts', 'ckpt-59')

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

print('exporting FP32 .tflite (quantization_config=None)...')
model.export(
    export_dir=out_dir,
    tflite_filename='model_run3_fp32.tflite',
    quantization_config=None,
)

tflite_path = os.path.join(out_dir, 'model_run3_fp32.tflite')
if os.path.exists(tflite_path):
    size_mb = os.path.getsize(tflite_path) / 1e6
    print(f'\ndone - model_run3_fp32.tflite is {size_mb:.1f} MB')
    if size_mb < 1:
        print('WARNING: file too small - export likely failed')
    else:
        print('size looks good - copy to ~/THESIS/models/')
else:
    print('export failed - no .tflite found in', out_dir)
