# kv260 quantization: amd quark int8, per-tensor power-of-two scales, 500 calibration images
# calibrate_method is MinMSE; the log shows quark fell back to NonOverflow (see results/kv260/quark_500_quant.log)
# runs on: host pc
# output: ~/THESIS/kria/model_run3_quark_500.onnx

import numpy as np
from PIL import Image
import os
from quark.onnx import ModelQuantizer, QuantFormat, QuantType, PowerOfTwoMethod
from quark.onnx.quantization.config.config import Config, QuantizationConfig
from onnxruntime.quantization import CalibrationDataReader

CALIB_DIR = os.path.expanduser('~/THESIS/datasets/bdd100k_calib_500')
MODEL_IN = os.path.expanduser('~/THESIS/kria/models/model_run3_fp32.onnx')
MODEL_OUT = os.path.expanduser('~/THESIS/kria/model_run3_quark_500.onnx')
TMPDIR = '/media/eagle/New Volume/quark_tmp'
H, W = 448, 448

os.makedirs(TMPDIR, exist_ok=True)
imgs = sorted([os.path.join(CALIB_DIR, f) for f in os.listdir(CALIB_DIR) if f.endswith('.jpg')])
print(f'using {len(imgs)} calibration images')


class Reader(CalibrationDataReader):
    def __init__(self):
        self.idx = 0

    def get_next(self):
        if self.idx >= len(imgs):
            return None
        img = Image.open(imgs[self.idx]).convert('RGB').resize((W, H), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)[np.newaxis] / 255.0
        self.idx += 1
        return {'input_1': arr}


quant_config = QuantizationConfig(
    quant_format=QuantFormat.QDQ,
    calibrate_method=PowerOfTwoMethod.MinMSE,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=False,
    enable_npu_cnn=True,
    extra_options={'TmpDir': TMPDIR},
)
config = Config(global_quant_config=quant_config)
quantizer = ModelQuantizer(config)
quantizer.quantize_model(MODEL_IN, MODEL_OUT, Reader())
print('done:', os.path.getsize(MODEL_OUT), 'bytes')
