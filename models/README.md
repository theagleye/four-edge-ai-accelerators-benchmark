# Models

The model files behind the published results. Use is limited to educational, research and not-for-profit purposes: see [LICENSE-MODELS](../LICENSE-MODELS).

| File | Used by | Made by | SHA-256 |
|---|---|---|---|
| `coral/model_run3_edgetpu.tflite` | Coral scripts | Edge TPU compiler on `model_ckpt59.tflite` (`results/training/model_ckpt59_edgetpu.log`) | `a20634ead159406ff81d4ce57e13a6d6e58e6cd82bbdc374dab23ff24a4b6351` |
| `vim3/model_run3.tflite` | VIM3 scripts | `scripts/training/export_ckpt59.py` | `b20302cb17e23d6ce37b73b39aa4b42cd73b99e3620c2cfda2b9bfcf8a871566` |
| `kv260/model_run3_quark_500.onnx` | KV260 timing and DPU mAP (deployed) | `scripts/kv260/quark_quantize_500.py` | `479405b15f9145319af5f0f896fd66ffb64978c7396bef6c47a8c37c465a5e81` |
| `kv260/model_run3_quark_200.onnx` | KV260 DPU mAP, 200 calibration images | no generating script kept | `da5bf849b88ce06902e2ea03cc223b221bc39254b71a8335ea8dae5d44990698` |
| `kv260/model_run3_quark_50.onnx` | KV260 DPU mAP, 50 calibration images | no generating script kept | `efa92ea706f50534d64427dd2765874030a125ee3680886feb3a33d328d487a8` |
| `kv260/model_run3.onnx` | KV260 CPU reference | TFLite INT8 model converted to ONNX | `0f2eb5e149e1ea7aa20a7feb3ec772c304bdda5b62a4331d365cfe9d18b3a3b7` |
| `kv260/model_run3_fp32.onnx` | input to `quark_quantize_500.py` | float SavedModel converted with tf2onnx | `aa03e9234e4e8e681d664c861111cb674f5d83fe5dabb4803e8ed90cd5a03cdc` |
| `kv260/anchors_run3.npy` | KV260 box decoding | anchors taken from the TFLite model | `e6026332876902d25432015fc4d1e8da003a3f02c047970f028207ce00b80962` |

The scripts load these files from the board paths in their constants, for example `/home/ubuntu/model_run3_quark_200.onnx`. Copy a file to that path or edit the constant.
