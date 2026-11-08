# A Controlled Benchmark of Edge AI Accelerators for Traffic Object Detection

Scripts and raw result files for my MSc thesis at the University of Turku (2026).

One EfficientDet-Lite2 detector was trained once on a 7-class BDD100K subset and deployed on three edge accelerators: the Coral Dev Board Mini (Edge TPU), the Khadas VIM3 (NPU) and the AMD Kria KV260 (FPGA DPU). The Jetson Nano could not build the EfficientDet-Lite2 graph with its TensorRT version, so it runs YOLOv4-416 in TensorRT FP16 as a reference point. Every board was measured the same way: latency, power, energy per inference and detection accuracy.

Thesis: [A Controlled Benchmark of Edge AI Accelerators for Traffic Object Detection](https://urn.fi/URN:NBN:fi-fe20260803114484) (UTUPub).

## Results

| Platform | Model | Invoke median (ms) | E2E median (ms) | FPS | Inference power (W) | Energy per inference (J) | mAP@0.50 |
|---|---|---|---|---|---|---|---|
| Coral Dev Board Mini | EfficientDet-Lite2 INT8 | 326.87 | 375.61 | 2.66 | 1.50 | 0.56 | 0.2288 |
| Khadas VIM3 | EfficientDet-Lite2 INT8 | 1374.12 | 1407.35 | 0.71 | 2.73 | 3.84 | 0.2278 |
| AMD Kria KV260 | EfficientDet-Lite2 INT8 (Quark) | 192.71 | 335.40 | 2.98 | 5.50 | 1.84 | 0.0233 on the DPU |
| Jetson Nano | YOLOv4-416 FP16 | 209.99 | 230.59 | 4.34 | 6.66 | 1.54 | 0.700 (COCO, not comparable) |

- Latency: 500 real images per board after warm-up, from the BDD100K validation set (COCO val2017 on the Jetson). E2E covers image load, resize, inference and post-processing.
- FPS is 1000 divided by the E2E median.
- Energy per inference is the mean inference power times the E2E median.
- Power: an ATORCH J7-C USB meter for the Coral and VIM3, tegrastats for the Jetson, the onboard INA260 for the KV260.
- mAP@0.50 uses the 4,993-image validation set for the Coral, VIM3 and KV260, with all-point interpolated AP and IoU of at least 0.5. The Jetson figure is on COCO val2017.

[results/NOTES.md](results/NOTES.md) shows the file and field behind every number in this table.

## The KV260 accuracy collapse

On the KV260 DPU the detector's accuracy collapses:

| KV260 model | mAP@0.50 |
|---|---|
| DPU, 50 calibration images | 0.0000 |
| DPU, 200 calibration images | 0.0216 |
| DPU, 500 calibration images (deployed) | 0.0233 |
| TFLite INT8 model as ONNX, on the KV260 CPU | 0.2400 |

The network itself is fine. Converted to ONNX and run on the KV260's ARM CPU, the TFLite INT8 model scores 0.2400. Its per-class AP stays within 0.034 of the Coral and VIM3 on every class. The Coral and VIM3 agree within 0.005. More calibration images barely help. The thesis traces the collapse to the DPU's quantization: AMD Quark uses per-tensor power-of-two scales, and they do not fit the depthwise-separable classification head.

Quantization of the deployed model, from [results/kv260/quark_500_quant.log](results/kv260/quark_500_quant.log):

- AMD Quark 0.11.1, INT8 weights and activations, per-tensor power-of-two scales.
- Cross-layer equalization was on: 82 patterns.
- The script asks for MinMSE calibration. MinMSE hit NaN at tensor 156 of 464, and Quark switched to NonOverflow on its own. The deployed model uses NonOverflow scales.

## Platforms

| Platform | Accelerator | System | Runtime | Precision |
|---|---|---|---|---|
| Coral Dev Board Mini | Edge TPU | Mendel Linux (Eagle), Python 3.7.3 | TFLite with the Edge TPU delegate (pycoral 1.0.0) | INT8 |
| Khadas VIM3 Basic | Amlogic A311D NPU | Ubuntu 22.04 (Khadas Fenix 1.6.3), kernel 5.15.137 | TFLite with the TIM-VX delegate, built from source | INT8 |
| Jetson Nano 4GB | 128-core Maxwell GPU | JetPack 4.6 (L4T 32.6.1), CUDA 10.2, cuDNN 8.2 | TensorRT 8.0.1.6 | FP16 |
| AMD Kria KV260 | DPUCZDX8G B4096 | Ubuntu 22.04, x07 firmware, Vitis AI 3.5 | ONNX Runtime 1.16.0 with the Vitis AI execution provider | INT8 |


## Data

- Source: the BDD100K 100k detection labels (`det_v2_train_release.json`). Images and labels are not in this repository. Get them from the [BDD100K site](https://bdd-data.berkeley.edu/).
- Classes: red, green and yellow traffic lights, pedestrian, car, truck, traffic sign. Traffic lights without a color label are dropped.
- Split: [scripts/training/prep_voc_50k.py](scripts/training/prep_voc_50k.py) samples 50,000 labeled images from the official BDD100K train split (seed 42) and makes a class-stratified split. The training log reports 45,000 training and 4,993 validation images. The 4,993-image validation set is the evaluation set for every BDD100K accuracy figure.
- The evaluation set comes from the official train split, not the official val split. BDD100K contains a pair of byte-identical images under different names. One landed in training and the other in the evaluation set, so the evaluation set overlaps training by 1 image of 4,993.
- KV260 calibration: the 200- and 500-image sets come from the BDD100K `10k/val` folder. By SHA-256, none of their images appears in the training or evaluation split.

## Repository layout

```
scripts/
  training/   run3 split, training (tflite-model-maker), exports of checkpoint ckpt-59
  coral/      timing, real-image E2E, mAP, 500-run thermal, power analysis
  vim3/       timing, real-image E2E, mAP, 500-run thermal, power analysis
  jetson/     timing, real-image E2E with power, COCO mAP, 500-run thermal, idle power, ONNX cast attempts
  kv260/      Quark quantization, timing, DPU mAP, CPU reference mAP
  shared/     ATORCH J7-C power logger
results/
  training/   epoch losses, export logs, Edge TPU compiler log
  coral/ vim3/ jetson/ kv260/   raw result files, exactly as the scripts wrote them
  NOTES.md    where each number comes from, and the known wrong labels inside the raw files
```

## Running the scripts

- The scripts are the versions that produced the results. Only comments, layout and console messages were cleaned. A syntax-tree comparison against the originals confirms the code is otherwise unchanged.
- Paths are the absolute paths used on each board: `/mnt/thesis` on the Coral, `/home/khadas` on the VIM3, `~/THESIS/jetson_nano` on the Jetson, `/home/ubuntu` on the KV260. Edit the constants at the top of a script before running it.
- File names use `kr260`, the project's working name for the board. The board is a KV260 Vision AI Starter Kit.
- Some result files came from copies of a script with a few constants changed, such as the model path for each calibration size. [results/NOTES.md](results/NOTES.md) lists each case.
- Models and datasets are not included.

## Citation

Cite the thesis. [CITATION.cff](CITATION.cff) has the details.

```bibtex
@mastersthesis{shrestha2026edgeai,
  author = {Shrestha, Ashish},
  title  = {A Controlled Benchmark of Edge AI Accelerators for Traffic Object Detection},
  school = {University of Turku},
  year   = {2026},
  url    = {https://urn.fi/URN:NBN:fi-fe20260803114484}
}
```

## License

Code is under the MIT License ([LICENSE](LICENSE)). Result files and documentation are under CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)).
