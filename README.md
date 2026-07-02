## A Controlled Benchmark of Edge AI Accelerators Using EfficientDet-Lite2 for Traffic Object Detection

## Description
This project benchmarks a single INT8 object detection model across four edge AI accelerators: the Coral Edge TPU, the Khadas VIM3 NPU,  the NVIDIA Jetson Nano GPU, and the AMD Kria KV260 FPGA/DPU. It measures latency, throughput, power and energy, and detection accuracy. This repo holds the raw result logs behind the thesis figures, along with the OS and dependency information needed to reproduce them.

The model is EfficientDet-Lite2, trained on BDD100K (7 classes) and quantized to INT8. The same TFLite model runs on the Coral, VIM3, and KV260. The Jetson runs a YOLOv4-416 FP16 substitute instead, because the shared model would not compile to TensorRT (four conversion routes all failed). So the Jetson accuracy is on a different model and dataset and is not comparable on accuracy, only on speed and energy.

## Dataset
BDD100K, filtered down to 7 traffic classes (truck, car, pedestrian, traffic sign, and traffic light green/red/yellow). The dataset is not redistributed here, it is downloaded from Berkeley: http://bdd-data.berkeley.edu
The split uses a fixed seed: 33,288 training images and 4,993 validation images. All accuracy numbers are computed on the validation set.

## Platforms and software stack
Each board runs a different OS and software stack, and these affect the results.

Coral Dev Board Mini: Mendel Linux. Edge TPU runtime + PyCoral. Edge TPU Compiler 16.0.384591198. INT8 (TFLite post-training quantization).

Khadas VIM3: Ubuntu 20.04, kernel 4.9. TFLite with the TIM-VX delegate built from source. tflite_runtime 2.14.0, aml-npu 6.4.8.7. INT8 (TFLite post-training quantization).

NVIDIA Jetson Nano: Jetpack 4.6 (Ubuntu 18.04, L4T 32.6.1). TensorRT 8.0.1, CUDA 10.2, cuDNN 8.2. FP16.

AMD Kria KV260: Ubuntu 22.04 (Python 3.10.20), x07 firmware. ONNX Runtime 1.16.0 with the Vitis AI execution provider / VART. Vitis AI 2023.1-r3.5.0, Quark 0.11.1. INT8 (Quark XINT8)

## Notes
- The VIM3 needs kernel 4.9. The NPU driver (galcore.ko) does not initialize on a mainline kernel.
- While on SSH, the VIM3 NPU only comes up under an active terminal. Run inference inside screen, not nohup, otherwise the NPU will not initialize.
- The Coral tflite_runtime is whatever ships with the Mendel image. The version that matters for the reproducing the Coral numbers is the Edge TPU Compiler version above.
- For the KV260, all final numbers are on the x07 firmware, which fixed an on-board compiler crash. The DPU fingerprint is 0x101000056010407, target DPUCZDX8G_ISA1_B4096, running at 300 MHz. Final latency and power metrics (calibration-independent) are captured using 500-image calibration and the mAP results are based on the 500-calibration model.
- All host-side work (training, export, ONNX conversion, quantization, evaluation) ran on a PC running Ubuntu 22.04, with each board driven over the network.
- Power is captured differently per board: Coral and VIM3 use an external ATORCH J7-C BLE power meter on the USB-C power rail. Jetson uses the on-board INA3221 sensor via tegrastats (subprocess + parsed log). KV260 uses the on-board INA260 sensor via direct sysfs reads.

The Coefficient of Variations in Table 5.1 is computed from each board's dedicated repeated-inference timing run; the per-run raw logs are available on request.

## Known reproduction caveats
- The Coral and VIM3 numbers require an ATORCH J7-C power meter paired over BLE; the logger script is available on request.
- The KV260 deployed model (model_run3_quark_500.onnx) is the deployed integer graph. The quantization generator scripts are not included in this repo; the exact configuration (500-image calibration, MinMSE, per-tensor power-of-two, QInt8 wt / QUInt8 act, symmetric INT8, no CLE, DPUCZDX8G target) is documented in the thesis.
- The Jetson conversion attempts (documented in the thesis, Section 4.4.1) covered four failed TFLite→TRT/ONNX routes on Maxwell SM5.3.
- The Quark ablation experiments that documented the KV260 mAP collapse are not included in this repo; available on request.

## Layout
The results folder has one subfolder per board:
    - results/coral     Edge TPU
    - results/vim3      NPU
    - results/jetson    GPU
    - results/kv260     FPGA/DPU

Each folder has the raw JSON logs for that board (latency, throughput, power and energy, per-class accuracy where it applies). Similarly, the benchmark and evaluation scripts are in scripts/<board>/ alongside these results.

## Results
End-to-end throughput, energy per inference, and mAP at IoU 0.50 on the BDD100K validation set (the Jetson is the exception noted above)

Coral Edge TPU:  2.66 FPS, 0.56 J per inference, 0.2288 mAP. Most energy-efficient
Khadas VIM3: 0.71 FPS, 3.84 J per inference, 0.2278 mAP. Most deterministic but highest energy.
Kria KV260: 2.98 FPS, 1.84 J per inference, 0.2400 mAP (host-CPU figure, see below).
Jetson Nano: 4.34 FPS, 1.54 J per inference, 0.700 mAP (COCO-val2017, YOLOv4-416, not comparable on accuracy)

On the KV260 DPU the INT8 quantizer collapses the accuracy to about 0.02 mAP. The 0.240 above is the same model evaluated on the host CPU (the integer-equivalent graph), which is what the model is actually capable of. The collapse is a quantizer problem, not the model or the hardware. The full analysis is in the thesis. The Coral, VIM3, and KV260 host-CPU per-class results agree within about 0.03 AP across three separate inference stacks.


