# Notes on the result files

The files in this folder are raw. Each one is a byte-for-byte copy of the file kept from its run, and one was renamed back to its original name (see Coral below). This page shows where every number in the main README comes from, which script wrote each file, and which labels inside the files are wrong or out of date.

## Where the README numbers come from

FPS is 1000 / E2E median. Energy per inference is mean inference power × E2E median.

| Platform | README value | File | Field | Raw value |
|---|---|---|---|---|
| Coral | invoke 326.87 ms | `coral/coral_realimage_e2e_results_run3.json` | `steady_state.invoke.median_ms` | 326.874 |
| Coral | E2E 375.61 ms | `coral/coral_realimage_e2e_results_run3.json` | `steady_state.e2e.median_ms` | 375.612 |
| Coral | invoke CV 3.53% | `coral/thermal_results_run3_coral.json` | `steady_state.invoke.cv` | 0.0353 |
| Coral | idle 0.80 W | `coral/coral_realimg_power_results_run3.json` | `idle.power_mean_W` | 0.7958 |
| Coral | inference 1.50 W | `coral/coral_realimg_power_results_run3.json` | `inference.power_mean_W` | 1.5037 |
| Coral | mAP 0.2288 | `coral/coral_map_results_run3.json` | `mAP_50` | 0.2288 |
| VIM3 | invoke 1374.12 ms | `vim3/realimg_e2e_results_run3_vim3.json` | `steady_state.invoke.median_ms` | 1374.117 |
| VIM3 | E2E 1407.35 ms | `vim3/realimg_e2e_results_run3_vim3.json` | `steady_state.e2e.median_ms` | 1407.354 |
| VIM3 | invoke CV 0.01% | `vim3/thermal_results_run3_vim3.json` | `steady_state.invoke.cv` | 0.0001 |
| VIM3 | idle 2.40 W | `vim3/realimg_power_results_run3_vim3.json` | `idle_power.mean_W` | 2.3951 |
| VIM3 | inference 2.73 W | `vim3/realimg_power_results_run3_vim3.json` | `inference_power.mean_W` | 2.7273 |
| VIM3 | mAP 0.2278 | `vim3/map_results_run3_vim3.json` | `mAP_50` | 0.2278 |
| Jetson | invoke 209.99 ms | `jetson/realimg_e2e_results_run3_jetson.json` | `invoke.median_ms` | 209.988 |
| Jetson | E2E 230.59 ms | `jetson/realimg_e2e_results_run3_jetson.json` | `e2e.median_ms` | 230.589 |
| Jetson | invoke CV 0.50% | `jetson/realimg_e2e_results_run3_jetson.json` | `invoke.cv` | 0.005 |
| Jetson | idle 1.95 W | `jetson/idle_power_results_run3_jetson.json` | `pom_5v_in_mean_mw` | 1946.6 |
| Jetson | inference 6.66 W | `jetson/realimg_e2e_results_run3_jetson.json` | `power.pom_5v_in_mean_mw` | 6658.3 |
| Jetson | mAP 0.700 (COCO) | `jetson/map_results_run3_jetson.json` | `AP_50` | 0.6997 |
| KV260 | invoke 192.71 ms | `kv260/benchmark_kr260_x07_500calib_final.json` | `real_image.invoke.median_ms` | 192.707 |
| KV260 | E2E 335.40 ms | `kv260/benchmark_kr260_x07_500calib_final.json` | `real_image.e2e.median_ms` | 335.402 |
| KV260 | invoke CV 6.39% | `kv260/benchmark_kr260_x07_500calib_final.json` | `real_image.invoke.cv_pct` | 6.3926 |
| KV260 | idle 4.31 W | `kv260/power_kr260_x07_500calib.json` | `idle_w.mean` | 4.3067 |
| KV260 | inference 5.50 W | `kv260/power_kr260_x07_500calib.json` | `inference_w.mean` | 5.496 |
| KV260 | DPU mAP 0.0233 / 0.0216 / 0.0000 | `kv260/map_results_quark_{500,200,50}calib_kr260.json` | `mAP_50` | 0.0233 / 0.0216 / 0.0 |
| KV260 | CPU mAP 0.2400 | `kv260/map_results_model3_orig.json` | `mAP_50` | 0.2400 |

Some files also hold their own `fps`, `throughput_fps`, `fps_per_W` or energy fields. Those use other definitions, such as total run time or invoke-only latency. The README figures use only the two formulas above.

## Which script wrote each file

| Result file | Script | How it was made |
|---|---|---|
| `training/epoch_losses.txt` | `training/train_run3.py` | as published |
| `training/export_ckpt59.log`, `export_fp32.log` | `training/export_ckpt59.py`, `export_fp32_ckpt59.py` | as published |
| `training/model_ckpt59_edgetpu.log` | Edge TPU compiler | compiler output for the deployed Edge TPU model |
| `coral/coral_timing_results_run3.json` | `coral/benchmark_coral_timing_run3.py` | as published |
| `coral/coral_realimage_e2e_results_run3.json` | `coral/benchmark_coral_realimage_e2e.py` | as published |
| `coral/coral_map_results_run3.json` | `coral/benchmark_coral_mAP_run3.py` | board copy with its `platform` and `model` labels set for the Coral; code otherwise the same |
| `coral/thermal_results_run3_coral.json` | `coral/thermal_benchmark_coral_run3.py` | as published |
| `coral/coral_realimg_power_results_run3.json` | `coral/analyze_realimg_power.py` | as published |
| `coral/realimg_power_run3_coral.csv` | `shared/atorch_logger.py` | logger output, renamed for the run |
| `vim3/*.json` | the matching `vim3/` script | as published |
| `vim3/idle_power_run3_vim3.csv`, `realimg_e2e_power_run3_vim3.csv` | `shared/atorch_logger.py` | logger output, renamed for the run |
| `jetson/*.json` | the matching `jetson/` script | as published |
| `kv260/quark_500_quant.log` | `kv260/quark_quantize_500.py` | console log of the run that made the deployed model |
| `kv260/benchmark_kr260_x07_500calib_final.json` | `kv260/benchmark_kr260_timing.py` | board copy with `MODEL_PATH`, `OUTPUT_JSON` and the `model`, `quantization_note` and `dpu_fingerprint` labels set for the 500-calibration model |
| `kv260/map_results_quark_200calib_kr260.json` | `kv260/map_benchmark_200calib_kr260.py` | as published |
| `kv260/map_results_quark_50calib_kr260.json`, `map_results_quark_500calib_kr260.json` | `kv260/map_benchmark_200calib_kr260.py` | board copies with `MODEL_PATH` and `OUTPUT_JSON` set for the 50- and 500-calibration models |
| `kv260/map_results_model3_orig.json` | `kv260/map_benchmark_model3_kr260.py` | as published |
| `kv260/map_full_model3_clean.log` | `kv260/map_benchmark_model3_kr260.py` | console log of the 4,993-image CPU reference run |
| `kv260/power_kr260_x07_500calib.json` | board-side script, not kept | method in the file: INA260 onboard sensor sampled at 10 s intervals during a 500-run inference loop |

"As published" means the script in `scripts/` writes this file as it stands, apart from the cleaned comments and console messages. The board copies were edited on the boards and not kept. Their output has the same layout as the published script's, field for field.

## Labels inside the files that are wrong or out of date

These are text labels typed into the scripts. None of them changes a measured value.

All files:

- `*_reference` blocks (`coral_reference`, `vim3_reference`, `timing_reference`) are constants typed into the scripts for side-by-side printing. They are not measurements from the file's own run, and some are out of date. Use the platform's own result files instead.
- A `coral_reference` mAP of 0.2347 comes from a separate, later Coral evaluation with its own pipeline: default image resize filter, fixed output tensor indices, its own matching code. That run is not part of the thesis comparison and is not included here. The Coral result used throughout is 0.2288, from `coral/coral_map_results_run3.json`, whose pipeline matches the VIM3 one.
- "BDD100K val" in `dataset` and `input_type` fields means the 4,993-image validation set of the run3 split. Those images come from the official BDD100K train split.
- `methodology` strings such as "matches eval_map_voc.py exactly" are labels. The IoU and AP code is the same computation in all four BDD100K mAP scripts: continuous-coordinate IoU, a match at IoU of at least 0.5, all-point interpolated AP.

Coral:

- `coral_map_results_run3.json`: `dataset` says `run2/val`. The 4,993 images are the run3 validation set. Its `coral_reference` block, with mAP 0.1658 and a note asking for a rerun, predates this run.
- The Dell copy of this file was named `coral_map_results_run3_old.json`. It is published here under the name the script writes.

Jetson:

- `map_results_run3_jetson.json`: `notes.model_note` says JetPack 4.6.1. The board ran TensorRT 8.0.1.6, which ships with JetPack 4.6.

KV260:

- `platform` says "KR260 Robotics Starter Kit". The board is a KV260 Vision AI Starter Kit, built on the same K26 module.
- `model`, `quantization_note` and `notes.calib_method` say MinMSE. For the 500-calibration model, `quark_500_quant.log` shows MinMSE hit NaN at tensor 156 of 464 and Quark switched to NonOverflow on its own. This repository has no quantization log for the 50- and 200-calibration models.
- `map_results_quark_50calib_kr260.json` and `map_results_quark_500calib_kr260.json`: the `model` label ends in "200 calib imgs". The model file name in the same label is the right one.
- `dpu_config.fingerprint` in the three DPU mAP files is a constant typed into the script, not a value read from the board. The timing file for the 500-calibration model records the x07 fingerprint, `0x101000056010407`.
- `benchmark_kr260_x07_500calib_final.json`: `system.power_watts` says ATORCH J7-C. KV260 power came from the onboard INA260, as `notes.power` and the power file say.
- `power_kr260_x07_500calib.json`: `energy_per_invoke_j`, `fps` and `fps_per_w` use the power above idle and the invoke-only latency. The README uses mean inference power × E2E median: 5.496 W × 0.335402 s = 1.84 J.

## Two details of the KV260 CPU reference

- `map_benchmark_model3_kr260.py` creates the session with the Vitis AI execution provider first. In the full run, the provider logged `Catch fatal exception, skip this subgraph` (`map_full_model3_clean.log`, line 12), so ONNX Runtime ran the whole graph on the CPU. The log is dated 10 May 2026. That is before the x07 firmware update, but the CPU result does not depend on the DPU firmware.
- This script resizes images with Pillow's default filter, because it passes no filter argument. That default is bicubic in current Pillow releases. The Coral and VIM3 mAP scripts pass bilinear explicitly.
