# Lightweight YOLO-Based Brain Tumor Detection

Code, model configuration, checkpoints, and analysis material for the study
**"Patient-Disjoint Brain Tumor Detection with a Lightweight YOLOv12n-Based Model"**
(revised version).

The detector combines a learnable BatchNorm input adaptor, a truncated ImageNet-pretrained
MobileNetV3-Small backbone, a depthwise-separable SSDLiteX-inspired FPN–PAN neck, a single A2C2f
area-attention block at P5, and the anchor-free YOLOv12n detection head. It has 746,015 parameters
(742,223 after fusion) and 2.4 GFLOPs at 640 × 640, that is, 71% fewer parameters and 62% fewer
GFLOPs than native YOLOv12n.

> **Version 1.1.0 (revision).** The single-fold evaluation of v1.0.0 is replaced by five-fold
> patient-level cross-validation on the official folds of the dataset, with patient-level bootstrap
> confidence intervals, repeated seeds, component ablations, an exploratory false-positive test on
> tumor-free images, and a matched smartphone benchmark of the proposed model and YOLOv12n.

## Repository contents

```
models/
  adopted_model_best.pt                  checkpoint used for export and deployment (v1.0.0)
  yolo12n_mnv3s_ssdlitex_lite.yaml       model configuration (Ultralytics 8.4.104)
scripts/
  train_adopted_full_finetuning.py       matched training protocol, adopted model
  train_frozen_backbone.py               frozen-backbone variant
  train_yolov12n_baseline.py             native YOLOv12n baseline
cross_validation/                        new in v1.1.0
  cv5fold_patient_level_multi.ipynb      five-fold patient-level cross-validation (Kaggle, T4 x2)
  cv5fold_patient_level_ablation.ipynb   component ablations (A2C2f attention, neck, input adaptor)
  split_metadata.csv                     one row per slice: patient ID, class, official fold, box
analysis/                                new in v1.1.0
  cv_analysis.py                         fold metrics, patient-level bootstrap CIs, paired differences
  seed_analysis.py                       run-to-run variability (seeds 42, 0, 1)
  results/                               all reports and tables produced by the two scripts
false_positive_test/                     new in v1.1.0
  negatives_notumor_cpu.ipynb            exploratory test on 1,681 tumor-free images (CPU)
  negatives_analysis.py                  false-alarm rates, thresholds, image-level AUC
  results/                               per-image scores, detections, summaries
deployment/                              new in v1.1.0
  r1_8_tflite_export_latency.ipynb       FP32 TensorFlow Lite export, export fidelity, CPU latency/memory
  android_benchmark_app/                 Android Studio project of the smartphone benchmark
  benchmark_images.csv                   the 100 fold-1 slices used for all latency measurements
  results/                               CPU and smartphone results
figures/                                 new in v1.1.0
  fig1.py, figs456.py, fig7.py, cmlib.py scripts of Figs. 1 and 4–7 of the revised manuscript
  fig_data.zip                           data used by the figure scripts
LICENSE                                  GNU AGPL-3.0
```

## Dataset

The study uses the public brain tumor dataset of Cheng et al. (3,064 contrast-enhanced T1-weighted
slices from 233 patients with meningioma, glioma, or pituitary tumor), converted to YOLO format with
one bounding box per slice.

- YOLO-formatted dataset (GitHub release asset):
  https://github.com/Reflet-hub/lightweight-yolov12-brain-tumor-detection/releases/latest/download/figshare_yolo_trainval_official.zip
- Kaggle mirror: https://kaggle.com/datasets/c0dbf930a618ff6427f0d1c4670942bcc3e1a3a504d3bf6c8673a4b9547a6503
- Original dataset (Figshare, CC BY 4.0): https://doi.org/10.6084/m9.figshare.1512427

```
figshare-yolo-trainval-official/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

The `train`/`val` folders correspond to the v1.0.0 split (folds 2–5 versus fold 1). The
cross-validation does not use this split. It reads the official patient-level fold of every slice from
`cross_validation/split_metadata.csv`, which must be placed at the root of the dataset folder (the
notebook checks that it finds exactly one copy).

## Installation

```
python -m pip install -r requirements.txt
```

All experiments used Ultralytics 8.4.104. The cross-validation ran with Python 3.13.15, PyTorch 2.11.0
and CUDA 12.8 on Kaggle (two NVIDIA T4 GPUs).

## Matched training protocol

All configurations use the same protocol: AdamW, initial learning rate 0.001 with cosine decay to
0.01 × the initial value, weight decay 0.0005, β1 = 0.937, three warm-up epochs, batch size 8,
640 × 640 input, at most 400 epochs with early stopping (patience 100), automatic mixed precision,
deterministic mode, and the default Ultralytics augmentation (mosaic, HSV, translation ±10%, scaling
±50%, horizontal flip). The checkpoint with the highest validation mAP50-95 is retained. The proposed
model receives ImageNet weights for its backbone only. YOLOv12n starts from the official COCO-pretrained
`yolo12n.pt`, so differences between the two models reflect architecture and initialization together.

### Training scripts (v1.0.0 split: folds 2–5 for training, fold 1 for validation)

```
python scripts/train_adopted_full_finetuning.py \
  --dataset-root /path/to/figshare-yolo-trainval-official \
  --project /kaggle/working/yolo_runs

python scripts/train_frozen_backbone.py \
  --dataset-root /path/to/figshare-yolo-trainval-official \
  --project /kaggle/working/yolo_runs

python scripts/train_yolov12n_baseline.py \
  --dataset-root /path/to/figshare-yolo-trainval-official \
  --project /kaggle/working/yolo_runs
```

The model trained this way (`models/adopted_model_best.pt`) is the one exported and benchmarked on
devices. Because fold 1 guided its checkpoint selection, its fold-1 accuracy is not a performance
estimate.

## Five-fold patient-level cross-validation (v1.1.0)

In rotation *k* (*k* = 1, …, 5), fold *k* is the test set, fold *k* + 1 the validation set (fold 1
for *k* = 5), and the three other folds the training set. The validation fold is used only for early
stopping and checkpoint selection. Patient identifiers are audited in every rotation.

**How to run (Kaggle):** open `cross_validation/cv5fold_patient_level_multi.ipynb`, attach the dataset
(with `split_metadata.csv` at its root), select **GPU T4 x2** and **Internet on**, and add the Kaggle
secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` (*Add-ons → Secrets*). The secrets are used only to back up
the runs to a private Kaggle dataset, so that an interrupted session can resume. No key is written in
the notebook. Run the smoke test first (`SMOKE_TEST = True`), then set `SMOKE_TEST = False`, choose the
`JOBS` (configuration, test fold) and use *Save & Run All (Commit)*. The ablation notebook works in the
same way, with `CONFIG` set to `a2conv`, `yolo12neck`, or `noadaptor`.

**Analysis:** the release asset `cv_runs_light.zip` contains light copies of all run archives
(test predictions, summaries, training logs; no weights). Unzip it into a folder `runs/`, then:

```
python analysis/cv_analysis.py --cv runs --meta cross_validation/split_metadata.csv --out out
python analysis/cv_analysis.py --cv runs --meta cross_validation/split_metadata.csv --out out_frozen --configs adopted frozen
python analysis/cv_analysis.py --cv runs --meta cross_validation/split_metadata.csv --out out_a2conv --configs adopted a2conv
python analysis/seed_analysis.py --cv runs --meta cross_validation/split_metadata.csv --out out_seeds
```

The bootstrap uses 2,000 resamples with seed 2026 by default. The outputs of the study are in
`analysis/results/`.

### Results (mean ± SD across the five test folds, seed 42; 95% patient-level bootstrap CI)

| Configuration | Parameters (fused) | GFLOPs | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Native YOLOv12n | 2,557,313 | 6.3 | 0.841 ± 0.032 | 0.813 ± 0.024 | 0.860 ± 0.023 (0.835–0.881) | 0.583 ± 0.026 (0.559–0.605) |
| Proposed, frozen backbone | 742,223 | 2.4 | 0.820 ± 0.047 | 0.776 ± 0.036 | 0.817 ± 0.041 (0.787–0.842) | 0.520 ± 0.044 (0.496–0.544) |
| Proposed, full fine-tuning (adopted) | 742,223 | 2.4 | 0.833 ± 0.032 | 0.816 ± 0.030 | 0.851 ± 0.027 (0.823–0.875) | 0.546 ± 0.034 (0.521–0.570) |

- YOLOv12n versus the adopted model: mAP50 +0.009 (95% CI −0.006 to +0.024, not significant);
  mAP50-95 +0.037 (95% CI 0.025 to 0.049).
- Backbone freezing versus full fine-tuning: mAP50 −0.035 (95% CI −0.047 to −0.021).
- Glioma is the most difficult class for both models (recall 0.641 and 0.644 at the operating
  threshold).
- Seeds 0 and 1 (rotations 1 and 2) and the three ablations are reported in `analysis/results/`.

The fold-1 values reported in v1.0.0 (for example, mAP50 0.904 for the adopted model) came from the
fold that was also used for checkpoint selection. They are superseded by the cross-validated values above.

### Checkpoints of the cross-validation

The release asset `cv_checkpoints_seed42.zip` (38 MiB) contains the 15 selected checkpoints of the
cross-validation (adopted, frozen, and YOLOv12n; test folds 1–5; seed 42), with their `args.yaml`,
`summary.json`, and SHA-256 prefixes in `CHECKPOINTS.txt`.

## Exploratory false-positive test

`false_positive_test/negatives_notumor_cpu.ipynb` applies the ten cross-validation models (adopted and
YOLOv12n, seed 42) to the "no tumor" images of the public Kaggle brain MRI dataset of Nickparvar
(https://doi.org/10.34740/KAGGLE/DSV/2645886), whose tumor-free images originate from Br35H. After
removal of byte-identical duplicates, 1,681 images remain. At a confidence threshold of 0.25, the
adopted models raised a false alarm on 21.7% ± 9.8% of these images and YOLOv12n on 27.7% ± 2.0%.
These images come from another source and include other MRI sequences, so the test also involves a
domain shift. It is exploratory and does not measure specificity on normal examinations.

## Deployment

`deployment/r1_8_tflite_export_latency.ipynb` exports the adopted model and YOLOv12n to FP32 TensorFlow
Lite in the same session, checks export fidelity on fold 1, and measures CPU latency and memory.
`deployment/android_benchmark_app/` is the Android Studio project used on a Xiaomi Redmi 14C (four CPU
threads, no delegate). Copy the two `.tflite` files to `app/src/main/assets/models/` and the 100 images
of `deployment/benchmark_images.csv` to `app/src/main/assets/benchmark_images/` before building.

| Platform | Model | End-to-end latency (ms) | Inference (ms) | Memory increase (MiB) |
|---|---|---|---|---|
| Kaggle CPU | Adopted model | 60.0 ± 4.2 | 42.6 | 141 (peak RSS) |
| Kaggle CPU | YOLOv12n | 139.3 ± 6.4 | 122.0 | 184 (peak RSS) |
| Redmi 14C | Adopted model | 311.2 ± 6.1 | 274.5 | 44.7 (PSS) |
| Redmi 14C | YOLOv12n | 811.3 ± 6.6 | 775.4 | 80.1 (PSS) |

## Dataset attribution

The derived dataset is based on: Cheng J. Brain tumor dataset. Figshare;
https://doi.org/10.6084/m9.figshare.1512427 (CC BY 4.0). The original MATLAB records were converted to
640 × 640 PNG images, and the tumor masks to one YOLO bounding box per slice. Any reuse must keep this
attribution and describe these transformations.

## Software and model licensing

The code, model configuration, and fine-tuned weights are released under the GNU AGPL-3.0 (see
`LICENSE`), in line with the Ultralytics license, unless an Ultralytics Enterprise License applies. The
dataset remains under CC BY 4.0.

## Intended use and limitations

This project is for research only. It is not a medical device and must not be used for clinical
diagnosis. The model was trained and evaluated on two-dimensional slices that each contain one tumor.
It was not evaluated on normal examinations, on slices with several lesions, or on volumetric MRI, and
no external clinical validation, calibration, or radiologist comparison was performed.

## Citation

A complete citation will be added after publication of the associated article.
