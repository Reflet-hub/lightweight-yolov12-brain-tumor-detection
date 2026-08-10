# Lightweight YOLO-Based Brain Tumor Detection

Official reproducibility materials for the study:

> **Lightweight YOLO-Based Brain Tumor Detection: Patient-Disjoint Validation and Smartphone Deployment**

This repository contains the custom model configuration, the three training scripts used for the matched experiments, the adopted-model checkpoint, and access to the patient-disjoint YOLO-formatted dataset.

## Repository contents

```text
lightweight-yolov12-brain-tumor-detection/
├── README.md
├── requirements.txt
├── models/
│   ├── adopted_model_best.pt
│   └── yolo12n_mnv3s_ssdlitex_lite.yaml
└── scripts/
    ├── train_adopted_full_finetuning.py
    ├── train_frozen_backbone.py
    └── train_yolov12n_baseline.py
```

`adopted_model_best.pt` is the best checkpoint from the full-fine-tuning run. Add it to `models/` before publishing the repository. After upload, it will be available at [models/adopted_model_best.pt](models/adopted_model_best.pt).

## Dataset

The study uses a derived YOLO-formatted version of the public **Brain Tumor Dataset** by Jun Cheng. The source collection contains 3,064 contrast-enhanced T1-weighted MRI slices from 233 patients and three tumor classes:

- `0`: meningioma
- `1`: glioma
- `2`: pituitary tumor

The official patient organization was preserved:

| Partition | Official folds | Images | Patients |
|---|---:|---:|---:|
| Training | 2–5 | 2,522 | 188 |
| Validation | 1 | 542 | 45 |
| Total | 1–5 | 3,064 | 233 |

No patient identifier is shared between training and validation.

The dataset ZIP is stored as a GitHub Release asset rather than committed to Git history.

### Download

- [GitHub Release dataset archive](https://github.com/Reflet-hub/lightweight-yolov12-brain-tumor-detection/releases/latest/download/figshare_yolo_trainval_official.zip)
- [Kaggle mirror](https://kaggle.com/datasets/c0dbf930a618ff6427f0d1c4670942bcc3e1a3a504d3bf6c8673a4b9547a6503)
- [Original Figshare dataset](https://doi.org/10.6084/m9.figshare.1512427)

Expected extracted structure:

```text
figshare-yolo-trainval-official/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

The public scripts validate the number of images and labels, image-label correspondence, normalized YOLO coordinates, class identifiers, and the expected class distributions before training.

## Proposed architecture

The adopted detector combines:

- a learnable three-channel input BatchNorm adaptor;
- a truncated ImageNet-pretrained MobileNetV3-Small backbone;
- a depthwise-separable SSDLiteX-inspired top-down and bottom-up feature-fusion neck;
- one A2C2f block on the low-resolution P5 semantic path;
- an anchor-free three-scale YOLOv12 detection head.

The unfused training architecture contains **746,015 parameters**.

The model uses only modules supplied by PyTorch/TorchVision and Ultralytics 8.4.104. No separate custom Python layer implementation is required; the architecture is fully defined in `models/yolo12n_mnv3s_ssdlitex_lite.yaml`.

## Installation

The original Kaggle experiments used:

- Python 3.12.13
- PyTorch 2.10.0+cu128
- CUDA 12.8
- NVIDIA Tesla T4
- Ultralytics 8.4.104

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

For Kaggle training, enable a GPU accelerator. Internet access is also required the first time the scripts download the ImageNet MobileNetV3-Small weights or the native `yolo12n.pt` checkpoint.

## Matched training protocol

All three experiments use the same optimization settings:

| Setting | Value |
|---|---:|
| Maximum epochs | 400 |
| Early-stopping patience | 100 |
| Input size | 640 × 640 |
| Batch size | 8 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Final LR fraction | 0.01 |
| Weight decay | 0.0005 |
| Momentum | 0.937 |
| Scheduler | Cosine |
| Warm-up | 3 epochs |
| Seed | 42 |
| AMP | Enabled |
| Data-loading workers | 2 |

MixUp, CutMix, and Copy-Paste are disabled. The best checkpoint is selected by the Ultralytics training pipeline according to validation mAP@0.50:0.95.

## Training

Replace `/path/to/figshare-yolo-trainval-official` with the extracted dataset directory.

### 1. Adopted model: full fine-tuning

```bash
python scripts/train_adopted_full_finetuning.py \
  --dataset-root /path/to/figshare-yolo-trainval-official \
  --project /kaggle/working/yolo_runs
```

### 2. Frozen-backbone ablation

This run uses `freeze=2`, which freezes top-level modules 0 and 1: the input adaptor and the truncated MobileNetV3-Small backbone.

```bash
python scripts/train_frozen_backbone.py \
  --dataset-root /path/to/figshare-yolo-trainval-official \
  --project /kaggle/working/yolo_runs
```

### 3. Native YOLOv12n baseline

```bash
python scripts/train_yolov12n_baseline.py \
  --dataset-root /path/to/figshare-yolo-trainval-official \
  --project /kaggle/working/yolo_runs
```

To use an already downloaded baseline checkpoint:

```bash
python scripts/train_yolov12n_baseline.py \
  --dataset-root /path/to/figshare-yolo-trainval-official \
  --weights /path/to/yolo12n.pt
```

Each script writes a corrected absolute-path dataset YAML to the selected project directory and stores the run under a distinct name. Use `--help` to display all available arguments.

## Fold-1 validation results

| Configuration | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|---:|
| Native YOLOv12n baseline | 0.918 | 0.875 | 0.922 | 0.655 |
| Proposed model, frozen backbone | 0.859 | 0.823 | 0.883 | 0.601 |
| Proposed model, full fine-tuning (adopted) | 0.901 | 0.825 | 0.904 | 0.617 |

All values correspond to the same 542-image patient-disjoint fold-1 validation partition.

## Dataset attribution

The derived YOLO-formatted dataset is based on:

> Cheng, J. (2024). *Brain Tumor Dataset* (Version 8). Figshare. https://doi.org/10.6084/m9.figshare.1512427

The original dataset is distributed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. This repository must retain attribution to the original author and state the transformations applied: MATLAB-to-image conversion, bounding-box conversion to YOLO format, and organization according to the official patient-level folds.

## Software and model licensing

This project uses Ultralytics YOLO. The training scripts, model configuration, and fine-tuned model weights are intended for release under **GNU AGPL-3.0**, unless an applicable Ultralytics Enterprise License is held. The dataset remains under its original **CC BY 4.0** terms.

## Intended use and limitations

This repository is provided for research and reproducibility. It is not a medical device and must not be used as a substitute for radiological, histopathological, or clinical diagnosis. The reported evaluation uses one internal patient-disjoint validation fold and does not constitute external clinical validation.

## Citation

A complete citation will be added after publication of the associated article.
