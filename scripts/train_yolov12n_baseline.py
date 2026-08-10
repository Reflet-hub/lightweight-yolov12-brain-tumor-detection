
"""Train the native YOLOv12n baseline under the matched protocol.

By default, Ultralytics downloads yolo12n.pt on first use. Pass --weights to use a local checkpoint instead."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Final

import yaml

CLASS_NAMES: Final[dict[int, str]] = {
    0: "meningioma",
    1: "glioma",
    2: "pituitary_tumor",
}
EXPECTED_SPLIT_COUNTS: Final[dict[str, int]] = {
    "train": 2522,
    "val": 542,
}
EXPECTED_CLASS_COUNTS: Final[dict[str, dict[int, int]]] = {
    "train": {0: 596, 1: 1181, 2: 745},
    "val": {0: 112, 1: 245, 2: 185},
}
IMAGE_EXTENSIONS: Final[set[str]] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root containing train/images, train/labels, val/images, and val/labels.",
    )
    parser.add_argument(
        "--weights",
        default="yolo12n.pt",
        help="Native YOLOv12n checkpoint path or Ultralytics asset name.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("/kaggle/working/yolo_runs"),
        help="Directory in which Ultralytics stores the run.",
    )
    parser.add_argument(
        "--name",
        default="yolov12n_baseline_seed42",
        help="Ultralytics run name.",
    )
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow an existing run directory to be reused.",
    )
    return parser.parse_args()


def _image_files(folder: Path) -> list[Path]:
    """Return supported image files directly contained in a folder."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _validate_label_file(
    label_path: Path,
    class_counts: Counter[int],
) -> None:
    """Validate one YOLO label file and update class counts."""
    nonempty_lines = [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(nonempty_lines) != 1:
        raise ValueError(
            f"{label_path} must contain exactly one annotation; "
            f"found {len(nonempty_lines)}."
        )

    parts = nonempty_lines[0].split()
    if len(parts) != 5:
        raise ValueError(
            f"{label_path} must contain five values: "
            "class x_center y_center width height."
        )

    try:
        class_id = int(float(parts[0]))
        coordinates = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"Non-numeric value in {label_path}.") from exc

    if class_id not in CLASS_NAMES:
        raise ValueError(f"Invalid class id {class_id} in {label_path}.")

    if not all(0.0 <= value <= 1.0 for value in coordinates):
        raise ValueError(
            f"Bounding-box coordinates outside [0, 1] in {label_path}."
        )

    class_counts[class_id] += 1


def validate_dataset(dataset_root: Path) -> None:
    """Validate the fixed patient-disjoint train/validation dataset."""
    dataset_root = dataset_root.resolve()

    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"

        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {images_dir}")
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {labels_dir}")

        images = _image_files(images_dir)
        labels = sorted(labels_dir.glob("*.txt"))

        if len(images) != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count} images, "
                f"found {len(images)}."
            )
        if len(labels) != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count} labels, "
                f"found {len(labels)}."
            )

        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            missing_labels = sorted(image_stems - label_stems)[:10]
            labels_without_images = sorted(label_stems - image_stems)[:10]
            raise ValueError(
                f"{split}: image-label mismatch. "
                f"Missing labels={missing_labels}; "
                f"labels without images={labels_without_images}."
            )

        class_counts: Counter[int] = Counter()
        for label_path in labels:
            _validate_label_file(label_path, class_counts)

        expected_classes = EXPECTED_CLASS_COUNTS[split]
        actual_classes = {
            class_id: class_counts[class_id]
            for class_id in CLASS_NAMES
        }
        if actual_classes != expected_classes:
            raise ValueError(
                f"{split}: unexpected class distribution. "
                f"Expected {expected_classes}, found {actual_classes}."
            )

        print(
            f"{split}: {len(images)} images, {len(labels)} labels, "
            f"class counts={actual_classes}"
        )


def create_data_yaml(dataset_root: Path, destination: Path) -> Path:
    """Create an absolute-path Ultralytics data YAML file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    destination.write_text(
        yaml.safe_dump(content, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination


def training_arguments(
    data_yaml: Path,
    project: Path,
    name: str,
    device: str,
    workers: int,
    exist_ok: bool,
) -> dict[str, object]:
    """Return the matched training protocol used in all experiments."""
    return {
        "data": str(data_yaml),
        "epochs": 400,
        "patience": 100,
        "imgsz": 640,
        "batch": 8,
        "device": device,
        "workers": workers,
        "amp": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "cos_lr": True,
        "seed": 42,
        "deterministic": True,
        "pretrained": True,
        "cls_remap": True,
        "cache": False,
        "rect": False,
        "close_mosaic": 10,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "iou": 0.7,
        "max_det": 300,
        "project": str(project),
        "name": name,
        "exist_ok": exist_ok,
        "save": True,
        "save_period": 25,
        "val": True,
        "plots": True,
        "verbose": True,
    }


def print_environment() -> None:
    """Print the software and accelerator environment."""
    import torch
    import ultralytics

    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


def main() -> None:
    """Validate inputs, construct the model, and start training."""
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.project = args.project.expanduser().resolve()

    validate_dataset(args.dataset_root)
    data_yaml = create_data_yaml(
        args.dataset_root,
        args.project / "data_fold1_generated.yaml",
    )

    print_environment()
    from ultralytics import YOLO

    model = YOLO(args.weights, task="detect")
    model.info(verbose=True)

    train_args = training_arguments(
        data_yaml=data_yaml,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        exist_ok=args.exist_ok,
    )
    train_args["freeze"] = None
    model.train(**train_args)

    save_dir = Path(model.trainer.save_dir)
    print(f"Training outputs: {save_dir}")
    print(f"Best checkpoint: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
