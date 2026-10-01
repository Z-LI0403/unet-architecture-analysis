"""
Dataset classes and registry for segmentation experiments.

Supported datasets:
  - VOC2012: Pascal VOC 2012 semantic segmentation (21 classes)

To add a new dataset:
  1. Create a Dataset subclass in this file
  2. Register it in DATASET_REGISTRY with default config
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Dataset configs: each entry defines defaults for a dataset
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "VOC2012": {
        "default_root": os.environ.get("VOC2012_ROOT", os.path.join("dataset", "VOC2012")),
        "num_classes": 21,
        "ignore_index": 255,
    },
    # Add new datasets here:
    # "Cityscapes": {
    #     "default_root": os.path.join("dataset", "Cityscapes"),
    #     "num_classes": 19,
    #     "ignore_index": 255,
    # },
}


def get_dataset_info(dataset_name):
    """Returns num_classes and ignore_index for a registered dataset."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}"
        )
    info = DATASET_REGISTRY[dataset_name]
    return info["num_classes"], info["ignore_index"]


# ---------------------------------------------------------------------------
# VOC2012
# ---------------------------------------------------------------------------

def _is_valid_voc_root(voc_root, split):
    split_file = os.path.join(voc_root, "ImageSets", "Segmentation", f"{split}.txt")
    return (
        os.path.isdir(os.path.join(voc_root, "JPEGImages"))
        and os.path.isdir(os.path.join(voc_root, "SegmentationClass"))
        and os.path.isfile(split_file)
    )


def _resolve_voc_root(voc_root, split):
    candidates = [
        voc_root,
        os.path.join(voc_root, "VOC2012"),
        os.path.join(voc_root, "VOC2012_train_val", "VOC2012_train_val"),
        os.path.join(voc_root, "VOC2012_test", "VOC2012_test"),
    ]
    for candidate in candidates:
        if _is_valid_voc_root(candidate, split):
            return candidate
    expected = os.path.join("ImageSets", "Segmentation", f"{split}.txt")
    raise FileNotFoundError(
        f"Could not resolve a VOC root from '{voc_root}' for split '{split}'. "
        f"Expected a VOC root with JPEGImages, SegmentationClass, and {expected}."
    )


class VOCSegmentationDataset(Dataset):
    """Pascal VOC 2012 semantic segmentation dataset."""

    def __init__(self, voc_root, split="train", img_size=(256, 256)):
        voc_root = _resolve_voc_root(voc_root, split)
        self.images_dir = os.path.join(voc_root, "JPEGImages")
        self.masks_dir = os.path.join(voc_root, "SegmentationClass")
        split_file = os.path.join(voc_root, "ImageSets", "Segmentation", f"{split}.txt")

        with open(split_file, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]

        self.samples = []
        for img_id in ids:
            img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
            mask_path = os.path.join(self.masks_dir, f"{img_id}.png")
            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No VOC segmentation pairs found in: {voc_root}")

        self.image_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.mask_resize = transforms.Resize(
            img_size, interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        image = self.image_transform(image)
        mask = self.mask_resize(mask)
        mask = torch.as_tensor(np.array(mask, dtype=np.int64), dtype=torch.long)
        return image, mask


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_DATASET_CLASSES = {
    "VOC2012": VOCSegmentationDataset,
    # Add new dataset classes here
}


def build_dataset(dataset_name, root=None, split="train", img_size=(256, 256)):
    """Instantiate a dataset by name."""
    if dataset_name not in _DATASET_CLASSES:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(_DATASET_CLASSES.keys())}"
        )
    if root is None:
        root = DATASET_REGISTRY[dataset_name]["default_root"]

    dataset_cls = _DATASET_CLASSES[dataset_name]
    return dataset_cls(root, split=split, img_size=img_size)


def get_dataloaders(dataset_name, root=None, img_size=(256, 256), batch_size=4, num_workers=4):
    """Creates train and validation dataloaders for a named dataset."""
    train_dataset = build_dataset(dataset_name, root=root, split="train", img_size=img_size)
    val_dataset = build_dataset(dataset_name, root=root, split="val", img_size=img_size)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
