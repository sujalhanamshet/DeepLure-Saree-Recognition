"""
Dataset loading, design-ID parsing, and color-invariant augmentation pipelines.
Supports:
- Design identity extraction from filenames in C:\\deeplure\\archive
- Two-crop multi-view generation for Supervised Contrastive Learning
- Standard Evaluation Datasets (Gallery & Query)
- Synthetic Color-Invariance Stress Test Datasets (Grayscale, Strong Jitter, Hue Shift)
- Pairwise verification dataset generation
"""

import os
import re
import random
from typing import List, Tuple, Dict, Any
from PIL import Image
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from src.config import (
    IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD,
    TRAIN_DIR, VALID_DIR, TEST_DIR, HANDLOOM_DIR
)


def extract_base_design_id(filename: str, category: str = "") -> str:
    """
    Extracts the invariant design ID from raw image filenames in C:\\deeplure\\archive.
    Strips Roboflow hash suffixes: <base_name>_jpg.rf.<32_hex_hash>.jpg -> <base_name>
    """
    base = os.path.basename(filename)
    clean = re.sub(r'(_jpg|_jpeg|_png)?\.rf\.[a-f0-9]+\.(jpg|jpeg|png)$', '', base, flags=re.IGNORECASE)
    clean = os.path.splitext(clean)[0]
    if category:
        return f"{category}_{clean}"
    return clean


def get_train_transforms(image_size: int = IMAGE_SIZE):
    """
    Strong color-invariant augmentation pipeline for Supervised Contrastive Learning.
    Includes:
      - RandomResizedCrop, RandomHorizontalFlip, RandomRotation
      - Severe RandomGrayscale, strong ColorJitter, RandomAutocontrast, GaussianBlur
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.4)], p=0.8),
        T.RandomGrayscale(p=0.35),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
        T.RandomAutocontrast(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])


def get_eval_transforms(image_size: int = IMAGE_SIZE):
    """Standard deterministic evaluation transform."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])


def get_synthetic_color_stress_transforms(image_size: int = IMAGE_SIZE, mode: str = "color_jitter"):
    """
    Transforms specifically crafted for the SYNTHETIC COLOR-INVARIANCE STRESS TEST.
    Modes:
      - 'grayscale': Complete removal of chromatic channels.
      - 'color_jitter': Heavy hue rotation, contrast, brightness, and saturation shifts.
      - 'hue_shift': Extreme hue variation.
    """
    if mode == "grayscale":
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    elif mode == "color_jitter":
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.8, hue=0.5),
            T.ToTensor(),
            T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    elif mode == "hue_shift":
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ColorJitter(hue=0.5),
            T.ToTensor(),
            T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    else:
        return get_eval_transforms(image_size)


class TwoCropTransform:
    """
    Generates two distinct, independently augmented views of the same saree image.
    Both views share identical underlying geometric motifs but undergo independent
    spatial cropping and chromatic perturbations.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class TwoCropSareeDataset(Dataset):
    """
    Multi-view dataset for Supervised Contrastive Training (SupCon).
    Returns: ([view1, view2], design_label, category_name, design_id, file_path)
    """
    def __init__(self, samples: List[Tuple[str, str, str]], transform, label_map: Dict[str, int]):
        self.samples = samples
        self.two_crop_transform = TwoCropTransform(transform)
        self.label_map = label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, category, design_id = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))

        views = self.two_crop_transform(image)
        label = self.label_map[design_id]
        return views, label, category, design_id, path


class SareeDataset(Dataset):
    """
    Standard single-view Saree dataset for Gallery, Query, and Evaluation.
    Returns: (image_tensor, design_label, category_name, design_id, file_path)
    """
    def __init__(self, samples: List[Tuple[str, str, str]], transform=None, label_map: Dict[str, int] = None):
        self.samples = samples
        self.transform = transform or get_eval_transforms()
        self.label_map = label_map or {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, category, design_id = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))

        tensor = self.transform(image)
        label = self.label_map.get(design_id, -1)
        return tensor, label, category, design_id, path


def scan_dataset_partition(partition_dir: str) -> List[Tuple[str, str, str]]:
    """
    Scans a categorized folder partition in C:\\deeplure\\archive.
    Returns list of tuples: (file_path, category, design_id)
    """
    samples = []
    if not os.path.exists(partition_dir):
        return samples

    for cat in sorted(os.listdir(partition_dir)):
        cat_dir = os.path.join(partition_dir, cat)
        if os.path.isdir(cat_dir):
            for fname in sorted(os.listdir(cat_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(cat_dir, fname)
                    design_id = extract_base_design_id(fname, category=cat)
                    samples.append((fpath, cat, design_id))
    return samples


def scan_auxiliary_handloom_dataset(handloom_dir: str) -> List[Tuple[str, str, str]]:
    """
    Scans auxiliary unlabelled handloom saree collection in C:\\deeplure.
    Treated strictly as unlabelled/auxiliary gallery data (excluded from supervised training).
    """
    samples = []
    if not os.path.exists(handloom_dir):
        return samples

    for fname in sorted(os.listdir(handloom_dir)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fpath = os.path.join(handloom_dir, fname)
            design_id = f"Handloom_{os.path.splitext(fname)[0]}"
            samples.append((fpath, "Handloom", design_id))
    return samples


def build_verification_pairs(samples: List[Tuple[str, str, str]], num_pairs: int = 300, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Constructs a balanced set of same-design (label=1) and different-design (label=0) pairs
    from multi-view samples for validation threshold tuning.
    """
    rng = random.Random(seed)
    design_to_samples = defaultdict(list)
    for s in samples:
        design_to_samples[s[2]].append(s)

    multi_sample_designs = [d for d, slist in design_to_samples.items() if len(slist) >= 2]
    all_designs = list(design_to_samples.keys())
    pairs = []

    # 1. Positive Pairs
    pos_needed = num_pairs // 2
    if multi_sample_designs:
        while len(pairs) < pos_needed:
            d = rng.choice(multi_sample_designs)
            s1, s2 = rng.sample(design_to_samples[d], 2)
            pairs.append({
                'img1_path': s1[0], 'img2_path': s2[0],
                'label': 1, 'design1': s1[2], 'design2': s2[2],
                'cat1': s1[1], 'cat2': s2[1]
            })

    # 2. Negative Pairs
    while len(pairs) < num_pairs:
        d1, d2 = rng.sample(all_designs, 2)
        s1 = rng.choice(design_to_samples[d1])
        s2 = rng.choice(design_to_samples[d2])
        pairs.append({
            'img1_path': s1[0], 'img2_path': s2[0],
            'label': 0, 'design1': s1[2], 'design2': s2[2],
            'cat1': s1[1], 'cat2': s2[1]
        })

    rng.shuffle(pairs)
    return pairs
