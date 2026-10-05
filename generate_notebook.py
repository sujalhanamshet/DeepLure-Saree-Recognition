import json
import os

cells = []

def add_markdown(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    })

# Title & Executive Summary
add_markdown("""# Color-Invariant Saree Design Recognition Pipeline
### Technical Assignment: AIE-CASE - DeepLure Saree Recognition

**Objective:** Build and evaluate a deep metric-learning system that identifies and verifies sarees based on their **surface design, motifs, and weave textures**, completely invariant to the chromatic color palette.

**Dataset Location:** `C:\\deeplure` (Read-only direct access from local storage)

**Core Architectural & Methodological Highlights:**
1. **Model Architecture:** `ColorInvariantSareeEncoder` (Pretrained ResNet-18 backbone + 512-d MLP projection head + L2 Normalization to 256-D unit hypersphere).
2. **Metric Learning Objective:** `SupConLoss` (Supervised Contrastive Loss, $\\tau = 0.07$) with Two-Crop multi-view contrastive learning under extreme chromatic perturbations (ColorJitter, Grayscale, Hue rotations).
3. **Gallery Retrieval (`retrieve_top_k`):** Real-time ranking of reference gallery designs using exact cosine similarity.
4. **Pairwise Verification (`verify_pair`):** High-precision binary verification with an operating threshold $\\tau^* = 0.4279$ calibrated strictly on validation data.
5. **Color-Invariance Stress Testing:** Verified across unseen test saree designs under Grayscale, Color Jitter, and Hue-Shift perturbations.
6. **Disjoint Identities:** Train (431 designs / 1,293 images), Validation (115 designs), and Test (60 designs) partitions have zero identity leakage.""")

# Section 1: Environment Setup
add_markdown("## 1. Environment Setup & Dependency Imports")
add_code("""import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

# Import modular components from src
from src.config import (
    DATASET_ROOT, ARCHIVE_DIR, TRAIN_DIR, VALID_DIR, TEST_DIR, HANDLOOM_DIR,
    IMAGE_SIZE, EMBEDDING_DIM, PROJECTION_HIDDEN_DIM, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, TEMPERATURE, SEED,
    CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR,
    seed_everything, get_device
)
from src.dataset import (
    scan_dataset_partition, scan_auxiliary_handloom_dataset,
    TwoCropSareeDataset, SareeDataset,
    get_train_transforms, get_eval_transforms, get_synthetic_color_stress_transforms
)
from src.model import ColorInvariantSareeEncoder
from src.loss import SupConLoss
from src.metrics import compute_accuracy, compute_precision_recall_f1, compute_roc_curve, compute_auc
from src.retrieval import build_gallery_index, retrieve_top_k, evaluate_retrieval_metrics
from src.verification import verify_pair, calibrate_verification_threshold, evaluate_verification_on_test
from src.evaluate import benchmark_efficiency

seed_everything(SEED)
device = get_device()
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"Active Compute Device: {device}")""")

# Section 2: Configuration
add_markdown("## 2. Configuration & Hyperparameters")
add_code("""print("=== PIPELINE CONFIGURATION ===")
print(f"Dataset Root:           {DATASET_ROOT}")
print(f"Input Resolution:       {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Backbone:               ResNet-18 (Pretrained)")
print(f"Embedding Dimension:    {EMBEDDING_DIM}")
print(f"Projection Hidden Dim:  {PROJECTION_HIDDEN_DIM}")
print(f"Batch Size:             {BATCH_SIZE}")
print(f"Training Epochs:        {NUM_EPOCHS}")
print(f"Initial Learning Rate:  {LEARNING_RATE}")
print(f"Weight Decay:           {WEIGHT_DECAY}")
print(f"SupCon Temperature:     {TEMPERATURE}")""")

# Section 3: Dataset Inspection
add_markdown("## 3. Dataset Inspection & Identity Integrity Verification")
add_code("""train_samples = scan_dataset_partition(TRAIN_DIR)
valid_samples = scan_dataset_partition(VALID_DIR)
test_samples = scan_dataset_partition(TEST_DIR)
handloom_samples = scan_auxiliary_handloom_dataset(HANDLOOM_DIR)

train_dids = set(s[2] for s in train_samples)
valid_dids = set(s[2] for s in valid_samples)
test_dids = set(s[2] for s in test_samples)

print(f"Train Set:      {len(train_samples):,} images across {len(train_dids):,} unique designs (3 views/design)")
print(f"Validation Set: {len(valid_samples):,} images across {len(valid_dids):,} unique designs")
print(f"Test Set:       {len(test_samples):,} images across {len(test_dids):,} unique designs")
print(f"Aux Handloom:   {len(handloom_samples):,} unlabelled auxiliary images")

# Cross-split disjointness check
leak_tv = len(train_dids.intersection(valid_dids))
leak_tt = len(train_dids.intersection(test_dids))
leak_vt = len(valid_dids.intersection(test_dids))
print(f"\\nDisjoint Check: Train ∩ Valid = {leak_tv}, Train ∩ Test = {leak_tt}, Valid ∩ Test = {leak_vt}")
assert leak_tv == 0 and leak_tt == 0 and leak_vt == 0, "Data leakage detected across splits!""")

# Section 4: DataLoaders & Augmentations
add_markdown("## 4. Color-Invariant Multi-View Augmentation & DataLoaders")
add_code("""train_label_map = {did: i for i, did in enumerate(sorted(list(train_dids)))}
valid_label_map = {did: i for i, did in enumerate(sorted(list(valid_dids)))}
test_label_map = {did: i for i, did in enumerate(sorted(list(test_dids)))}

train_loader = DataLoader(
    TwoCropSareeDataset(train_samples, get_train_transforms(), train_label_map),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

valid_loader = DataLoader(
    TwoCropSareeDataset(valid_samples, get_train_transforms(), valid_label_map),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Training Batches:   {len(train_loader)} (Batch Size: {BATCH_SIZE})")
print(f"Validation Batches: {len(valid_loader)}")""")

# Section 5: Model Architecture
add_markdown("## 5. Model Architecture & Parameter Footprint")
add_code("""model = ColorInvariantSareeEncoder(
    backbone_name="resnet18",
    embedding_dim=EMBEDDING_DIM,
    projection_hidden_dim=PROJECTION_HIDDEN_DIM,
    pretrained=True
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
disk_size_mb = (total_params * 4) / (1024 * 1024)

print(f"Model Architecture:   ColorInvariantSareeEncoder (ResNet-18 Backbone)")
print(f"Total Parameters:     {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Embedding Dimension:  {EMBEDDING_DIM}")
print(f"Model Size on Disk:   ~{disk_size_mb:.2f} MB")""")

# Section 6: Checkpoint Loading & Training Verification
add_markdown("## 6. Model Training & Saved Checkpoint Verification")
add_code("""best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
final_ckpt_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")

print(f"Loading trained weights from: {best_ckpt_path}")
checkpoint = torch.load(best_ckpt_path, map_location=device)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

# Load saved training history
with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "r") as f:
    eval_data = json.load(f)

train_sum = eval_data["training_summary"]
print(f"\\nTraining Summary:")
print(f"  Epochs Completed:     {train_sum['num_epochs']}")
print(f"  Total Training Time:  {train_sum['total_train_time_min']:.2f} minutes ({train_sum['total_train_time_sec']:.1f} s)")
print(f"  Initial Train Loss:   {train_sum['history'][0]['train_loss']:.4f}")
print(f"  Best Validation Loss: {train_sum['best_val_loss']:.4f} (Epoch 7)")
print(f"  Final Train Loss:     {train_sum['final_train_loss']:.4f} (Epoch 8)")""")

# Section 7: Threshold Calibration
add_markdown("## 7. Pairwise Verification Threshold Calibration (Validation Data Only)")
add_code("""calib_metrics = eval_data["verification_validation_calibration"]
tau_star = calib_metrics["optimal_threshold"]

print("=== VALIDATION THRESHOLD CALIBRATION ===")
print(f"Optimal Decision Threshold (tau*): {tau_star:.4f}")
print(f"Validation F1-Score at tau*:        {calib_metrics['best_val_f1'] * 100:.2f}%")
print(f"Validation Accuracy at tau*:        {calib_metrics['best_val_acc'] * 100:.2f}%")
print(f"Validation ROC-AUC:                 {calib_metrics['val_roc_auc']:.4f}")""")

# Section 8: Test Set Verification Evaluation
add_markdown("## 8. Test Set Verification Performance (Unseen Designs)")
add_code("""test_verif = eval_data["verification_test_evaluation"]

print("=== UNSEEN TEST VERIFICATION EVALUATION ===")
print(f"Evaluation Pairs:     {test_verif['num_pairs']}")
print(f"Operating Threshold:  {test_verif['threshold_applied']:.4f}")
print(f"Accuracy:             {test_verif['accuracy'] * 100:.2f}%")
print(f"Precision:            {test_verif['precision'] * 100:.2f}%")
print(f"Recall:               {test_verif['recall'] * 100:.2f}%")
print(f"F1-Score:             {test_verif['f1_score'] * 100:.2f}%")
print(f"ROC-AUC:              {test_verif['roc_auc']:.4f}")
print(f"Confusion Matrix:     TP={test_verif['confusion_matrix']['TP']}, FP={test_verif['confusion_matrix']['FP']}, TN={test_verif['confusion_matrix']['TN']}, FN={test_verif['confusion_matrix']['FN']}")""")

# Section 9: Top-K Retrieval & Synthetic Stress Testing
add_markdown("## 9. Gallery Retrieval & Synthetic Color-Invariance Stress Testing")
add_code("""stress_metrics = eval_data["synthetic_color_invariance_stress_test"]

print("=== SYNTHETIC COLOR-INVARIANCE STRESS TEST RESULTS ===")
print(f"A. Grayscale Query Retrieval (Full Chromatic Removal):")
print(f"   Top-1 Accuracy: {stress_metrics['grayscale']['top1_accuracy'] * 100:.2f}%")
print(f"   Top-5 Accuracy: {stress_metrics['grayscale']['top5_accuracy'] * 100:.2f}%")
print(f"   MRR:            {stress_metrics['grayscale']['mrr']:.4f}")

print(f"\\nB. Color-Jittered Query Retrieval (Extreme Brightness/Contrast/Saturation Shifts):")
print(f"   Top-1 Accuracy: {stress_metrics['color_jitter']['top1_accuracy'] * 100:.2f}%")
print(f"   Top-5 Accuracy: {stress_metrics['color_jitter']['top5_accuracy'] * 100:.2f}%")
print(f"   MRR:            {stress_metrics['color_jitter']['mrr']:.4f}")

print(f"\\nC. Hue-Shifted Query Retrieval (Extreme Chromatic Phase Rotations):")
print(f"   Top-1 Accuracy: {stress_metrics['hue_shift']['top1_accuracy'] * 100:.2f}%")
print(f"   Top-5 Accuracy: {stress_metrics['hue_shift']['top5_accuracy'] * 100:.2f}%")
print(f"   MRR:            {stress_metrics['hue_shift']['mrr']:.4f}")""")

# Section 10: Live Inference Demo
add_markdown("## 10. Live Inference Demonstration: Retrieval & Verification")
add_code("""# Build Gallery Index using 60 test reference designs
eval_tf = get_eval_transforms()
test_gallery_loader = DataLoader(SareeDataset(test_samples, eval_tf, test_label_map), batch_size=32, shuffle=False)
gallery_index = build_gallery_index(model, test_gallery_loader, device=device)

# 1. Live Retrieval Demo on a Grayscale Stress Query
sample_query = test_samples[0]
gray_query_tf = get_synthetic_color_stress_transforms(mode="grayscale")
gray_query_tensor = gray_query_tf(Image.open(sample_query[0]).convert("RGB"))

results = retrieve_top_k(gray_query_tensor, gallery_index, model, k=5, device=device)
print(f"Query Saree: {sample_query[1]} (Design: {sample_query[2]}) [GRAYSCALE PERTURBED]")
for r in results:
    match_tag = "[MATCH]" if r["is_match"] else "[NON-MATCH]"
    print(f"  Rank {r['rank']}: Sim={r['similarity']:.4f} | {match_tag} | {r['category']} | {r['design_id']}")

# 2. Live Verification Demo
pair_demo = verify_pair(sample_query[0], sample_query[0], threshold=tau_star, model=model, device=device)
print(f"\\nPairwise Verification Output (Identical Design, Original vs Original):")
print(pair_demo)""")

# Section 11: Visualizations
add_markdown("## 11. Diagnostic Visualizations & Empirical Curves")
add_code("""fig_paths = [
    os.path.join(FIGURES_DIR, "similarity_distribution.png"),
    os.path.join(FIGURES_DIR, "roc_curve.png"),
    os.path.join(FIGURES_DIR, "confusion_matrix.png"),
    os.path.join(FIGURES_DIR, "retrieval_examples.png"),
    os.path.join(FIGURES_DIR, "synthetic_color_invariance_demo.png")
]

for p in fig_paths:
    if os.path.exists(p):
        img = mpimg.imread(p)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(os.path.basename(p), fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()""")

# Section 12: Summary Table
add_markdown("## 12. Final Metrics Summary Table")
add_code("""summary_df = pd.read_csv(os.path.join(RESULTS_DIR, "summary_metrics.csv"))
display(summary_df)""")

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r"c:\Users\sujal\OneDrive\Desktop\DeepLure-Saree-Recognition\saree_recognition_pipeline.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Generated and updated saree_recognition_pipeline.ipynb successfully!")
