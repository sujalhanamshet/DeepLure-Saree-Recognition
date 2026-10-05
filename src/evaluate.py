"""
Comprehensive Evaluation, Efficiency Benchmarking, and Visualization Pipeline.
Generates publication-quality figures and exports detailed metric reports for:
1. Standard Design Retrieval
2. Synthetic Color-Invariance Stress Testing (Grayscale, Strong ColorJitter)
3. Pair Verification & Similarity Distributions (Same-Design vs Different-Design)
4. Computational Efficiency (Parameters, Size, Latency, FPS)
"""

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.config import (
    FIGURES_DIR, METRICS_DIR, TOP_K, IMAGE_SIZE,
    EMBEDDING_DIM, get_device
)
from src.retrieval import (
    build_gallery_index, retrieve_top_k,
    evaluate_retrieval_metrics, compute_single_embedding
)
from src.verification import (
    calibrate_verification_threshold,
    evaluate_verification_on_test
)
from src.dataset import (
    get_eval_transforms,
    get_synthetic_color_stress_transforms
)


def benchmark_efficiency(model: torch.nn.Module, device: torch.device = None, input_size=(1, 3, 224, 224), n_runs=100) -> dict:
    """
    Measures model parameter count, disk footprint, inference latency, and throughput.
    """
    if device is None:
        device = get_device()

    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Warmup runs
    with torch.no_grad():
        for _ in range(15):
            _ = model.extract_embedding(dummy_input)

    # Timing latency over n_runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model.extract_embedding(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    return {
        "backbone": getattr(model, "backbone_name", "resnet18"),
        "embedding_dim": getattr(model, "embedding_dim", EMBEDDING_DIM),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "approx_model_size_mb": round((total_params * 4) / (1024 * 1024), 2),
        "device": str(device),
        "inference_latency_mean_ms": round(mean_latency, 3),
        "inference_latency_std_ms": round(std_latency, 3),
        "throughput_fps": round(fps, 1)
    }


def plot_similarity_distribution(val_results: dict, test_results: dict, save_path: str):
    """
    Plots Cosine Similarity Distributions for Same-Design vs Different-Design pairs.
    """
    plt.figure(figsize=(9, 5), dpi=300)
    
    y_test = test_results["y_true"]
    sims = test_results["similarities"]
    
    pos_sims = sims[y_test == 1]
    neg_sims = sims[y_test == 0]
    
    plt.hist(neg_sims, bins=35, alpha=0.6, color="#e74c3c", label=f"Different Design Pairs (N={len(neg_sims)})", density=True)
    plt.hist(pos_sims, bins=35, alpha=0.6, color="#2ecc71", label=f"Same Design Pairs (N={len(pos_sims)})", density=True)
    
    threshold = test_results.get("threshold_applied", 0.70)
    plt.axvline(threshold, color="#2c3e50", linestyle="--", linewidth=2, label=f"Validation Calibrated Threshold ($\\tau^*={threshold:.2f}$)")
    
    plt.title("Cosine Similarity Distribution: Same Design vs Different Design Pairs", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Cosine Similarity", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.legend(frameon=True, fontsize=10, loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(val_results: dict, test_results: dict, save_path: str):
    """Plots Pair Verification ROC Curve with AUC score."""
    plt.figure(figsize=(7, 6), dpi=300)
    
    fpr_test = test_results["fpr"]
    tpr_test = test_results["tpr"]
    auc_test = test_results["roc_auc"]
    
    plt.plot(fpr_test, tpr_test, color="#2980b9", linewidth=2.5, label=f"Test Verification ROC (AUC = {auc_test:.4f})")
    plt.plot([0, 1], [0, 1], color="#95a5a6", linestyle=":", label="Random Guess (AUC = 0.5000)")
    
    plt.title("Saree Design Pair Verification ROC Curve", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.legend(frameon=True, fontsize=10, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(test_results: dict, save_path: str):
    """Plots Confusion Matrix for pair verification at calibrated threshold."""
    plt.figure(figsize=(6, 5), dpi=300)
    cm = test_results["confusion_matrix"]
    matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Verification Confusion Matrix ($\\tau^*={test_results['threshold_applied']:.2f}$)", fontsize=12, fontweight="bold")
    plt.colorbar()
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Different Design', 'Same Design'], fontsize=9)
    plt.yticks(tick_marks, ['Different Design', 'Same Design'], fontsize=9)
    
    thresh = matrix.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if matrix[i, j] > thresh else "black",
                     fontsize=12, fontweight="bold")
            
    plt.ylabel('True Ground Truth Class', fontsize=10)
    plt.xlabel('Predicted Verification Class', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_retrieval_visual_grid(
    query_samples: list,
    gallery_index: dict,
    model: torch.nn.Module,
    save_path: str,
    device: torch.device = None,
    k: int = 5
):
    """Generates visual grid showing query saree images alongside Top-K retrieved matches."""
    if device is None:
        device = get_device()

    n_samples = min(len(query_samples), 4)
    fig, axes = plt.subplots(n_samples, k + 1, figsize=(16, 3.5 * n_samples), dpi=250)
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for row, (q_path, q_cat, q_did) in enumerate(query_samples[:n_samples]):
        # Query image
        q_img = Image.open(q_path).convert("RGB")
        axes[row, 0].imshow(q_img)
        axes[row, 0].set_title(f"QUERY\n{q_cat}\n[{q_did[:12]}]", fontsize=9, fontweight="bold", color="#2c3e50")
        axes[row, 0].axis("off")

        # Retrieve top k
        matches = retrieve_top_k(
            query_image=q_path,
            gallery=gallery_index,
            k=k,
            model=model,
            device=device,
            query_design_id=q_did
        )

        for col, m in enumerate(matches):
            ax = axes[row, col + 1]
            try:
                g_img = Image.open(m['gallery_path']).convert("RGB")
                ax.imshow(g_img)
            except Exception:
                ax.text(0.5, 0.5, "Image Error", ha="center")

            border_color = "#27ae60" if m['is_match'] else "#e74c3c"
            match_label = "MATCH" if m['is_match'] else "DIFF"
            
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)

            ax.set_title(f"Rank {m['rank']} ({m['similarity_score']:.2f})\n{m['category']} | {match_label}",
                         fontsize=8, fontweight="bold", color=border_color)
            ax.axis("off")

    plt.suptitle("Top-K Saree Design Retrieval Results (Standard Test)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_color_invariance_demo_grid(
    query_samples: list,
    gallery_index: dict,
    model: torch.nn.Module,
    save_path: str,
    device: torch.device = None,
    k: int = 3
):
    """
    Visual demonstration of SYNTHETIC COLOR-INVARIANCE STRESS TEST:
    Shows original query, severely color-jittered query, grayscale query,
    and the retrieved true gallery match.
    """
    if device is None:
        device = get_device()

    n_samples = min(len(query_samples), 3)
    fig, axes = plt.subplots(n_samples, 4, figsize=(15, 4 * n_samples), dpi=250)
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    jitter_transform = get_synthetic_color_stress_transforms(mode="color_jitter")
    gray_transform = get_synthetic_color_stress_transforms(mode="grayscale")

    for row, (q_path, q_cat, q_did) in enumerate(query_samples[:n_samples]):
        orig_img = Image.open(q_path).convert("RGB")
        
        # 1. Original Gallery Image
        axes[row, 0].imshow(orig_img)
        axes[row, 0].set_title(f"1. Original Reference\n{q_cat} [{q_did[:10]}]", fontsize=9, fontweight="bold")
        axes[row, 0].axis("off")

        # 2. Color-Jittered Query
        jitter_tensor = jitter_transform(orig_img)
        jitter_disp = np.clip((jitter_tensor.permute(1, 2, 0).numpy() * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]), 0, 1)
        axes[row, 1].imshow(jitter_disp)
        axes[row, 1].set_title("2. Synthetic Color Shift\n(Extreme Jitter Query)", fontsize=9, fontweight="bold", color="#d35400")
        axes[row, 1].axis("off")

        # 3. Grayscale Query
        gray_img = orig_img.convert("L").convert("RGB")
        axes[row, 2].imshow(gray_img)
        axes[row, 2].set_title("3. Pure Grayscale\n(No Color Channels Query)", fontsize=9, fontweight="bold", color="#7f8c8d")
        axes[row, 2].axis("off")

        # Retrieve top match using the color-perturbed query
        perturbed_matches = retrieve_top_k(
            query_image=jitter_tensor.unsqueeze(0).to(device),
            gallery=gallery_index,
            k=1,
            model=model,
            device=device,
            query_design_id=q_did
        )

        top_match = perturbed_matches[0]
        top_img = Image.open(top_match['gallery_path']).convert("RGB")
        axes[row, 3].imshow(top_img)
        
        border_col = "#27ae60" if top_match['is_match'] else "#e74c3c"
        for spine in axes[row, 3].spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(3)
            
        axes[row, 3].set_title(f"4. Retrieved Gallery Match\nSim: {top_match['similarity_score']:.2f} | {'SUCCESS' if top_match['is_match'] else 'MISMATCH'}",
                               fontsize=9, fontweight="bold", color=border_col)
        axes[row, 3].axis("off")

    plt.suptitle("Synthetic Color-Invariance Stress Test: Surface Pattern Retrieval Across Chromatic Variations",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
