import os
import sys
import json
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import (
    CHECKPOINT_DIR, RESULTS_DIR, METRICS_DIR, FIGURES_DIR,
    TRAIN_DIR, VALID_DIR, TEST_DIR, HANDLOOM_DIR,
    IMAGE_SIZE, EMBEDDING_DIM, PROJECTION_HIDDEN_DIM, get_device
)
from src.dataset import (
    scan_dataset_partition,
    SareeDataset,
    get_eval_transforms,
    get_synthetic_color_stress_transforms
)
from src.model import ColorInvariantSareeEncoder
from src.retrieval import build_gallery_index, evaluate_retrieval_metrics
from src.metrics import compute_accuracy, compute_precision_recall_f1, compute_roc_curve, compute_auc
from src.evaluate import (
    plot_similarity_distribution,
    plot_roc_curve,
    plot_confusion_matrix,
    generate_retrieval_visual_grid,
    generate_color_invariance_demo_grid,
    benchmark_efficiency
)

def main():
    device = get_device()
    print(f"Running comprehensive evaluation using checkpoint on {device}...")

    model = ColorInvariantSareeEncoder(
        backbone_name="resnet18",
        embedding_dim=EMBEDDING_DIM,
        projection_hidden_dim=PROJECTION_HIDDEN_DIM,
        pretrained=False
    ).to(device)

    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"Checkpoint not found at {best_ckpt}")

    checkpoint = torch.load(best_ckpt, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded weights from {best_ckpt}")

    # Dataset partitions
    train_samples = scan_dataset_partition(TRAIN_DIR)
    valid_samples = scan_dataset_partition(VALID_DIR)
    test_samples = scan_dataset_partition(TEST_DIR)

    eval_tf = get_eval_transforms()
    jitter_tf = get_synthetic_color_stress_transforms(mode="color_jitter")
    gray_tf = get_synthetic_color_stress_transforms(mode="grayscale")
    hue_tf = get_synthetic_color_stress_transforms(mode="hue_shift")

    # 1. Validation Threshold Calibration using Synthetic Color Pairs
    # For each validation design (115 designs), generate:
    # - Positive pairs: (original, color-altered view) [Same Design, Different Color]
    # - Negative pairs: (design_A, design_B with random colors) [Different Designs]
    rng_val = random.Random(42)
    val_pairs = []
    # Positive pairs (115 pairs)
    for s in valid_samples:
        img_p = s[0]
        # Alternate between jitter, gray, hue
        mode = rng_val.choice(['jitter', 'gray', 'hue'])
        val_pairs.append({'img1': img_p, 'img2': img_p, 'mode2': mode, 'label': 1, 'd1': s[2], 'd2': s[2]})

    # Negative pairs (115 pairs)
    all_val_indices = list(range(len(valid_samples)))
    for i in all_val_indices:
        j = rng_val.choice([idx for idx in all_val_indices if idx != i])
        mode = rng_val.choice(['jitter', 'gray', 'hue', 'none'])
        val_pairs.append({
            'img1': valid_samples[i][0],
            'img2': valid_samples[j][0],
            'mode2': mode,
            'label': 0,
            'd1': valid_samples[i][2],
            'd2': valid_samples[j][2]
        })

    print(f"Calibrating threshold on {len(val_pairs)} validation color-invariance pairs...")
    val_sims = []
    val_y = []
    with torch.no_grad():
        for p in val_pairs:
            im1 = Image.open(p['img1']).convert("RGB")
            t1 = eval_tf(im1).unsqueeze(0).to(device)
            im2 = Image.open(p['img2']).convert("RGB")
            if p['mode2'] == 'jitter':
                t2 = jitter_tf(im2).unsqueeze(0).to(device)
            elif p['mode2'] == 'gray':
                t2 = gray_tf(im2).unsqueeze(0).to(device)
            elif p['mode2'] == 'hue':
                t2 = hue_tf(im2).unsqueeze(0).to(device)
            else:
                t2 = eval_tf(im2).unsqueeze(0).to(device)
            
            e1 = model(t1)
            e2 = model(t2)
            sim = float(torch.sum(e1 * e2).item())
            val_sims.append(sim)
            val_y.append(p['label'])

    val_sims = np.array(val_sims)
    val_y = np.array(val_y)

    candidate_thresholds = np.linspace(-0.2, 0.99, 200)
    best_thresh = 0.5
    best_f1 = -1.0
    best_acc = 0.0

    for t in candidate_thresholds:
        preds = (val_sims >= t).astype(int)
        _, _, f1, _ = compute_precision_recall_f1(val_y, preds)
        acc = compute_accuracy(val_y, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_acc = acc

    fpr_val, tpr_val, _ = compute_roc_curve(val_y, val_sims)
    val_auc = compute_auc(fpr_val, tpr_val)
    print(f"Optimal Threshold (tau*): {best_thresh:.4f} (Val F1: {best_f1:.4f}, Val Acc: {best_acc*100:.2f}%, Val AUC: {val_auc:.4f})")

    # 2. Test Verification Evaluation on Unseen Test Designs (60 designs)
    rng_test = random.Random(999)
    test_pairs = []
    for s in test_samples:
        img_p = s[0]
        mode = rng_test.choice(['jitter', 'gray', 'hue'])
        test_pairs.append({'img1': img_p, 'img2': img_p, 'mode2': mode, 'label': 1, 'd1': s[2], 'd2': s[2]})
    
    all_test_idx = list(range(len(test_samples)))
    for i in all_test_idx:
        j = rng_test.choice([idx for idx in all_test_idx if idx != i])
        mode = rng_test.choice(['jitter', 'gray', 'hue', 'none'])
        test_pairs.append({
            'img1': test_samples[i][0],
            'img2': test_samples[j][0],
            'mode2': mode,
            'label': 0,
            'd1': test_samples[i][2],
            'd2': test_samples[j][2]
        })

    test_sims = []
    test_y = []
    with torch.no_grad():
        for p in test_pairs:
            im1 = Image.open(p['img1']).convert("RGB")
            t1 = eval_tf(im1).unsqueeze(0).to(device)
            im2 = Image.open(p['img2']).convert("RGB")
            if p['mode2'] == 'jitter':
                t2 = jitter_tf(im2).unsqueeze(0).to(device)
            elif p['mode2'] == 'gray':
                t2 = gray_tf(im2).unsqueeze(0).to(device)
            elif p['mode2'] == 'hue':
                t2 = hue_tf(im2).unsqueeze(0).to(device)
            else:
                t2 = eval_tf(im2).unsqueeze(0).to(device)
            
            e1 = model(t1)
            e2 = model(t2)
            sim = float(torch.sum(e1 * e2).item())
            test_sims.append(sim)
            test_y.append(p['label'])

    test_sims = np.array(test_sims)
    test_y = np.array(test_y)
    test_preds = (test_sims >= best_thresh).astype(int)

    precision, recall, f1, _ = compute_precision_recall_f1(test_y, test_preds)
    acc = compute_accuracy(test_y, test_preds)
    fpr_test, tpr_test, _ = compute_roc_curve(test_y, test_sims)
    roc_auc_test = compute_auc(fpr_test, tpr_test)

    tp = int(np.sum((test_preds == 1) & (test_y == 1)))
    fp = int(np.sum((test_preds == 1) & (test_y == 0)))
    tn = int(np.sum((test_preds == 0) & (test_y == 0)))
    fn = int(np.sum((test_preds == 0) & (test_y == 1)))

    print(f"\n--- TEST VERIFICATION RESULTS (Color-Invariance Test Pairs) ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc_test:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # 3. Retrieval Evaluations
    full_test_gallery_loader = DataLoader(SareeDataset(test_samples, eval_tf), batch_size=32, shuffle=False)
    stress_jitter_loader = DataLoader(SareeDataset(test_samples, jitter_tf), batch_size=32, shuffle=False)
    stress_gray_loader = DataLoader(SareeDataset(test_samples, gray_tf), batch_size=32, shuffle=False)
    stress_hue_loader = DataLoader(SareeDataset(test_samples, hue_tf), batch_size=32, shuffle=False)

    full_gallery_index = build_gallery_index(model, full_test_gallery_loader, device=device)

    stress_jitter_metrics = evaluate_retrieval_metrics(model, stress_jitter_loader, full_gallery_index, device=device)
    stress_gray_metrics = evaluate_retrieval_metrics(model, stress_gray_loader, full_gallery_index, device=device)
    stress_hue_metrics = evaluate_retrieval_metrics(model, stress_hue_loader, full_gallery_index, device=device)

    print(f"\n--- SYNTHETIC COLOR-INVARIANCE STRESS TEST RETRIEVAL ---")
    print(f"Color-Jitter: Top-1={stress_jitter_metrics['top1_accuracy']*100:.2f}%, Top-5={stress_jitter_metrics['top5_accuracy']*100:.2f}%, MRR={stress_jitter_metrics['mrr']:.4f}")
    print(f"Grayscale:    Top-1={stress_gray_metrics['top1_accuracy']*100:.2f}%, Top-5={stress_gray_metrics['top5_accuracy']*100:.2f}%, MRR={stress_gray_metrics['mrr']:.4f}")
    print(f"Hue-Shift:    Top-1={stress_hue_metrics['top1_accuracy']*100:.2f}%, Top-5={stress_hue_metrics['top5_accuracy']*100:.2f}%, MRR={stress_hue_metrics['mrr']:.4f}")

    # 4. Benchmarking Efficiency
    efficiency_res = benchmark_efficiency(model, device=device)
    print(f"\n--- EFFICIENCY ---")
    print(f"Parameters:        {efficiency_res['total_parameters']:,}")
    print(f"Embedding Dim:     {efficiency_res['embedding_dim']}")
    print(f"Model Disk Size:   ~{efficiency_res['approx_model_size_mb']} MB")
    print(f"Inference Latency: {efficiency_res['inference_latency_mean_ms']} ms/img")
    print(f"Throughput:        {efficiency_res['throughput_fps']} FPS")

    # 5. Diagnostic Plots
    calib_dict = {
        "optimal_threshold": best_thresh,
        "best_val_f1": best_f1,
        "best_val_acc": best_acc,
        "val_roc_auc": val_auc,
        "y_true": val_y,
        "similarities": val_sims,
        "same_design_similarities": val_sims[val_y == 1].tolist(),
        "diff_design_similarities": val_sims[val_y == 0].tolist(),
        "fpr": fpr_val,
        "tpr": tpr_val
    }
    test_verif_dict = {
        "num_pairs": len(test_pairs),
        "threshold_applied": float(best_thresh),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc_test),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "same_design_similarities": test_sims[test_y == 1].tolist(),
        "diff_design_similarities": test_sims[test_y == 0].tolist(),
        "y_true": test_y,
        "similarities": test_sims,
        "fpr": fpr_test,
        "tpr": tpr_test
    }

    for save_dir in [FIGURES_DIR, RESULTS_DIR]:
        plot_similarity_distribution(calib_dict, test_verif_dict, os.path.join(save_dir, "similarity_distribution.png"))
        plot_roc_curve(calib_dict, test_verif_dict, os.path.join(save_dir, "roc_curve.png"))
        plot_confusion_matrix(test_verif_dict, os.path.join(save_dir, "confusion_matrix.png"))
        generate_retrieval_visual_grid(test_samples[:10], full_gallery_index, model, os.path.join(save_dir, "retrieval_examples.png"), device=device)
        generate_color_invariance_demo_grid(test_samples, full_gallery_index, model, os.path.join(save_dir, "synthetic_color_invariance_demo.png"), device=device)

    # 6. Save comprehensive results JSON & CSV
    # Load training history from existing evaluation_results.json
    prev_results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(prev_results_path, "r") as f:
        prev_data = json.load(f)

    all_results = {
        "training_summary": prev_data["training_summary"],
        "efficiency": efficiency_res,
        "verification_validation_calibration": {
            "optimal_threshold": float(best_thresh),
            "best_val_f1": float(best_f1),
            "best_val_acc": float(best_acc),
            "val_roc_auc": float(val_auc)
        },
        "verification_test_evaluation": {
            "num_pairs": len(test_pairs),
            "threshold_applied": float(best_thresh),
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc_test),
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
        },
        "retrieval_standard_test": prev_data["retrieval_standard_test"],
        "synthetic_color_invariance_stress_test": {
            "color_jitter": stress_jitter_metrics,
            "grayscale": stress_gray_metrics,
            "hue_shift": stress_hue_metrics
        }
    }

    for target_dir in [RESULTS_DIR, METRICS_DIR]:
        with open(os.path.join(target_dir, "evaluation_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    summary_df = pd.DataFrame([
        {"Category": "Training", "Metric": "Total Training Time", "Value": f"{prev_data['training_summary']['total_train_time_min']:.2f} min"},
        {"Category": "Training", "Metric": "Best Validation Loss", "Value": f"{prev_data['training_summary']['best_val_loss']:.4f}"},
        {"Category": "Training", "Metric": "Final Train Loss", "Value": f"{prev_data['training_summary']['final_train_loss']:.4f}"},
        {"Category": "Pair Verification", "Metric": "Calibrated Threshold (tau*)", "Value": f"{best_thresh:.4f}"},
        {"Category": "Pair Verification", "Metric": "Accuracy", "Value": f"{acc*100:.2f}%"},
        {"Category": "Pair Verification", "Metric": "Precision", "Value": f"{precision*100:.2f}%"},
        {"Category": "Pair Verification", "Metric": "Recall", "Value": f"{recall*100:.2f}%"},
        {"Category": "Pair Verification", "Metric": "F1-Score", "Value": f"{f1*100:.2f}%"},
        {"Category": "Pair Verification", "Metric": "ROC-AUC", "Value": f"{roc_auc_test:.4f}"},
        {"Category": "Synthetic Stress (Jitter)", "Metric": "Top-1 Accuracy", "Value": f"{stress_jitter_metrics['top1_accuracy']*100:.2f}%"},
        {"Category": "Synthetic Stress (Jitter)", "Metric": "Top-5 Accuracy", "Value": f"{stress_jitter_metrics['top5_accuracy']*100:.2f}%"},
        {"Category": "Synthetic Stress (Jitter)", "Metric": "MRR", "Value": f"{stress_jitter_metrics['mrr']:.4f}"},
        {"Category": "Synthetic Stress (Grayscale)", "Metric": "Top-1 Accuracy", "Value": f"{stress_gray_metrics['top1_accuracy']*100:.2f}%"},
        {"Category": "Synthetic Stress (Grayscale)", "Metric": "Top-5 Accuracy", "Value": f"{stress_gray_metrics['top5_accuracy']*100:.2f}%"},
        {"Category": "Synthetic Stress (Grayscale)", "Metric": "MRR", "Value": f"{stress_gray_metrics['mrr']:.4f}"},
        {"Category": "Synthetic Stress (Hue Shift)", "Metric": "Top-1 Accuracy", "Value": f"{stress_hue_metrics['top1_accuracy']*100:.2f}%"},
        {"Category": "Synthetic Stress (Hue Shift)", "Metric": "Top-5 Accuracy", "Value": f"{stress_hue_metrics['top5_accuracy']*100:.2f}%"},
        {"Category": "Synthetic Stress (Hue Shift)", "Metric": "MRR", "Value": f"{stress_hue_metrics['mrr']:.4f}"},
        {"Category": "Efficiency", "Metric": "Total Parameters", "Value": f"{efficiency_res['total_parameters']:,}"},
        {"Category": "Efficiency", "Metric": "Embedding Dimension", "Value": f"{efficiency_res['embedding_dim']}"},
        {"Category": "Efficiency", "Metric": "Model Size", "Value": f"~{efficiency_res['approx_model_size_mb']} MB"},
        {"Category": "Efficiency", "Metric": "Inference Latency", "Value": f"{efficiency_res['inference_latency_mean_ms']} ms"},
        {"Category": "Efficiency", "Metric": "Throughput", "Value": f"{efficiency_res['throughput_fps']} FPS"}
    ])

    for target_dir in [RESULTS_DIR, METRICS_DIR]:
        summary_df.to_csv(os.path.join(target_dir, "summary_metrics.csv"), index=False)

    print("\nSaved updated evaluation_results.json and summary_metrics.csv.")
    print("=" * 75)
    print("COMPREHENSIVE EVALUATION COMPLETE!")
    print("=" * 75)

if __name__ == "__main__":
    main()
