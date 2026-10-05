"""
End-to-End Training and Complete Evaluation Pipeline for Color-Invariant Saree Recognition.
Trains ColorInvariantSareeEncoder on C:\\deeplure for 8 epochs, then immediately executes
the complete evaluation suite (Retrieval, Verification, Synthetic Color-Invariance Stress Testing,
Efficiency Benchmarking, and Publication Plot Generation).
"""

import os
import sys
import json
import time
import random
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR, HANDLOOM_DIR,
    CHECKPOINT_DIR, METRICS_DIR, RESULTS_DIR, FIGURES_DIR,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    TEMPERATURE, EMBEDDING_DIM, BACKBONE_NAME, SEED,
    seed_everything, get_device
)
from src.dataset import (
    scan_dataset_partition, TwoCropSareeDataset, SareeDataset,
    get_train_transforms, get_eval_transforms, get_synthetic_color_stress_transforms,
    build_verification_pairs
)
from src.model import ColorInvariantSareeEncoder
from src.loss import SupConLoss
from src.retrieval import build_gallery_index, evaluate_retrieval_metrics
from src.verification import calibrate_verification_threshold, evaluate_verification_on_test
from src.evaluate import (
    benchmark_efficiency, plot_similarity_distribution,
    plot_roc_curve, plot_confusion_matrix,
    generate_retrieval_visual_grid, generate_color_invariance_demo_grid
)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: SupConLoss,
    device: torch.device
) -> float:
    """Trains model for one epoch using Supervised Contrastive Loss across two views."""
    model.train()
    running_loss = 0.0
    total_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for views, labels, _, _, _ in pbar:
        v1, v2 = views[0].to(device), views[1].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        z1 = model(v1)
        z2 = model(v2)
        features = torch.stack([z1, z2], dim=1)  # (B, 2, 256)

        loss = criterion(features, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / max(1, total_batches)


def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: SupConLoss,
    device: torch.device
) -> float:
    """Computes validation contrastive loss."""
    model.eval()
    running_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for views, labels, _, _, _ in dataloader:
            v1, v2 = views[0].to(device), views[1].to(device)
            labels = labels.to(device)

            z1 = model(v1)
            z2 = model(v2)
            features = torch.stack([z1, z2], dim=1)

            loss = criterion(features, labels=labels)
            running_loss += loss.item()
            total_batches += 1

    return running_loss / max(1, total_batches)


def run_full_training_and_evaluation():
    print("=" * 75)
    print("FULL TRAINING AND COMPLETE EVALUATION PIPELINE")
    print(f"Dataset Root: C:\\deeplure\\archive")
    print(f"Epochs to Train: {NUM_EPOCHS}")
    print("=" * 75)

    # 1. Reproducibility & Compute Device
    seed_everything(SEED)
    device = get_device()
    print(f"Active Compute Device: {device}")

    # 2. Scan Dataset Partitions from C:\deeplure
    print("\n--- 1. Scanning Dataset Partitions ---")
    train_samples = scan_dataset_partition(TRAIN_DIR)
    valid_samples = scan_dataset_partition(VALID_DIR)
    test_samples = scan_dataset_partition(TEST_DIR)

    print(f"Training samples:   {len(train_samples)} images")
    print(f"Validation samples: {len(valid_samples)} images")
    print(f"Test samples:       {len(test_samples)} images")

    # Build unique design label map for training
    train_design_ids = sorted(list(set(s[2] for s in train_samples)))
    train_label_map = {did: idx for idx, did in enumerate(train_design_ids)}
    print(f"Unique training design classes: {len(train_design_ids)}")

    valid_design_ids = sorted(list(set(s[2] for s in valid_samples)))
    valid_label_map = {did: idx for idx, did in enumerate(valid_design_ids)}

    test_design_ids = sorted(list(set(s[2] for s in test_samples)))
    test_label_map = {did: idx for idx, did in enumerate(test_design_ids)}

    # 3. Create Datasets & DataLoaders
    train_transform = get_train_transforms()
    eval_transform = get_eval_transforms()

    train_dataset = TwoCropSareeDataset(train_samples, transform=train_transform, label_map=train_label_map)
    valid_dataset = TwoCropSareeDataset(valid_samples, transform=train_transform, label_map=valid_label_map)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)

    # Standard Test Gallery & Query Setup:
    # 60 test images -> 30 gallery, 30 query
    rng = random.Random(SEED)
    shuffled_test = list(test_samples)
    rng.shuffle(shuffled_test)
    half = len(shuffled_test) // 2
    test_gallery_samples = shuffled_test[:half]
    test_query_samples = shuffled_test[half:]

    test_gallery_loader = DataLoader(SareeDataset(test_gallery_samples, transform=eval_transform, label_map=test_label_map), batch_size=32, shuffle=False)
    test_query_loader = DataLoader(SareeDataset(test_query_samples, transform=eval_transform, label_map=test_label_map), batch_size=32, shuffle=False)

    # Synthetic Color-Invariance Stress Test Loaders
    # The original 60 test images form the gallery, and their color-transformed versions form the queries
    full_test_gallery_loader = DataLoader(SareeDataset(test_samples, transform=eval_transform, label_map=test_label_map), batch_size=32, shuffle=False)
    
    stress_jitter_loader = DataLoader(
        SareeDataset(test_samples, transform=get_synthetic_color_stress_transforms(mode="color_jitter"), label_map=test_label_map),
        batch_size=32, shuffle=False
    )
    stress_gray_loader = DataLoader(
        SareeDataset(test_samples, transform=get_synthetic_color_stress_transforms(mode="grayscale"), label_map=test_label_map),
        batch_size=32, shuffle=False
    )
    stress_hue_loader = DataLoader(
        SareeDataset(test_samples, transform=get_synthetic_color_stress_transforms(mode="hue_shift"), label_map=test_label_map),
        batch_size=32, shuffle=False
    )

    # Validation pairs for threshold calibration
    val_pairs = build_verification_pairs(valid_samples, num_pairs=300, seed=SEED)
    test_pairs = build_verification_pairs(test_samples, num_pairs=400, seed=SEED)

    # 4. Initialize Model, Loss, Optimizer
    print("\n--- 2. Initializing ColorInvariantSareeEncoder ---")
    model = ColorInvariantSareeEncoder(
        backbone_name=BACKBONE_NAME,
        embedding_dim=EMBEDDING_DIM,
        projection_hidden_dim=512,
        pretrained=True
    ).to(device)

    summary = model.get_parameter_summary()
    print(f"Backbone:             {summary['backbone']}")
    print(f"Embedding Dimension:  {summary['embedding_dim']}")
    print(f"Total Parameters:     {summary['total_parameters']:,}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"Model Size on Disk:   ~{summary['approx_size_mb']:.2f} MB")

    criterion = SupConLoss(temperature=TEMPERATURE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # 5. Training Loop for exactly 8 epochs
    print(f"\n--- 3. Starting Training for Exactly {NUM_EPOCHS} Epochs ---")
    best_val_loss = float("inf")
    history = []
    training_start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, valid_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr_curr = scheduler.get_last_lr()[0]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr_curr,
            "time_sec": elapsed
        })

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr_curr:.6f} | Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "config": {
                    "backbone": BACKBONE_NAME,
                    "embedding_dim": EMBEDDING_DIM,
                    "temperature": TEMPERATURE
                }
            }, best_ckpt_path)

    total_train_time = time.time() - training_start_time

    # Save final model checkpoint
    final_ckpt_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    torch.save({
        "epoch": NUM_EPOCHS,
        "model_state_dict": model.state_dict(),
        "final_val_loss": val_loss,
        "final_train_loss": train_loss,
        "total_train_time_sec": total_train_time
    }, final_ckpt_path)
    print(f"\nTraining Complete in {total_train_time:.2f}s ({total_train_time/60:.2f} min).")
    print(f"Saved: {best_ckpt_path}")
    print(f"Saved: {final_ckpt_path}")

    # Load best checkpoint for evaluation
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # 6. Verification Threshold Calibration (Validation Data ONLY)
    print("\n--- 4. Calibrating Verification Threshold on Validation Set ---")
    calib_res = calibrate_verification_threshold(val_pairs, model, device=device)
    tau_star = calib_res["optimal_threshold"]
    print(f"Optimal Verification Threshold (tau*): {tau_star:.4f}")
    print(f"Validation F1-Score at tau*:           {calib_res['best_val_f1']:.4f}")
    print(f"Validation ROC-AUC:                     {calib_res['val_roc_auc']:.4f}")

    # 7. Evaluate Verification on Test Set
    print("\n--- 5. Evaluating Pair Verification on Test Set ---")
    test_verif_res = evaluate_verification_on_test(test_pairs, threshold=tau_star, model=model, device=device)
    print(f"Test Accuracy:  {test_verif_res['accuracy'] * 100:.2f}%")
    print(f"Test Precision: {test_verif_res['precision'] * 100:.2f}%")
    print(f"Test Recall:    {test_verif_res['recall'] * 100:.2f}%")
    print(f"Test F1-Score:  {test_verif_res['f1_score'] * 100:.2f}%")
    print(f"Test ROC-AUC:   {test_verif_res['roc_auc']:.4f}")
    print(f"Confusion Matrix: {test_verif_res['confusion_matrix']}")

    # 8. Standard Gallery-Query Retrieval Evaluation
    print("\n--- 6. Evaluating Standard Gallery-Query Retrieval ---")
    test_gallery_index = build_gallery_index(model, test_gallery_loader, device=device)
    std_retrieval_metrics = evaluate_retrieval_metrics(model, test_query_loader, test_gallery_index, device=device)
    print(f"Standard Top-1 Accuracy: {std_retrieval_metrics['top1_accuracy'] * 100:.2f}%")
    print(f"Standard Top-5 Accuracy: {std_retrieval_metrics['top5_accuracy'] * 100:.2f}%")
    print(f"Standard MRR:            {std_retrieval_metrics['mrr']:.4f}")

    # 9. Synthetic Color-Invariance Stress Testing
    print("\n--- 7. Evaluating SYNTHETIC COLOR-INVARIANCE STRESS TEST ---")
    full_gallery_index = build_gallery_index(model, full_test_gallery_loader, device=device)

    stress_jitter_metrics = evaluate_retrieval_metrics(model, stress_jitter_loader, full_gallery_index, device=device)
    print(f"Color-Jittered Query Top-1 Accuracy: {stress_jitter_metrics['top1_accuracy'] * 100:.2f}%")
    print(f"Color-Jittered Query Top-5 Accuracy: {stress_jitter_metrics['top5_accuracy'] * 100:.2f}%")
    print(f"Color-Jittered Query MRR:            {stress_jitter_metrics['mrr']:.4f}")

    stress_gray_metrics = evaluate_retrieval_metrics(model, stress_gray_loader, full_gallery_index, device=device)
    print(f"Grayscale Query Top-1 Accuracy:      {stress_gray_metrics['top1_accuracy'] * 100:.2f}%")
    print(f"Grayscale Query Top-5 Accuracy:      {stress_gray_metrics['top5_accuracy'] * 100:.2f}%")
    print(f"Grayscale Query MRR:                 {stress_gray_metrics['mrr']:.4f}")

    stress_hue_metrics = evaluate_retrieval_metrics(model, stress_hue_loader, full_gallery_index, device=device)
    print(f"Hue-Shift Query Top-1 Accuracy:      {stress_hue_metrics['top1_accuracy'] * 100:.2f}%")
    print(f"Hue-Shift Query Top-5 Accuracy:      {stress_hue_metrics['top5_accuracy'] * 100:.2f}%")
    print(f"Hue-Shift Query MRR:                 {stress_hue_metrics['mrr']:.4f}")

    # 10. Computational Efficiency Benchmark
    print("\n--- 8. Benchmarking Computational Efficiency ---")
    efficiency_res = benchmark_efficiency(model, device=device)
    print(f"Parameters:        {efficiency_res['total_parameters']:,}")
    print(f"Embedding Dim:     {efficiency_res['embedding_dim']}")
    print(f"Model Disk Size:   ~{efficiency_res['approx_model_size_mb']} MB")
    print(f"Inference Latency: {efficiency_res['inference_latency_mean_ms']} ± {efficiency_res['inference_latency_std_ms']} ms/image")
    print(f"Throughput:        {efficiency_res['throughput_fps']} FPS")

    # 11. Plot Generation (Saved to outputs/figures/ and outputs/results/)
    print("\n--- 9. Generating Diagnostic Visualizations ---")
    for save_dir in [FIGURES_DIR, RESULTS_DIR]:
        sim_dist_path = os.path.join(save_dir, "similarity_distribution.png")
        plot_similarity_distribution(calib_res, test_verif_res, sim_dist_path)

        roc_path = os.path.join(save_dir, "roc_curve.png")
        plot_roc_curve(calib_res, test_verif_res, roc_path)

        cm_path = os.path.join(save_dir, "confusion_matrix.png")
        plot_confusion_matrix(test_verif_res, cm_path)

        retrieval_grid_path = os.path.join(save_dir, "retrieval_examples.png")
        generate_retrieval_visual_grid(test_query_samples, test_gallery_index, model, retrieval_grid_path, device=device)

        color_demo_path = os.path.join(save_dir, "synthetic_color_invariance_demo.png")
        generate_color_invariance_demo_grid(test_samples, full_gallery_index, model, color_demo_path, device=device)

    print("Saved all figures to outputs/figures/ and outputs/results/.")

    # 12. Metric Export to JSON and CSV
    print("\n--- 10. Exporting Evaluation Results to outputs/results/ ---")
    all_results = {
        "training_summary": {
            "num_epochs": NUM_EPOCHS,
            "total_train_time_sec": total_train_time,
            "total_train_time_min": total_train_time / 60.0,
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "best_val_loss": best_val_loss,
            "history": history
        },
        "efficiency": efficiency_res,
        "verification_validation_calibration": {
            "optimal_threshold": calib_res["optimal_threshold"],
            "best_val_f1": calib_res["best_val_f1"],
            "best_val_acc": calib_res["best_val_acc"],
            "val_roc_auc": calib_res["val_roc_auc"]
        },
        "verification_test_evaluation": {
            "num_pairs": test_verif_res["num_pairs"],
            "threshold_applied": test_verif_res["threshold_applied"],
            "accuracy": test_verif_res["accuracy"],
            "precision": test_verif_res["precision"],
            "recall": test_verif_res["recall"],
            "f1_score": test_verif_res["f1_score"],
            "roc_auc": test_verif_res["roc_auc"],
            "confusion_matrix": test_verif_res["confusion_matrix"]
        },
        "retrieval_standard_test": std_retrieval_metrics,
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
        {"Category": "Training", "Metric": "Total Training Time", "Value": f"{total_train_time/60:.2f} min"},
        {"Category": "Training", "Metric": "Best Validation Loss", "Value": f"{best_val_loss:.4f}"},
        {"Category": "Training", "Metric": "Final Train Loss", "Value": f"{history[-1]['train_loss']:.4f}"},
        {"Category": "Pair Verification", "Metric": "Calibrated Threshold (tau*)", "Value": f"{tau_star:.4f}"},
        {"Category": "Pair Verification", "Metric": "Accuracy", "Value": f"{test_verif_res['accuracy']*100:.2f}%"},
        {"Category": "Pair Verification", "Metric": "Precision", "Value": f"{test_verif_res['precision']*100:.2f}%"},
        {"Category": "Pair Verification", "Metric": "Recall", "Value": f"{test_verif_res['recall']*100:.2f}%"},
        {"Category": "Pair Verification", "Metric": "F1-Score", "Value": f"{test_verif_res['f1_score']*100:.2f}%"},
        {"Category": "Pair Verification", "Metric": "ROC-AUC", "Value": f"{test_verif_res['roc_auc']:.4f}"},
        {"Category": "Standard Retrieval", "Metric": "Top-1 Accuracy", "Value": f"{std_retrieval_metrics['top1_accuracy']*100:.2f}%"},
        {"Category": "Standard Retrieval", "Metric": "Top-5 Accuracy", "Value": f"{std_retrieval_metrics['top5_accuracy']*100:.2f}%"},
        {"Category": "Standard Retrieval", "Metric": "MRR", "Value": f"{std_retrieval_metrics['mrr']:.4f}"},
        {"Category": "Synthetic Stress (Jitter)", "Metric": "Top-1 Accuracy", "Value": f"{stress_jitter_metrics['top1_accuracy']*100:.2f}%"},
        {"Category": "Synthetic Stress (Jitter)", "Metric": "Top-5 Accuracy", "Value": f"{stress_jitter_metrics['top5_accuracy']*100:.2f}%"},
        {"Category": "Synthetic Stress (Jitter)", "Metric": "MRR", "Value": f"{stress_jitter_metrics['mrr']:.4f}"},
        {"Category": "Synthetic Stress (Grayscale)", "Metric": "Top-1 Accuracy", "Value": f"{stress_gray_metrics['top1_accuracy']*100:.2f}%"},
        {"Category": "Synthetic Stress (Grayscale)", "Metric": "Top-5 Accuracy", "Value": f"{stress_gray_metrics['top5_accuracy']*100:.2f}%"},
        {"Category": "Synthetic Stress (Grayscale)", "Metric": "MRR", "Value": f"{stress_gray_metrics['mrr']:.4f}"},
        {"Category": "Synthetic Stress (Hue Shift)", "Metric": "Top-1 Accuracy", "Value": f"{stress_hue_metrics['top1_accuracy']*100:.2f}%"},
        {"Category": "Synthetic Stress (Hue Shift)", "Metric": "Top-5 Accuracy", "Value": f"{stress_hue_metrics['top5_accuracy']*100:.2f}%"},
        {"Category": "Efficiency", "Metric": "Total Parameters", "Value": f"{efficiency_res['total_parameters']:,}"},
        {"Category": "Efficiency", "Metric": "Embedding Dimension", "Value": f"{efficiency_res['embedding_dim']}"},
        {"Category": "Efficiency", "Metric": "Model Size", "Value": f"~{efficiency_res['approx_model_size_mb']} MB"},
        {"Category": "Efficiency", "Metric": "Inference Latency", "Value": f"{efficiency_res['inference_latency_mean_ms']} ms"},
        {"Category": "Efficiency", "Metric": "Throughput", "Value": f"{efficiency_res['throughput_fps']} FPS"}
    ])

    for target_dir in [RESULTS_DIR, METRICS_DIR]:
        summary_df.to_csv(os.path.join(target_dir, "summary_metrics.csv"), index=False)

    print("\nSaved evaluation_results.json and summary_metrics.csv under outputs/results/ and outputs/metrics/.")
    print("=" * 75)
    print("FULL TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 75)

    return all_results


if __name__ == "__main__":
    run_full_training_and_evaluation()
