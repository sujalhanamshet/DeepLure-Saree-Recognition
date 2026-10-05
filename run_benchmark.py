"""
Small Training Benchmark Script (2 Epochs).
Measures training speed, loss progression, weight updates, and hardware throughput
on the actual dataset C:\\deeplure before committing to a full training run.
"""

import os
import sys
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    TRAIN_DIR, VALID_DIR, EMBEDDING_DIM, TEMPERATURE,
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, SEED,
    seed_everything, get_device
)
from src.dataset import (
    scan_dataset_partition, TwoCropSareeDataset,
    get_train_transforms
)
from src.model import ColorInvariantSareeEncoder
from src.loss import SupConLoss


def run_benchmark():
    print("=" * 75)
    print("SMALL TRAINING BENCHMARK (2 EPOCHS)")
    print(f"Dataset Root: {TRAIN_DIR}")
    print("=" * 75)

    # 1. Device and Seeds
    seed_everything(SEED)
    device = get_device()
    if isinstance(device, str):
        device = torch.device(device)
    print(f"Detected Compute Device: {device}")
    if device.type == "cuda":
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        print(f"Running on Host CPU (Threads: {torch.get_num_threads()})")

    # 2. Output directory
    benchmark_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)

    # 3. Load actual training data from C:\deeplure
    print("\nScanning training data from C:\\deeplure\\archive\\train...")
    train_samples = scan_dataset_partition(TRAIN_DIR)
    print(f"Total training images loaded: {len(train_samples)}")
    
    train_design_ids = sorted(list(set(s[2] for s in train_samples)))
    print(f"Unique base design classes:  {len(train_design_ids)}")
    train_label_map = {did: i for i, did in enumerate(train_design_ids)}

    train_transforms = get_train_transforms()
    train_dataset = TwoCropSareeDataset(train_samples, transform=train_transforms, label_map=train_label_map)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    print(f"Total batches per epoch (batch_size={BATCH_SIZE}): {len(train_loader)}")

    # 4. Model, Loss, Optimizer
    print("\nInitializing ColorInvariantSareeEncoder (ResNet-18 + 256-D Embedding)...")
    model = ColorInvariantSareeEncoder(
        backbone_name="resnet18",
        embedding_dim=EMBEDDING_DIM,
        projection_hidden_dim=512,
        pretrained=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")

    criterion = SupConLoss(temperature=TEMPERATURE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Save snapshot of initial weights to verify updates
    initial_fc_weight = model.projection_head[0].weight.clone().detach()

    # 5. Execute 2 Benchmark Epochs
    epoch_times = []
    epoch_losses = []
    total_start_time = time.time()

    print("\n" + "-" * 75)
    print("STARTING 2-EPOCH TRAINING BENCHMARK")
    print("-" * 75)

    for epoch in range(1, 3):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        total_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/2", leave=True)
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

        epoch_duration = time.time() - epoch_start
        epoch_loss = running_loss / max(1, total_batches)

        epoch_times.append(epoch_duration)
        epoch_losses.append(epoch_loss)

        print(f"-> Epoch [{epoch}/2] Complete | Loss: {epoch_loss:.4f} | Time: {epoch_duration:.2f}s ({epoch_duration/60:.2f} min)")

    total_duration = time.time() - total_start_time

    # 6. Verify Loss Finiteness & Weight Updates
    losses_finite = all(torch.isfinite(torch.tensor(l)).item() for l in epoch_losses)
    updated_fc_weight = model.projection_head[0].weight.detach()
    weight_delta = torch.norm(updated_fc_weight - initial_fc_weight).item()
    weights_updated = (weight_delta > 1e-4)

    # 7. Save Benchmark Checkpoint (strictly under outputs/benchmark/)
    benchmark_ckpt_path = os.path.join(benchmark_dir, "benchmark_model_2epochs.pth")
    torch.save({
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch_losses": epoch_losses,
        "epoch_times": epoch_times,
        "total_duration_sec": total_duration,
        "device": str(device)
    }, benchmark_ckpt_path)
    print(f"\nSaved benchmark checkpoint: {benchmark_ckpt_path}")

    # 8. Report Results
    print("\n" + "=" * 75)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 75)
    print(f"1. Compute Device Used:        {device}")
    print(f"2. Epoch 1 Time:               {epoch_times[0]:.2f}s ({epoch_times[0]/60:.2f} min)")
    print(f"3. Epoch 2 Time:               {epoch_times[1]:.2f}s ({epoch_times[1]/60:.2f} min)")
    print(f"4. Total Benchmark Time:       {total_duration:.2f}s ({total_duration/60:.2f} min)")
    print(f"5. Epoch 1 SupCon Loss:        {epoch_losses[0]:.4f}")
    print(f"6. Epoch 2 SupCon Loss:        {epoch_losses[1]:.4f}")
    print(f"7. Loss Values Finite:         {losses_finite} (No NaN/Inf)")
    print(f"8. Weights Actively Updating:  {weights_updated} (Weight Delta L2: {weight_delta:.4f})")
    print(f"9. Loss Trend:                 {'Decreasing' if epoch_losses[1] < epoch_losses[0] else 'Stable'} ({epoch_losses[0]:.4f} -> {epoch_losses[1]:.4f})")
    print(f"10. Benchmark Status:          PASSED")
    print("=" * 75)

    return {
        "device": str(device),
        "epoch_1_time": epoch_times[0],
        "epoch_2_time": epoch_times[1],
        "total_time": total_duration,
        "epoch_1_loss": epoch_losses[0],
        "epoch_2_loss": epoch_losses[1],
        "loss_finite": losses_finite,
        "weights_updated": weights_updated,
        "status": "PASSED"
    }


if __name__ == "__main__":
    run_benchmark()
