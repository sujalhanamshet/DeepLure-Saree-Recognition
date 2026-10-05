"""
Sanity Check Script for Color-Invariant Saree Recognition.
Executes the 10 mandatory pre-training checks:
1. Load 16 training samples from C:\\deeplure\\archive\\train.
2. Create two independently augmented views.
3. Print tensor shapes.
4. Run both views through ColorInvariantSareeEncoder.
5. Confirm embedding shape is [16, 256].
6. Confirm embeddings are finite (no NaN / Inf).
7. Calculate Supervised Contrastive Loss (SupConLoss).
8. Confirm loss value is finite.
9. Run one optimizer step (AdamW).
10. Confirm model parameters change.
"""

import os
import sys
import torch
import torch.optim as optim

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR, HANDLOOM_DIR,
    EMBEDDING_DIM, TEMPERATURE, LEARNING_RATE, WEIGHT_DECAY,
    seed_everything, get_device
)
from src.dataset import (
    scan_dataset_partition, TwoCropSareeDataset,
    get_train_transforms
)
from src.model import ColorInvariantSareeEncoder
from src.loss import SupConLoss


def run_sanity_checks():
    print("=" * 75)
    print("MANDATORY PRE-TRAINING SANITY CHECKS")
    print("=" * 75)

    seed_everything(42)
    device = get_device()
    print(f"Active Device: {device}")

    # Check 1: Scan training partition directly from C:\deeplure
    print("\n[Check 1] Scanning training samples from C:\\deeplure\\archive\\train...")
    train_samples = scan_dataset_partition(TRAIN_DIR)
    print(f"Found {len(train_samples)} total training samples.")
    assert len(train_samples) == 1293, f"Expected 1293 samples, found {len(train_samples)}"

    train_design_ids = sorted(list(set(s[2] for s in train_samples)))
    print(f"Found {len(train_design_ids)} unique design identities.")
    assert len(train_design_ids) == 431, f"Expected 431 designs, found {len(train_design_ids)}"
    train_label_map = {did: i for i, did in enumerate(train_design_ids)}

    # Select exactly 16 samples for sanity check
    batch_16_samples = train_samples[:16]
    print(f"Loaded 16 training samples successfully: {[os.path.basename(s[0])[:20] for s in batch_16_samples[:3]]} ...")

    # Check 2 & 3: Create two augmented views & verify shapes
    print("\n[Check 2 & 3] Generating two independently augmented views...")
    train_transform = get_train_transforms()
    sanity_dataset = TwoCropSareeDataset(batch_16_samples, transform=train_transform, label_map=train_label_map)
    
    loader = torch.utils.data.DataLoader(sanity_dataset, batch_size=16, shuffle=False)
    views, labels, cats, dids, paths = next(iter(loader))

    view1 = views[0].to(device)
    view2 = views[1].to(device)
    labels = labels.to(device)

    print(f"View 1 Tensor Shape: {view1.shape} (Expected: [16, 3, 224, 224])")
    print(f"View 2 Tensor Shape: {view2.shape} (Expected: [16, 3, 224, 224])")
    print(f"Labels Tensor Shape: {labels.shape} (Values: {labels.tolist()[:8]}...)")
    assert view1.shape == (16, 3, 224, 224), f"Unexpected view1 shape {view1.shape}"
    assert view2.shape == (16, 3, 224, 224), f"Unexpected view2 shape {view2.shape}"

    # Check 4: Instantiate Model and parameter summary
    print("\n[Check 4] Initializing ColorInvariantSareeEncoder (ResNet-18 backbone)...")
    model = ColorInvariantSareeEncoder(
        backbone_name="resnet18",
        embedding_dim=EMBEDDING_DIM,
        projection_hidden_dim=512,
        pretrained=True
    ).to(device)

    summary = model.get_parameter_summary()
    print(f"Total Parameters:     {summary['total_parameters']:,}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"Model Disk Footprint: ~{summary['approx_size_mb']:.2f} MB")

    # Check 5 & 6: Run both views and check embedding shape & finiteness
    print("\n[Check 5 & 6] Extracting L2-normalized embeddings for both views...")
    emb1 = model(view1)
    emb2 = model(view2)

    print(f"Embedding 1 Shape: {emb1.shape} (Expected: [16, 256])")
    print(f"Embedding 2 Shape: {emb2.shape} (Expected: [16, 256])")
    assert emb1.shape == (16, 256), f"Unexpected emb1 shape {emb1.shape}"
    assert emb2.shape == (16, 256), f"Unexpected emb2 shape {emb2.shape}"

    # Check finiteness (no NaN / Inf)
    is_finite1 = torch.isfinite(emb1).all().item()
    is_finite2 = torch.isfinite(emb2).all().item()
    print(f"Embedding 1 all finite: {is_finite1}")
    print(f"Embedding 2 all finite: {is_finite2}")
    assert is_finite1 and is_finite2, "Embeddings contain NaN or Inf values!"

    # Check L2 norm equals 1.0
    norm1 = torch.norm(emb1, p=2, dim=1)
    norm2 = torch.norm(emb2, p=2, dim=1)
    print(f"Embedding 1 L2 norms (sample): {norm1[:4].detach().cpu().numpy()}")
    assert torch.allclose(norm1, torch.ones_like(norm1), atol=1e-4), "Embeddings are not L2 normalized!"

    # Check 7 & 8: Supervised Contrastive Loss computation
    print("\n[Check 7 & 8] Computing Supervised Contrastive Loss (SupCon)...")
    criterion = SupConLoss(temperature=TEMPERATURE)
    features = torch.stack([emb1, emb2], dim=1)  # Shape: (16, 2, 256)
    print(f"Stacked Multi-View Features Shape: {features.shape}")

    loss = criterion(features, labels=labels)
    loss_val = loss.item()
    print(f"SupCon Loss Value: {loss_val:.4f}")
    assert torch.isfinite(loss).item(), "Loss is NaN or Inf!"
    print(f"Loss is finite: True")

    # Check 9 & 10: Optimizer step & parameter change confirmation
    print("\n[Check 9 & 10] Running backward pass and optimizer step...")
    # Clone initial weight tensor of projection head
    initial_weight = model.projection_head[0].weight.clone()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    updated_weight = model.projection_head[0].weight
    weight_diff = torch.norm(updated_weight - initial_weight).item()
    print(f"L2 norm of weight update in projection head: {weight_diff:.6e}")
    assert weight_diff > 0, "Parameters did not change after optimizer step!"
    print("Parameter update confirmed: True")

    print("\n" + "=" * 75)
    print("ALL 10 SANITY CHECKS PASSED PERFECTLY!")
    print("=" * 75)
    return {
        "status": "PASSED",
        "dataset_path": TRAIN_DIR,
        "num_training_images": len(train_samples),
        "num_design_ids": len(train_design_ids),
        "embedding_dim": EMBEDDING_DIM,
        "total_parameters": summary['total_parameters'],
        "loss_value": loss_val,
        "weight_update_norm": weight_diff
    }


if __name__ == "__main__":
    res = run_sanity_checks()
