"""
Pairwise Saree Design Verification Module.
Implements:
- Pair verification function: verify_pair(image_a, image_b, threshold)
- Validation-only threshold calibration (preventing test set leakage)
- Verification evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
- Similarity distribution generation for same-design vs different-design pairs
"""

from typing import Union, Dict, Any, List, Tuple
from PIL import Image
import numpy as np
import torch

from src.dataset import get_eval_transforms
from src.retrieval import compute_single_embedding
from src.config import DEFAULT_VERIFICATION_THRESHOLD, get_device
from src.metrics import compute_accuracy, compute_precision_recall_f1, compute_roc_curve, compute_auc


def verify_pair(
    image_a: Union[str, Image.Image, torch.Tensor],
    image_b: Union[str, Image.Image, torch.Tensor],
    threshold: float = DEFAULT_VERIFICATION_THRESHOLD,
    model: torch.nn.Module = None,
    device: torch.device = None,
    transform = None
) -> Dict[str, Any]:
    """
    Given two saree images, determines whether they contain the same underlying design.
    
    Args:
        image_a: First saree image (path, PIL Image, or tensor)
        image_b: Second saree image (path, PIL Image, or tensor)
        threshold: Cosine similarity decision threshold (tau)
        model: ColorInvariantSareeEncoder model instance
        device: Compute device
        transform: Evaluation transform
        
    Returns:
        Dict containing:
          - 'prediction': 'SAME DESIGN' or 'DIFFERENT DESIGN'
          - 'similarity_score': float cosine similarity [-1.0, 1.0]
          - 'is_same_design': bool (True if similarity >= threshold)
          - 'threshold': float
    """
    if device is None:
        device = get_device()
    if transform is None:
        transform = get_eval_transforms()

    if model is None:
        raise ValueError("A trained ColorInvariantSareeEncoder model instance is required.")

    emb_a = compute_single_embedding(model, image_a, device=device, transform=transform)
    emb_b = compute_single_embedding(model, image_b, device=device, transform=transform)

    # Cosine similarity between unit L2-normalized vectors is dot product
    similarity = float(torch.sum(emb_a * emb_b).item())
    is_same = (similarity >= threshold)
    prediction = "SAME DESIGN" if is_same else "DIFFERENT DESIGN"

    return {
        "prediction": prediction,
        "similarity_score": round(similarity, 4),
        "is_same_design": is_same,
        "threshold": threshold
    }


def calibrate_verification_threshold(
    val_pairs: List[Dict[str, Any]],
    model: torch.nn.Module,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Calibrates the optimal verification threshold using VALIDATION data ONLY.
    Guarantees zero leakage into the test set.
    """
    if device is None:
        device = get_device()

    model.eval()
    y_true = []
    similarities = []

    transform = get_eval_transforms()
    with torch.no_grad():
        for pair in val_pairs:
            emb_a = compute_single_embedding(model, pair['img1_path'], device=device, transform=transform)
            emb_b = compute_single_embedding(model, pair['img2_path'], device=device, transform=transform)
            sim = float(torch.sum(emb_a * emb_b).item())
            similarities.append(sim)
            y_true.append(pair['label'])

    y_true = np.array(y_true)
    similarities = np.array(similarities)

    # Candidate threshold scan
    candidate_thresholds = np.linspace(-0.2, 0.99, 200)
    best_thresh = 0.5
    best_f1 = -1.0
    best_acc = 0.0

    for t in candidate_thresholds:
        preds = (similarities >= t).astype(int)
        _, _, f1, _ = compute_precision_recall_f1(y_true, preds)
        acc = compute_accuracy(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_acc = acc

    fpr, tpr, _ = compute_roc_curve(y_true, similarities)
    roc_auc = compute_auc(fpr, tpr)

    return {
        "optimal_threshold": float(best_thresh),
        "best_val_f1": float(best_f1),
        "best_val_acc": float(best_acc),
        "val_roc_auc": float(roc_auc),
        "y_true": y_true,
        "similarities": similarities,
        "same_design_similarities": similarities[y_true == 1].tolist(),
        "diff_design_similarities": similarities[y_true == 0].tolist(),
        "fpr": fpr,
        "tpr": tpr
    }


def evaluate_verification_on_test(
    test_pairs: List[Dict[str, Any]],
    threshold: float,
    model: torch.nn.Module,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Evaluates verification performance on unseen test pairs using the validation-calibrated threshold.
    """
    if device is None:
        device = get_device()

    model.eval()
    y_true = []
    similarities = []

    transform = get_eval_transforms()
    with torch.no_grad():
        for pair in test_pairs:
            emb_a = compute_single_embedding(model, pair['img1_path'], device=device, transform=transform)
            emb_b = compute_single_embedding(model, pair['img2_path'], device=device, transform=transform)
            sim = float(torch.sum(emb_a * emb_b).item())
            similarities.append(sim)
            y_true.append(pair['label'])

    y_true = np.array(y_true)
    similarities = np.array(similarities)
    preds = (similarities >= threshold).astype(int)

    precision, recall, f1, _ = compute_precision_recall_f1(y_true, preds)
    acc = compute_accuracy(y_true, preds)
    fpr, tpr, _ = compute_roc_curve(y_true, similarities)
    roc_auc = compute_auc(fpr, tpr)

    # Confusion matrix breakdown
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))

    return {
        "num_pairs": len(test_pairs),
        "threshold_applied": float(threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "same_design_similarities": similarities[y_true == 1].tolist(),
        "diff_design_similarities": similarities[y_true == 0].tolist(),
        "y_true": y_true,
        "similarities": similarities,
        "fpr": fpr,
        "tpr": tpr
    }
