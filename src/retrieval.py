"""
Gallery-Query Retrieval and Identification Module.
Implements:
- Gallery index computation
- Top-K retrieval based on Cosine Similarity (retrieve_top_k)
- Identification metrics: Top-1 Accuracy, Top-5 Accuracy, Mean Reciprocal Rank (MRR)
"""

from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.dataset import get_eval_transforms
from src.config import TOP_K, get_device


def compute_single_embedding(
    model: torch.nn.Module,
    image_input: Union[str, Image.Image, torch.Tensor],
    device: torch.device = None,
    transform = None
) -> torch.Tensor:
    """
    Computes L2-normalized embedding (dim=256) for a single image input.
    """
    if device is None:
        device = get_device()
    if transform is None:
        transform = get_eval_transforms()

    model.eval()
    with torch.no_grad():
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
        elif isinstance(image_input, Image.Image):
            tensor = transform(image_input).unsqueeze(0).to(device)
        elif isinstance(image_input, torch.Tensor):
            if image_input.dim() == 3:
                tensor = image_input.unsqueeze(0).to(device)
            else:
                tensor = image_input.to(device)
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        emb = model.extract_embedding(tensor)
    return emb.cpu()


def build_gallery_index(
    model: torch.nn.Module,
    gallery_loader: torch.utils.data.DataLoader,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Computes and caches L2-normalized embeddings for the entire reference gallery.
    """
    if device is None:
        device = get_device()

    model.eval()
    embeddings = []
    paths = []
    design_ids = []
    categories = []
    labels = []

    with torch.no_grad():
        for batch in gallery_loader:
            imgs, batch_labels, batch_cats, batch_dids, batch_paths = batch
            imgs = imgs.to(device)
            emb = model.extract_embedding(imgs)
            embeddings.append(emb.cpu())
            paths.extend(batch_paths)
            design_ids.extend(batch_dids)
            categories.extend(batch_cats)
            labels.extend(batch_labels.tolist() if isinstance(batch_labels, torch.Tensor) else batch_labels)

    gallery_embeddings = torch.cat(embeddings, dim=0)  # Shape: (N_gallery, 256)
    
    return {
        "embeddings": gallery_embeddings,
        "paths": paths,
        "design_ids": design_ids,
        "categories": categories,
        "labels": labels
    }


def retrieve_top_k(
    query_image: Union[str, Image.Image, torch.Tensor],
    gallery: Dict[str, Any],
    k: int = TOP_K,
    model: torch.nn.Module = None,
    device: torch.device = None,
    transform = None,
    query_design_id: str = None
) -> List[Dict[str, Any]]:
    """
    Given a query saree image, retrieves the top-K most similar saree designs from the reference gallery.
    
    Args:
        query_image: Filepath, PIL Image, or precomputed embedding tensor
        gallery: Precomputed gallery index dictionary
        k: Number of top matches to retrieve
        model: ColorInvariantSareeEncoder model
        device: Compute device
        transform: Image transform
        query_design_id: Optional ground-truth design ID of the query
    
    Returns:
        List of dicts: [
            {
                'rank': int,
                'gallery_path': str,
                'similarity_score': float,
                'design_id': str,
                'category': str,
                'is_match': bool
            }, ...
        ]
    """
    if device is None:
        device = get_device()

    # 1. Compute query embedding
    if isinstance(query_image, torch.Tensor) and query_image.dim() == 2 and query_image.shape[0] == 1:
        query_emb = query_image.cpu()
    else:
        if model is None:
            raise ValueError("Model is required to extract query embedding.")
        query_emb = compute_single_embedding(model, query_image, device=device, transform=transform)

    # 2. Extract gallery embeddings and metadata
    gallery_embs = gallery["embeddings"]
    gallery_paths = gallery["paths"]
    gallery_dids = gallery["design_ids"]
    gallery_cats = gallery["categories"]

    # 3. Calculate Cosine Similarity: query_emb (1, 256) @ gallery_embs.T (256, N)
    sims = torch.matmul(query_emb, gallery_embs.T).squeeze(0)  # Shape: (N,)

    # 4. Sort in descending order
    topk_sims, topk_indices = torch.topk(sims, min(k, len(gallery_paths)), largest=True)

    results = []
    for rank, (sim, idx) in enumerate(zip(topk_sims.tolist(), topk_indices.tolist())):
        g_did = gallery_dids[idx]
        is_match = (query_design_id is not None and g_did == query_design_id)
        results.append({
            "rank": rank + 1,
            "gallery_path": gallery_paths[idx],
            "similarity_score": float(sim),
            "design_id": g_did,
            "category": gallery_cats[idx],
            "is_match": is_match
        })

    return results


def evaluate_retrieval_metrics(
    model: torch.nn.Module,
    query_loader: torch.utils.data.DataLoader,
    gallery_index: Dict[str, Any],
    device: torch.device = None
) -> Dict[str, float]:
    """
    Evaluates Top-1, Top-5, and MRR metrics over a query set against the reference gallery.
    """
    if device is None:
        device = get_device()

    model.eval()
    query_embs = []
    query_dids = []
    
    with torch.no_grad():
        for batch in query_loader:
            imgs, _, _, batch_dids, _ = batch
            imgs = imgs.to(device)
            emb = model.extract_embedding(imgs)
            query_embs.append(emb.cpu())
            query_dids.extend(batch_dids)

    query_embeddings = torch.cat(query_embs, dim=0)
    gallery_embeddings = gallery_index["embeddings"]
    gallery_dids = gallery_index["design_ids"]

    # Cosine similarity matrix: (N_query, N_gallery)
    sim_matrix = torch.matmul(query_embeddings, gallery_embeddings.T)

    n_queries = len(query_dids)
    top1_correct = 0
    top5_correct = 0
    reciprocal_ranks = []

    for i in range(n_queries):
        q_did = query_dids[i]
        sims = sim_matrix[i]
        sorted_indices = torch.argsort(sims, descending=True).tolist()
        
        match_ranks = [r + 1 for r, idx in enumerate(sorted_indices) if gallery_dids[idx] == q_did]
        if match_ranks:
            first_rank = match_ranks[0]
            if first_rank == 1:
                top1_correct += 1
            if first_rank <= 5:
                top5_correct += 1
            reciprocal_ranks.append(1.0 / first_rank)
        else:
            reciprocal_ranks.append(0.0)

    return {
        "num_queries": n_queries,
        "top1_accuracy": top1_correct / max(1, n_queries),
        "top5_accuracy": top5_correct / max(1, n_queries),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    }
