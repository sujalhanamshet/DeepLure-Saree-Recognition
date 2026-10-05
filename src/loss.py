"""
Metric Learning Loss Functions for Color-Invariant Saree Recognition.
Implements:
1. Supervised Contrastive Loss (SupConLoss) - Khosla et al., NeurIPS 2020
2. Triplet Margin Loss with Online Hard Negative Mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    
    Why SupCon is ideal for Color-Invariant Saree Recognition:
    1. Multi-positive pulling: In contrast to standard SimCLR (which only pairs identical
       crops), SupCon pulls together ALL views and color-augmented instances belonging to
       the same base design identity.
    2. Global negative push: It simultaneously pushes away all instances belonging to
       different designs, even if they share similar dominant colors or background palettes.
    3. Unit hypersphere geometry: By operating on L2-normalized embeddings, cosine similarity
       directly governs metric separation with temperature scaling.
    
    Reference:
      Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    """
    def __init__(self, temperature: float = 0.07, contrast_mode: str = "all", base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: Hidden vectors of shape (batch_size, n_views, embedding_dim)
                      or (batch_size, embedding_dim).
            labels: Ground truth design IDs of shape (batch_size,).
            mask: Contrastive mask of shape (batch_size, batch_size), mask_{i,j}=1 if same design.
        Returns:
            Scalar loss value.
        """
        device = features.device

        # If features shape is (B, D), add view dimension -> (B, 1, D)
        if len(features.shape) < 3:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        n_views = features.shape[1]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask` simultaneously.")
        elif labels is None and mask is None:
            # Unsupervised SimCLR mode: identical image views are positives
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Number of labels does not match batch size.")
            # Binary mask where mask[i, j] = 1 if labels[i] == labels[j]
            mask = torch.eq(labels, labels.T).float().to(device)

        # Reshape features to (batch_size * n_views, embedding_dim)
        contrast_count = n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast mode: {self.contrast_mode}")

        # Compute cosine similarity matrix: (anchor_count*B) x (contrast_count*B)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # For numerical stability: subtract max per row
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask across views
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Mask-out self-contrast cases (diagonal where sample is compared with itself)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        # Sum over positive pairs in each row, divided by count of positive pairs
        mask_pos_pairs = mask.sum(1)
        # Avoid division by zero when a sample has no other positive in the batch
        mask_pos_pairs = torch.where(mask_pos_pairs == 0, torch.ones_like(mask_pos_pairs), mask_pos_pairs)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Loss formulation
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class TripletHardNegativeLoss(nn.Module):
    """
    Triplet Loss with online batch hard-negative mining based on Cosine Distance.
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalized embeddings of shape (B, D)
            labels: 1D tensor of class labels (B,)
        """
        # Cosine distance = 1 - cosine_similarity = 1 - (u . v)
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        dist_matrix = 1.0 - sim_matrix
        
        labels_eq = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
        
        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        valid_triplets = 0
        
        batch_size = embeddings.shape[0]
        for i in range(batch_size):
            # Positives: same label, not self
            pos_indices = torch.where(labels_eq[i] & (torch.arange(batch_size, device=embeddings.device) != i))[0]
            # Negatives: different label
            neg_indices = torch.where(~labels_eq[i])[0]
            
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                # Hardest positive: maximum distance among positives
                hardest_pos_dist = dist_matrix[i, pos_indices].max()
                # Hardest negative: minimum distance among negatives
                hardest_neg_dist = dist_matrix[i, neg_indices].min()
                
                triplet_loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
                if triplet_loss > 0:
                    loss = loss + triplet_loss
                    valid_triplets += 1
                    
        if valid_triplets > 0:
            loss = loss / valid_triplets
            
        return loss
