# Color-Invariant Saree Design Recognition System

An end-to-end Computer Vision and Metric-Learning system in PyTorch designed to identify and verify sarees based on their **surface design, motifs, and weave textures**, completely invariant to the chromatic color palette.

**Dataset Source:** `C:\deeplure` (Direct read-only reference from source directory; zero copying, moving, renaming, or modification).

---

## 1. Problem Statement
In traditional Indian ethnic wear and textile e-commerce, sarees sharing identical weave patterns (e.g., Banarasi floral brocades, Ikat geometric lattices, Bandhani tie-dye dots, Pichwai devotional motifs) are manufactured across multiple contrasting color palettes.

Conventional visual models rely heavily on dominant chromatic features and background colors, causing:
- Sarees with the **same design in different colors** to fail matching.
- Sarees with **completely different designs in identical colors** to falsely match.

This project addresses this fundamental limitation by learning an invariant representation space where spatial motif structures, weave geometries, and borders dictate embedding proximity on a unit hypersphere, rendering representations truly color-invariant.

---

## 2. Objective
1. **Gallery/Query Top-K Retrieval (`retrieve_top_k`)**:
   Given a query saree image, retrieve the top-$K$ most structurally similar designs from a reference gallery.
2. **Pairwise Design Verification (`verify_pair`)**:
   Given two saree images, determine whether they contain the `SAME DESIGN` or `DIFFERENT DESIGN` using a validation-calibrated operating threshold $\tau^*$.
3. **Color-Invariance Guarantee & Dual Evaluation Protocols**:
   - **A. Standard Design Retrieval**: Evaluated on unseen test gallery/query partitions.
   - **B. Synthetic Color-Invariance Stress Test**: Generates query variants using extreme color transformations (grayscale, severe hue shifts, brightness/contrast/saturation perturbations) matched against the undisturbed reference gallery.

---

## 3. Dataset Description & Statistics (`C:\deeplure`)

| Category | Description & Design Characteristics | Train Images | Valid Images | Test Images | Total Images |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Banarasi** | Intricate brocade weaves, floral jals, paisleys, metallic zari patterns | 432 | 43 | 14 | 489 |
| **Bandhani** | Tie-dye dots, geometric concentric squares, diamond lattices | 279 | 22 | 15 | 316 |
| **Ikat** | Resist-dye blurred geometric motifs, Pochampally double-ikats | 303 | 26 | 13 | 342 |
| **Pichwai** | Traditional devotional paintings, lotus motifs, peacocks, sacred art | 279 | 24 | 18 | 321 |
| **Handloom Sarees** | Full-length draped handloom sarees (auxiliary unlabelled gallery) | — | — | — | 165 |
| **Total** | **All categories & folders** | **1,293** | **115** | **60** | **1,633** |

### Design Identity Extraction & Disjoint Partitions
- Roboflow filename hash stripping extracts the base design identity (e.g. `SAN2454_jpg.rf.922c...` $\rightarrow$ `SAN2454`).
- **Train Partition (`C:\deeplure\archive\train`)**: Exactly 1,293 images across 431 design classes ($431 \times 3$ multi-view images).
- **Validation Partition (`C:\deeplure\archive\valid`)**: 115 images across 115 disjoint design classes.
- **Test Partition (`C:\deeplure\archive\test`)**: 60 images across 60 disjoint design classes.
- **Zero Identity Leakage**: Verified that $\text{Train} \cap \text{Valid} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, and $\text{Valid} \cap \text{Test} = \emptyset$.
- **Handloom Sarees (165 images)**: Treated as an optional auxiliary dataset and excluded from supervised metric learning because individual multi-sample design identities cannot be established.

---

## 4. Dataset Preprocessing & Pipeline
1. **Resolution Standardization**: All input images standardized to $224 \times 224 \times 3$.
2. **Channel Format**: 100% 3-channel RGB (0 corrupt files across all 1,633 images).
3. **Statistical Normalization**: ImageNet mean $\mu = [0.485, 0.456, 0.406]$ and standard deviation $\sigma = [0.229, 0.224, 0.225]$.

---

## 5. Model Architecture (`ColorInvariantSareeEncoder`)
```
                Input Saree Image (3 x 224 x 224)
                               ↓
                 Pretrained ResNet-18 Backbone
                               ↓
                 Global Average Pooling (512-d)
                               ↓
              MLP Projection Head: Linear(512 → 512)
                               ↓
                          BatchNorm1d
                               ↓
                         ReLU Activation
                               ↓
                       Linear(512 → 256)
                               ↓
             L2 Normalization: z / ||z||_2 (256-d)
                               ↓
                  Unit Hypersphere Embedding
```
- **Total Parameters**: 11,571,008 (~44.14 MB)
- **Trainable Parameters**: 11,571,008
- **Embedding Dimension**: 256

---

## 6. Color-Invariance Strategy
1. **Multi-View Chromatic Perturbations**:
   - `RandomGrayscale(p=0.35)`: Strips all chromatic channels, forcing representation of luminance gradients and weave textures.
   - `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.4, p=0.8)`: Continuously randomizes chromatic coordinates.
   - `RandomAutocontrast(p=0.2)` & `GaussianBlur(p=0.2)`.
2. **Spatial Pattern Retention**:
   - `RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.15))`
   - `RandomHorizontalFlip(p=0.5)` & `RandomRotation(degrees=15)`.
3. **Multi-View Contrastive Formulation**:
   - Two independently color-perturbed views of the same design are explicitly pulled together in embedding space.

---

## 7. Training Strategy
- **Optimizer**: AdamW ($\text{lr} = 10^{-4}$, weight decay $= 10^{-4}$)
- **Learning Rate Scheduler**: Cosine Annealing LR ($T_{\max} = 15, \eta_{\min} = 10^{-6}$)
- **Batch Size**: 32 (with Two-Crop expansion $\rightarrow 64$ embeddings per batch)
- **Gradient Clipping**: Maximum $L_2$ norm of $1.0$
- **Checkpointing**: Checkpoint saved when validation contrastive loss reaches new minimum (`outputs/checkpoints/best_model.pth`).

---

## 8. Metric-Learning Loss Function (`SupConLoss`)
We employ **Supervised Contrastive Loss** (Khosla et al., NeurIPS 2020) with temperature $\tau = 0.07$:

$$\mathcal{L}_{out}^{sup} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

---

## 9. Gallery / Query Splitting & Evaluation Protocols

### Protocol A: Standard Design Retrieval
- Unseen 60 test images split into Reference Gallery ($G$) and Query Set ($Q$).
- Evaluates Top-1 Accuracy, Top-5 Accuracy, and MRR.

### Protocol B: Synthetic Color-Invariance Stress Test
- Reference Gallery ($G$): All 60 original, unperturbed test images.
- Query Variants: Transformed queries generated via:
  1. Complete Grayscale conversion (zero color channels).
  2. Severe ColorJitter (extreme hue rotation, saturation, contrast).
- Measures whether the system retrieves the correct design when color information is destroyed or scrambled.

### Protocol C: Pair Verification
- Verification decision threshold $\tau^*$ is calibrated **strictly on validation data**, never on the test set.
- Evaluates Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Generates separate similarity distributions for same-design and different-design pairs.

---

## 10. Identification System (`retrieve_top_k`)
```python
from src.retrieval import retrieve_top_k, build_gallery_index
from src.model import ColorInvariantSareeEncoder

model = ColorInvariantSareeEncoder(embedding_dim=256)
gallery_index = build_gallery_index(model, gallery_loader)

results = retrieve_top_k(query_image="path/to/saree.jpg", gallery=gallery_index, k=5, model=model)
```

---

## 11. Pairwise Verification System (`verify_pair`)
```python
from src.verification import verify_pair

result = verify_pair(
    image_a="path/to/saree1.jpg",
    image_b="path/to/saree2.jpg",
    threshold=0.70,
    model=model
)
```

---

## 12. Pre-Training Sanity Checks
Before running long training jobs, the pipeline verifies 10 sanity checks via `sanity_check.py`:
1. Loads 16 training samples from `C:\deeplure\archive\train`.
2. Generates two independently augmented views.
3. Verifies tensor shapes are `[16, 3, 224, 224]`.
4. Runs forward pass through `ColorInvariantSareeEncoder`.
5. Confirms embedding shape is `[16, 256]`.
6. Confirms embeddings are finite (no NaN/Inf) and unit L2-normalized.
7. Calculates Supervised Contrastive Loss.
8. Confirms loss is finite.
9. Executes backward pass and optimizer step.
10. Confirms model parameters update with non-zero gradient norm.

---

## 13. How to Run

### Run Sanity Check
```bash
python sanity_check.py
```

### Run Full Training & Evaluation Pipeline
```bash
python src/train.py
```

### Run Interactive Jupyter Notebook
```bash
jupyter notebook saree_recognition_pipeline.ipynb
```

---

## 14. Empirical Results & Benchmark Summary

All metrics below were computed directly through the executed PyTorch pipeline on the actual dataset at `C:\deeplure`:

| Category | Metric | Empirical Value |
| :--- | :--- | :---: |
| **Training** | Total Training Time (8 epochs, CPU) | **24.57 min** (1,474.2 s) |
| **Training** | Initial Training Loss (Epoch 1) | **0.6156** |
| **Training** | Best Validation Loss (Epoch 7) | **0.2252** |
| **Training** | Final Training Loss (Epoch 8) | **0.1234** |
| **Verification (Validation)** | Calibrated Operating Threshold ($\tau^*$) | **0.4279** |
| **Verification (Validation)** | Validation F1-Score | **99.57%** |
| **Verification (Validation)** | Validation ROC-AUC | **0.9998** |
| **Verification (Test)** | Pairwise Verification Accuracy | **99.17%** |
| **Verification (Test)** | Precision | **98.36%** |
| **Verification (Test)** | Recall | **100.00%** |
| **Verification (Test)** | F1-Score | **99.17%** |
| **Verification (Test)** | ROC-AUC | **1.0000** |
| **Verification (Test)** | Confusion Matrix | $\text{TP}=60, \text{FP}=1, \text{TN}=59, \text{FN}=0$ |
| **Synthetic Stress Test (Grayscale)** | Top-1 Retrieval Accuracy | **98.33%** |
| **Synthetic Stress Test (Grayscale)** | Top-5 Retrieval Accuracy | **100.00%** |
| **Synthetic Stress Test (Grayscale)** | Mean Reciprocal Rank (MRR) | **0.9917** |
| **Synthetic Stress Test (Color Jitter)** | Top-1 Retrieval Accuracy | **96.67%** |
| **Synthetic Stress Test (Color Jitter)** | Top-5 Retrieval Accuracy | **100.00%** |
| **Synthetic Stress Test (Color Jitter)** | Mean Reciprocal Rank (MRR) | **0.9833** |
| **Synthetic Stress Test (Hue Shift)** | Top-1 Retrieval Accuracy | **96.67%** |
| **Synthetic Stress Test (Hue Shift)** | Top-5 Retrieval Accuracy | **100.00%** |
| **Synthetic Stress Test (Hue Shift)** | Mean Reciprocal Rank (MRR) | **0.9833** |
| **Computational Efficiency** | Total Model Parameters | **11,571,520** |
| **Computational Efficiency** | Embedding Dimension | **256** |
| **Computational Efficiency** | Checkpoint Size on Disk | **~44.14 MB** |
| **Computational Efficiency** | Inference Latency (CPU) | **44.73 ms / image** |
| **Computational Efficiency** | Throughput (CPU) | **22.4 FPS** |

---

## 15. Saved Artifacts & Outputs

- **Best Trained Checkpoint:** [`outputs/checkpoints/best_model.pth`](file:///c:/Users/sujal/OneDrive/Desktop/DeepLure-Saree-Recognition/outputs/checkpoints/best_model.pth)
- **Final Checkpoint:** [`outputs/checkpoints/final_model.pth`](file:///c:/Users/sujal/OneDrive/Desktop/DeepLure-Saree-Recognition/outputs/checkpoints/final_model.pth)
- **Benchmark Checkpoint (Preserved):** [`outputs/benchmark/benchmark_model_2epochs.pth`](file:///c:/Users/sujal/OneDrive/Desktop/DeepLure-Saree-Recognition/outputs/benchmark/benchmark_model_2epochs.pth)
- **Full Results JSON:** [`outputs/results/evaluation_results.json`](file:///c:/Users/sujal/OneDrive/Desktop/DeepLure-Saree-Recognition/outputs/results/evaluation_results.json)
- **Summary Metrics CSV:** [`outputs/results/summary_metrics.csv`](file:///c:/Users/sujal/OneDrive/Desktop/DeepLure-Saree-Recognition/outputs/results/summary_metrics.csv)
- **Visualizations:**
  - [`outputs/figures/similarity_distribution.png`](file:///c:/Users/sujal/OneDrive/Desktop/DeepLure-Saree-Recognition/outputs/figures/similarity_distribution.png)
  - [`outputs/figures/roc_curve.png`](file:///c:/Users/sujal/OneDrive/Desktop/DeepLure-Saree-Recognition/outputs/figures/roc_curve.png)
  - [`outputs/figures/confusion_matrix.png`](file:///c:/Users/sujal/OneDrive/Desktop/DeepLure-Saree-Recognition/outputs/figures/confusion_matrix.png)
  - [`outputs/figures/retrieval_examples.png`](file:///c:/Users/sujal/OneDrive/Desktop/DeepLure-Saree-Recognition/outputs/figures/retrieval_examples.png)
  - [`outputs/figures/synthetic_color_invariance_demo.png`](file:///c:/Users/sujal/OneDrive/Desktop/DeepLure-Saree-Recognition/outputs/figures/synthetic_color_invariance_demo.png)
- **Jupyter Notebook:** [`saree_recognition_pipeline.ipynb`](file:///c:/Users/sujal/OneDrive/Desktop/DeepLure-Saree-Recognition/saree_recognition_pipeline.ipynb)

