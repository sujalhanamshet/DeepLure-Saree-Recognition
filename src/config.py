"""
Configuration settings for Color-Invariant Saree Recognition.
Points to the actual dataset at C:\\deeplure (read-only) and configures
the training, evaluation, retrieval, and verification pipelines.
"""

import os
import random
import numpy as np

# Base Project Directory (in Workspace)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Actual Dataset Paths (READ-ONLY directly from C:\deeplure)
DATASET_ROOT = r"C:\deeplure"
ARCHIVE_DIR = os.path.join(DATASET_ROOT, "archive")
TRAIN_DIR = os.path.join(ARCHIVE_DIR, "train")
VALID_DIR = os.path.join(ARCHIVE_DIR, "valid")
TEST_DIR = os.path.join(ARCHIVE_DIR, "test")
HANDLOOM_DIR = os.path.join(DATASET_ROOT, "handloom_sarees-20261005T164300Z-1-001", "handloom_sarees")

# Project Output Directories
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
BENCHMARK_DIR = os.path.join(OUTPUT_DIR, "benchmark")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

for directory in [OUTPUT_DIR, CHECKPOINT_DIR, BENCHMARK_DIR, FIGURES_DIR, METRICS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Image & Normalization Specifications
IMAGE_SIZE = 224  # Standard resolution for ResNet-18 backbone
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Model Specifications
BACKBONE_NAME = "resnet18"
EMBEDDING_DIM = 256
PROJECTION_HIDDEN_DIM = 512
PRETRAINED = True

# Metric-Learning & Optimization Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 8  # Exactly 8 epochs for full training
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.07  # Temperature scaling factor for Supervised Contrastive Loss
SEED = 42
NUM_WORKERS = 0  # 0 for Windows compatibility

# Retrieval & Verification Thresholds
TOP_K = 5
DEFAULT_VERIFICATION_THRESHOLD = 0.70

def seed_everything(seed=SEED):
    """Sets random seeds across all libraries for deterministic reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def get_device():
    """Detects and returns compute device (CUDA GPU if available, else CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        pass
    return "cpu"
