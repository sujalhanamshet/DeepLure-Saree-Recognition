import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    TRAIN_DIR, VALID_DIR, TEST_DIR, HANDLOOM_DIR
)
from src.dataset import (
    scan_dataset_partition, scan_auxiliary_handloom_dataset,
    extract_base_design_id
)

print("=" * 70)
print("TESTING DATASET PARSING ON C:\\deeplure")
print("=" * 70)

train_samples = scan_dataset_partition(TRAIN_DIR)
valid_samples = scan_dataset_partition(VALID_DIR)
test_samples = scan_dataset_partition(TEST_DIR)
handloom_samples = scan_auxiliary_handloom_dataset(HANDLOOM_DIR)

print(f"Train samples:     {len(train_samples)} (Expected: 1293)")
print(f"Valid samples:     {len(valid_samples)} (Expected: 115)")
print(f"Test samples:      {len(test_samples)} (Expected: 60)")
print(f"Handloom samples:  {len(handloom_samples)} (Expected: 165)")

train_dids = set(s[2] for s in train_samples)
valid_dids = set(s[2] for s in valid_samples)
test_dids = set(s[2] for s in test_samples)

print(f"Train unique designs: {len(train_dids)} (Expected: 431)")
print(f"Valid unique designs: {len(valid_dids)} (Expected: 115)")
print(f"Test unique designs:  {len(test_dids)} (Expected: 60)")

assert len(train_samples) == 1293
assert len(valid_samples) == 115
assert len(test_samples) == 60
assert len(handloom_samples) == 165
assert len(train_dids) == 431
assert len(valid_dids) == 115
assert len(test_dids) == 60

# Check zero cross-split leakage
assert len(train_dids.intersection(valid_dids)) == 0
assert len(train_dids.intersection(test_dids)) == 0
assert len(valid_dids.intersection(test_dids)) == 0

print("Zero cross-split data leakage verified: PASS")
print("All assertions passed successfully!")
