import os
import re
from collections import defaultdict, Counter

data_dir = r"c:\Users\sujal\OneDrive\Desktop\DeepLure-Saree-Recognition\data"

# Handloom sarees
hl_dir = os.path.join(data_dir, "handloom_sarees")
if os.path.exists(hl_dir):
    hl_files = os.listdir(hl_dir)
    print(f"Handloom sarees count: {len(hl_files)}")
    print(f"Sample handloom filenames: {hl_files[:10]}")

# Fabric patterns
fabric_files = []
for split in ["train", "valid", "test"]:
    for cat in ["Banarasi", "Bandhani", "Ikat", "Pichwai"]:
        cdir = os.path.join(data_dir, split, cat)
        if os.path.exists(cdir):
            for f in os.listdir(cdir):
                fabric_files.append((split, cat, f))

print(f"Total fabric pattern files: {len(fabric_files)}")

# Groupings
base_names = defaultdict(list)
for split, cat, f in fabric_files:
    clean = re.sub(r'(_jpg|_jpeg|_png)?\.rf\.[a-f0-9]+\.(jpg|jpeg|png)$', '', f, flags=re.IGNORECASE)
    base_names[(cat, clean)].append((split, f))

multi_occurrences = {k: v for k, v in base_names.items() if len(v) > 1}
print(f"Unique (category, base_name) pairs: {len(base_names)}")
print(f"Base names with multiple occurrences: {len(multi_occurrences)}")

# Check prefix / SKU patterns
sku_groups = defaultdict(list)
for cat, clean in base_names.keys():
    # check if there is an SKU prefix like DN-1007 or 94235 or SAN or R-TSBR
    prefix_match = re.match(r'^([A-Za-z0-9_-]+?)(?:[-_][A-Za-z0-9]+|\.[a-z]+)?$', clean)
    if prefix_match:
        sku_groups[(cat, prefix_match.group(1))].append(clean)

print("\nSample base names:")
for (cat, bname), occurrences in list(base_names.items())[:15]:
    print(f"  [{cat}] '{bname}' -> {len(occurrences)} files")

if multi_occurrences:
    print("\nSample multi occurrences:")
    for (cat, bname), occ in list(multi_occurrences.items())[:5]:
        print(f"  [{cat}] '{bname}': {occ}")
