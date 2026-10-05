import os
import re
import hashlib
from collections import defaultdict, Counter
from PIL import Image

ROOT_DIR = r"C:\deeplure"

print("=" * 80)
print(f"DEEP INSPECTION OF ACTUAL DATASET: {ROOT_DIR}")
print("=" * 80)

# 1. Directory Tree & Folder Structure
print("\n--- 1. DIRECTORY TREE & FOLDER STRUCTURE ---")
all_dirs = []
all_files = []
for root, dirs, files in os.walk(ROOT_DIR):
    rel_root = os.path.relpath(root, ROOT_DIR)
    all_dirs.append(rel_root)
    for f in files:
        all_files.append(os.path.join(root, f))

print(f"Total directories found: {len(all_dirs)}")
for d in sorted(all_dirs):
    print(f"  {d}")

# 2. Files & Extensions breakdown
print("\n--- 2. FILE FORMATS & EXTENSIONS BREAKDOWN ---")
ext_counts = Counter([os.path.splitext(f)[1].lower() for f in all_files])
print(f"Total files: {len(all_files)}")
print(f"File extensions: {dict(ext_counts)}")

# Check non-image files (e.g. metadata, readmes, csv, json)
non_image_files = [f for f in all_files if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'))]
print(f"\nNon-image/metadata files found ({len(non_image_files)}):")
for nf in non_image_files:
    rel_path = os.path.relpath(nf, ROOT_DIR)
    sz = os.path.getsize(nf)
    print(f"  {rel_path} ({sz} bytes)")
    # Print content if small text file
    if sz < 2000 and nf.endswith(('.txt', '.json', '.csv', '.yaml', '.yml')):
        try:
            with open(nf, 'r', encoding='utf-8', errors='ignore') as f_in:
                print(f"    Content preview:\n    " + f_in.read().strip().replace("\n", "\n    "))
        except Exception as e:
            print(f"    Error reading file: {e}")

# 3. Image Analysis & Integrity Check
print("\n--- 3. IMAGE RESOLUTIONS, CHANNELS, & CORRUPTION CHECK ---")
image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'))]
print(f"Total image files: {len(image_files)}")

corrupted_images = []
color_modes = Counter()
dimensions = Counter()
folder_image_counts = Counter()
md5_to_files = defaultdict(list)

for img_path in image_files:
    rel_dir = os.path.relpath(os.path.dirname(img_path), ROOT_DIR)
    folder_image_counts[rel_dir] += 1
    
    # Check MD5 for duplicate files
    try:
        with open(img_path, 'rb') as f_in:
            md5 = hashlib.md5(f_in.read()).hexdigest()
            md5_to_files[md5].append(img_path)
    except Exception as e:
        corrupted_images.append((img_path, f"File read error: {e}"))
        continue

    # PIL verification
    try:
        with Image.open(img_path) as img:
            color_modes[img.mode] += 1
            dimensions[img.size] += 1
    except Exception as e:
        corrupted_images.append((img_path, f"PIL open error: {e}"))

print(f"\nCorrupted Images: {len(corrupted_images)}")
for cp, err in corrupted_images[:10]:
    print(f"  {cp}: {err}")

print(f"\nColor Modes: {dict(color_modes)}")
print(f"Distinct Resolutions Count: {len(dimensions)}")
print("Top 10 Image Dimensions (Width x Height):")
for dim, cnt in dimensions.most_common(10):
    print(f"  {dim}: {cnt} images")

# 4. Folder Image Counts & Train/Val/Test Breakdown
print("\n--- 4. IMAGES PER FOLDER BREAKDOWN ---")
for fldr, count in sorted(folder_image_counts.items()):
    print(f"  {fldr}: {count} images")

# 5. Exact Duplicate Images Check
duplicates = {md5: paths for md5, paths in md5_to_files.items() if len(paths) > 1}
print(f"\n--- 5. EXACT DUPLICATE IMAGES (BY MD5) ---")
print(f"Unique image hashes: {len(md5_to_files)}")
print(f"Duplicate image hash groups: {len(duplicates)}")
if duplicates:
    print("Sample duplicate groups:")
    for md5, paths in list(duplicates.items())[:5]:
        rel_paths = [os.path.relpath(p, ROOT_DIR) for p in paths]
        print(f"  Hash {md5[:8]}: {rel_paths}")

# 6. Filename Analysis, Design IDs, & Labels
print("\n--- 6. FILENAME PATTERNS & DESIGN ID EXTRACTION ---")
sample_filenames = defaultdict(list)
for img_path in image_files:
    rel_dir = os.path.relpath(os.path.dirname(img_path), ROOT_DIR)
    sample_filenames[rel_dir].append(os.path.basename(img_path))

for fldr, fnames in sorted(sample_filenames.items()):
    print(f"\nFolder: '{fldr}' (Sample 5 filenames):")
    for fn in fnames[:5]:
        print(f"  {fn}")

# Let's inspect Roboflow / SKU filename structure
print("\n--- 7. DESIGN ID & COLOR VARIANT ANALYSIS ---")
base_design_map = defaultdict(list)
color_words = ['red', 'blue', 'green', 'yellow', 'pink', 'black', 'white', 'orange', 'purple', 'violet', 'cyan', 'gold', 'golden', 'maroon', 'beige', 'grey', 'gray', 'peach', 'teal', 'navy', 'magenta', 'mustard', 'cream', 'copper', 'silver', 'brown', 'multicolor', 'multi']

color_variant_candidates = defaultdict(list)

for img_path in image_files:
    fname = os.path.basename(img_path)
    rel_dir = os.path.relpath(os.path.dirname(img_path), ROOT_DIR)
    
    # Strip Roboflow hash if present
    clean_base = re.sub(r'(_jpg|_jpeg|_png)?\.rf\.[a-f0-9]+\.(jpg|jpeg|png)$', '', fname, flags=re.IGNORECASE)
    clean_base = os.path.splitext(clean_base)[0]
    
    base_design_map[(rel_dir, clean_base)].append(img_path)
    
    # Check if filename contains color words
    fname_lower = fname.lower()
    found_colors = [c for c in color_words if c in fname_lower]
    if found_colors:
        # Check SKU or title prefix
        prefix = re.sub(r'(red|blue|green|yellow|pink|black|white|orange|purple|violet|cyan|gold|golden|maroon|beige|grey|gray|peach|teal|navy|magenta|mustard|cream|copper|silver|brown|multicolor|multi)', '', clean_base, flags=re.IGNORECASE)
        color_variant_candidates[prefix].append((found_colors, fname, rel_dir))

print(f"Total (folder, base_name) instances: {len(base_design_map)}")
multi_instance_bases = {k: v for k, v in base_design_map.items() if len(v) > 1}
print(f"Base designs with multiple instances within the same folder: {len(multi_instance_bases)}")

if multi_instance_bases:
    print("\nSample multi-instance base designs:")
    for (fldr, bname), paths in list(multi_instance_bases.items())[:5]:
        print(f"  [{fldr}] '{bname}': {len(paths)} images")

print(f"\nColor variant prefix groups detected: {len(color_variant_candidates)}")
multi_color_groups = {k: v for k, v in color_variant_candidates.items() if len(v) > 1}
print(f"Prefix groups with multiple colorway mentions: {len(multi_color_groups)}")
if multi_color_groups:
    print("Sample multi-colorway design groups:")
    for prefix, items in list(multi_color_groups.items())[:5]:
        print(f"  Prefix '{prefix[:40]}':")
        for cols, fn, fldr in items[:3]:
            print(f"    - Colors: {cols} | File: {fn[:50]} | Folder: {fldr}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
