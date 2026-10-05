import os
import json

def validate_notebook(nb_path):
    assert os.path.exists(nb_path), f"Notebook missing at {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"[OK] Notebook parses successfully as valid JSON (format {nb.get('nbformat')}.{nb.get('nbformat_minor')})")
    print(f"     Total Cells: {len(nb.get('cells', []))}")
    code_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'code']
    md_cells = [c for c in nb.get('cells', []) if c.get('cell_type') == 'markdown']
    print(f"     Markdown Cells: {len(md_cells)}, Code Cells: {len(code_cells)}")
    return True

def check_files():
    required_files = [
        "README.md",
        "approach_note.txt",
        "requirements.txt",
        ".gitignore",
        "saree_recognition_pipeline.ipynb",
        "outputs/results/evaluation_results.json",
        "outputs/results/summary_metrics.csv",
        "outputs/checkpoints/final_model.pth",
        "outputs/checkpoints/best_model_weights.pth",
        "src/__init__.py",
        "src/config.py",
        "src/dataset.py",
        "src/model.py",
        "src/loss.py",
        "src/metrics.py",
        "src/retrieval.py",
        "src/verification.py",
        "src/evaluate.py",
        "src/train.py"
    ]
    all_ok = True
    for f in required_files:
        if os.path.exists(f):
            print(f"[OK] {f} ({os.path.getsize(f):,} bytes)")
        else:
            print(f"[FAIL] Missing {f}")
            all_ok = False
    return all_ok

def check_for_dataset_files(root_dir="."):
    dataset_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    found_images = []
    for root, dirs, files in os.walk(root_dir):
        # Skip .git or cache dirs
        if ".git" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.lower().endswith(dataset_extensions):
                rel_path = os.path.relpath(os.path.join(root, file), root_dir)
                found_images.append(rel_path)
    
    print("\n--- Image / Asset Audit ---")
    print(f"Total image files in project: {len(found_images)}")
    for img in found_images:
        print(f"  - {img}")
    
    # Verify no dataset images
    has_dataset_img = any("data" in p or "archive" in p or "handloom" in p for p in found_images)
    if not has_dataset_img:
        print("[OK] ZERO dataset images found inside the project!")
    else:
        print("[WARN] Dataset images detected!")
    return not has_dataset_img

if __name__ == "__main__":
    print("=== GITHUB SUBMISSION READINESS AUDIT ===")
    v_nb = validate_notebook("saree_recognition_pipeline.ipynb")
    v_files = check_files()
    v_ds = check_for_dataset_files(".")
    print("\n=== FINAL STATUS ===")
    if v_nb and v_files and v_ds:
        print("ALL VERIFICATIONS PASSED! Project is 100% clean and ready for GitHub.")
    else:
        print("SOME CHECKS FAILED! Please review above.")
