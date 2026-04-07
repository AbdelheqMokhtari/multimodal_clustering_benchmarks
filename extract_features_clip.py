"""
extract_features_clip.py
========================
Extract CLIP (ViT-B/32) image and text features from a raw multimodal
dataset and save them to a .mat file.

Usage
-----
    python extract_features_clip.py --dataset coco_cross
    python extract_features_clip.py --dataset coco_all
    python extract_features_clip.py --dataset pascal
    python extract_features_clip.py --dataset coco_cross coco_all pascal   # all at once

Output
------
For each dataset a ``<dataset>/<dataset>_clip.mat`` file is created with:

    X1  – image features   (N × 512)   float32
    X2  – text features    (N × 512)   float32
    Y   – ground-truth labels (N × 1)  int
    label_mapping – [[int_id, class_name], ...]
"""

import os
import glob
import argparse
import numpy as np
import scipy.io as sio

import torch
import clip
from PIL import Image


# ── helpers ──────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def encode_text_lines(lines: list[str], model, device, batch_size: int = 64):
    """
    Tokenise and encode a list of captions with CLIP, then return
    the mean-pooled 512-D text embedding (L2-normalised).
    CLIP's tokeniser truncates to 77 tokens per caption automatically.
    """
    if not lines:
        return np.zeros(512, dtype=np.float32)

    all_feats = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i : i + batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)  # (B, 512)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())

    all_feats = np.concatenate(all_feats, axis=0)          # (num_captions, 512)
    mean_feat = all_feats.mean(axis=0)                      # (512,)
    mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-10)
    return mean_feat.astype(np.float32)


# ── main processing ─────────────────────────────────────────────────
def process_dataset(dataset_name: str, model, preprocess, device):
    print(f"\n{'='*60}")
    print(f"  Processing dataset: {dataset_name}")
    print(f"{'='*60}")

    base_dir = dataset_name
    image_dir = os.path.join(base_dir, "raw", "images")
    text_dir = os.path.join(base_dir, "raw", "texts")
    output_mat = os.path.join(base_dir, f"{dataset_name}_clip.mat")

    if not os.path.isdir(image_dir) or not os.path.isdir(text_dir):
        raise FileNotFoundError(
            f"Ensure that '{image_dir}' and '{text_dir}' exist."
        )

    # Discover all image files
    image_paths = sorted(
        p
        for p in glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True)
        if os.path.isfile(p) and is_image_file(p)
    )
    print(f"Found {len(image_paths)} images.")

    # Data containers
    X1_images: list[np.ndarray] = []
    X2_texts: list[np.ndarray] = []
    Y_labels: list[int] = []

    class_to_id: dict[str, int] = {}
    next_id = 1

    skipped = 0

    for idx, img_path in enumerate(image_paths, 1):
        if idx % 200 == 0 or idx == len(image_paths):
            print(f"  [{idx}/{len(image_paths)}]")

        class_name = os.path.basename(os.path.dirname(img_path))
        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(text_dir, class_name, f"{stem}.txt")

        if not os.path.isfile(txt_path):
            skipped += 1
            continue

        # ── class label ──
        if class_name not in class_to_id:
            class_to_id[class_name] = next_id
            next_id += 1
        label = class_to_id[class_name]

        # ── image features ──
        try:
            pil_img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img_tensor)  # (1, 512)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            img_feat = img_feat[0].cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"  ⚠ image error {img_path}: {e}")
            skipped += 1
            continue

        # ── text features ──
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                captions = [
                    line.strip() for line in f.readlines() if line.strip()
                ]
            txt_feat = encode_text_lines(captions, model, device)
        except Exception as e:
            print(f"  ⚠ text error {txt_path}: {e}")
            skipped += 1
            continue

        X1_images.append(img_feat)
        X2_texts.append(txt_feat)
        Y_labels.append(label)

    if not X1_images:
        print("  ❌ No valid samples found — skipping.")
        return

    # ── assemble matrices ──
    X1 = np.stack(X1_images)                 # (N, 512)
    X2 = np.stack(X2_texts)                  # (N, 512)
    Y = np.array(Y_labels).reshape(-1, 1)    # (N, 1)

    id_to_class = {v: k for k, v in class_to_id.items()}
    label_mapping = np.array(
        [[k, v] for k, v in id_to_class.items()], dtype=object
    )

    # ── save ──
    sio.savemat(output_mat, {
        "X1": X1,
        "X2": X2,
        "Y": Y,
        "label_mapping": label_mapping,
    })

    print(f"\n  ✅ Saved {output_mat}")
    print(f"     Samples : {X1.shape[0]}  (skipped {skipped})")
    print(f"     X1 (img): {X1.shape}")
    print(f"     X2 (txt): {X2.shape}")
    print(f"     Y       : {Y.shape}")
    print(f"     Classes : {len(class_to_id)}")


# ── entry point ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP (ViT-B/32) features for multimodal datasets"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="One or more dataset folder names (e.g., coco_cross coco_all pascal)",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-B/32",
        help="CLIP model variant (default: ViT-B/32)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading CLIP model: {args.clip_model} ...")
    model, preprocess = clip.load(args.clip_model, device=device)
    model.eval()

    for ds in args.dataset:
        process_dataset(ds, model, preprocess, device)

    print("\nAll done! 🎉")


if __name__ == "__main__":
    main()
