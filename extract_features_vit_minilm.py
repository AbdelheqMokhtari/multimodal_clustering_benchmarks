"""
extract_features_vit_minilm.py
==============================
Extract **ViT-B/16** image features and **MiniLM** (Sentence-BERT) text
features from a raw multimodal dataset and save them to a .mat file.

Models
------
- Image : ``vit_base_patch16_224`` from **timm** (ImageNet-pretrained)
          → 768-dimensional CLS token embedding.
- Text  : ``all-MiniLM-L6-v2`` from **sentence-transformers**
          → 384-dimensional sentence embedding.

Usage
-----
    python extract_features_vit_minilm.py --dataset coco_cross
    python extract_features_vit_minilm.py --dataset coco_all
    python extract_features_vit_minilm.py --dataset pascal
    python extract_features_vit_minilm.py --dataset coco_cross coco_all pascal

Output
------
For each dataset a ``<dataset>/<dataset>_vit_minilm.mat`` file is created:

    X1  – image features      (N × 768)   float32
    X2  – text features       (N × 384)   float32
    Y   – ground-truth labels  (N × 1)    int
    label_mapping – [[int_id, class_name], ...]
"""

import os
import glob
import argparse
import numpy as np
import scipy.io as sio

import torch
from PIL import Image

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sentence_transformers import SentenceTransformer


# ── helpers ──────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


def encode_captions_sbert(captions: list[str], sbert_model) -> np.ndarray:
    """
    Encode a list of captions with Sentence-BERT (MiniLM) and return the
    mean-pooled, L2-normalised embedding.
    """
    if not captions:
        return np.zeros(sbert_model.get_sentence_embedding_dimension(),
                        dtype=np.float32)

    embeddings = sbert_model.encode(captions, convert_to_numpy=True)  # (K, D)
    mean_emb = embeddings.mean(axis=0)                                 # (D,)
    norm = np.linalg.norm(mean_emb) + 1e-10
    return (mean_emb / norm).astype(np.float32)


# ── main processing ─────────────────────────────────────────────────
def process_dataset(
    dataset_name: str,
    vit_model,
    vit_transform,
    sbert_model,
    device: str,
    img_dim: int,
    txt_dim: int,
):
    print(f"\n{'='*60}")
    print(f"  Processing dataset: {dataset_name}")
    print(f"{'='*60}")

    base_dir = dataset_name
    image_dir = os.path.join(base_dir, "raw", "images")
    text_dir = os.path.join(base_dir, "raw", "texts")
    output_mat = os.path.join(base_dir, f"{dataset_name}_vit_minilm.mat")

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

        # ── image features (ViT-B/16 → 768-D) ──
        try:
            pil_img = Image.open(img_path).convert("RGB")
            img_tensor = vit_transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = vit_model(img_tensor)  # (1, 768)
            img_feat = img_feat[0].cpu().numpy().astype(np.float32)
            # L2 normalise
            img_feat = img_feat / (np.linalg.norm(img_feat) + 1e-10)
        except Exception as e:
            print(f"  ⚠ image error {img_path}: {e}")
            skipped += 1
            continue

        # ── text features (MiniLM / Sentence-BERT → 384-D) ──
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                captions = [
                    line.strip() for line in f.readlines() if line.strip()
                ]
            txt_feat = encode_captions_sbert(captions, sbert_model)
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
    X1 = np.stack(X1_images)                  # (N, 768)
    X2 = np.stack(X2_texts)                   # (N, 384)
    Y = np.array(Y_labels).reshape(-1, 1)     # (N, 1)

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
    print(f"     X1 (img): {X1.shape}  [ViT-B/16]")
    print(f"     X2 (txt): {X2.shape}  [MiniLM]")
    print(f"     Y       : {Y.shape}")
    print(f"     Classes : {len(class_to_id)}")


# ── entry point ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract ViT-B/16 + MiniLM features for multimodal datasets"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="One or more dataset folder names (e.g., coco_cross coco_all pascal)",
    )
    parser.add_argument(
        "--sbert-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-BERT / MiniLM model name (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── load ViT-B/16 ──
    print("Loading ViT-B/16 (ImageNet-pretrained) ...")
    vit_model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=0
    )
    vit_model = vit_model.to(device).eval()

    data_cfg = resolve_data_config(vit_model.pretrained_cfg)
    vit_transform = create_transform(**data_cfg)
    img_dim = vit_model.num_features  # 768

    # ── load MiniLM ──
    print(f"Loading MiniLM model: {args.sbert_model} ...")
    sbert_model = SentenceTransformer(args.sbert_model, device=device)
    txt_dim = sbert_model.get_sentence_embedding_dimension()

    print(f"  Image feature dim : {img_dim}")
    print(f"  Text  feature dim : {txt_dim}")

    for ds in args.dataset:
        process_dataset(ds, vit_model, vit_transform, sbert_model, device,
                        img_dim, txt_dim)

    print("\nAll done! 🎉")


if __name__ == "__main__":
    main()
