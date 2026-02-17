#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil
from pathlib import Path
from collections import defaultdict, Counter


# --------------------------------------------------
# Configuration
# --------------------------------------------------
COCO_ROOT = Path("coco")
OUT_ROOT = Path("coco_all/raw")

IMAGES_DIRS = [
    COCO_ROOT / "train2017",
    # COCO_ROOT / "val2017",
]

INSTANCE_FILES = [
    COCO_ROOT / "annotations" / "instances_train2017.json",
    # COCO_ROOT / "annotations" / "instances_val2017.json",
]

CAPTION_FILES = [
    COCO_ROOT / "annotations" / "captions_train2017.json",
    # COCO_ROOT / "captions_val2017.json",
]

MIN_IMAGES_PER_CLASS = 100


# --------------------------------------------------
def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------
def main():
    img_out = OUT_ROOT / "images"
    txt_out = OUT_ROOT / "texts"
    img_out.mkdir(parents=True, exist_ok=True)
    txt_out.mkdir(parents=True, exist_ok=True)

    print("=== Building COCO-all ===")

    # --------------------------------------------------
    # Load instance annotations
    # --------------------------------------------------
    cat_id_to_name = {}
    img_to_categories = defaultdict(set)

    for ann_file in INSTANCE_FILES:
        data = load_json(ann_file)

        for cat in data["categories"]:
            cat_id_to_name[cat["id"]] = cat["name"]

        for ann in data["annotations"]:
            img_to_categories[ann["image_id"]].add(
                cat_id_to_name[ann["category_id"]]
            )

    # --------------------------------------------------
    # Keep images with exactly ONE category
    # --------------------------------------------------
    single_cat_images = {
        img_id: list(cats)[0]
        for img_id, cats in img_to_categories.items()
        if len(cats) == 1
    }

    print(f"[✓] Single-category images: {len(single_cat_images)}")

    # --------------------------------------------------
    # Distribution BEFORE ≥100 filtering
    # --------------------------------------------------
    raw_category_counts = Counter(single_cat_images.values())

    print("\n[INFO] Category distribution (single-category images):")
    for cat, cnt in raw_category_counts.most_common():
        print(f"  {cat:<20} : {cnt}")

    # --------------------------------------------------
    # Filter categories with < 100 images
    # --------------------------------------------------
    valid_categories = {
        cat for cat, cnt in raw_category_counts.items()
        if cnt >= MIN_IMAGES_PER_CLASS
    }

    print(
        f"\n[✓] Categories with ≥ {MIN_IMAGES_PER_CLASS} images: "
        f"{len(valid_categories)}"
    )

    valid_images = {
        img_id: cat
        for img_id, cat in single_cat_images.items()
        if cat in valid_categories
    }

    # --------------------------------------------------
    # FINAL distribution (COCO-all)
    # --------------------------------------------------
    final_category_counts = Counter(valid_images.values())

    print("\n[✓] FINAL COCO-all class distribution:")
    for cat, cnt in final_category_counts.most_common():
        print(f"  {cat:<20} : {cnt}")

    print(f"\n[✓] Final images after filtering: {sum(final_category_counts.values())}")

    # --------------------------------------------------
    # Load captions
    # --------------------------------------------------
    img_to_captions = defaultdict(list)
    for cap_file in CAPTION_FILES:
        data = load_json(cap_file)
        for ann in data["annotations"]:
            if ann["image_id"] in valid_images:
                img_to_captions[ann["image_id"]].append(ann["caption"])

    # --------------------------------------------------
    # Copy images + save captions
    # --------------------------------------------------
    copied = 0

    for img_id, category in valid_images.items():
        img_name = f"{img_id:012d}.jpg"

        src = None
        for d in IMAGES_DIRS:
            p = d / img_name
            if p.exists():
                src = p
                break

        if src is None:
            continue

        (img_out / category).mkdir(exist_ok=True)
        (txt_out / category).mkdir(exist_ok=True)

        shutil.copy(src, img_out / category / img_name)

        with open(txt_out / category / f"{img_id:012d}.txt", "w") as f:
            for c in img_to_captions[img_id]:
                f.write(c.strip() + "\n")

        copied += 1

    # --------------------------------------------------
    # Sanity check
    # --------------------------------------------------
    assert copied == sum(final_category_counts.values()), (
        "Mismatch between copied images and category counts!"
    )

    print("\n=== COCO-all built successfully ===")
    print(f"Total images  : {copied}")
    print(f"Total classes : {len(valid_categories)}")
    print(f"Images dir    : {img_out}")
    print(f"Texts dir     : {txt_out}")


# --------------------------------------------------
if __name__ == "__main__":
    main()
