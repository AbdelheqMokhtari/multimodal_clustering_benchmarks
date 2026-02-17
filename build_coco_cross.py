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
OUT_ROOT = Path("coco_cross/raw")

IMAGE_DIRS = [
    COCO_ROOT / "train2017",
    # COCO_ROOT / "val2017",
]

ANN_FILES = [
    COCO_ROOT / "annotations" / "instances_train2017.json",
    # COCO_ROOT / "annotations" / "instances_val2017.json",
]

CAPTION_FILES = [
    COCO_ROOT / "annotations" / "captions_train2017.json",
    # COCO_ROOT / "annotations" / "captions_val2017.json",
]

# EXACT categories from the paper
COCO_CROSS_CATEGORIES = {
    "stop sign",
    "airplane",
    "suitcase",
    "pizza",
    "cell phone",
    "person",
    "giraffe",
    "kite",
    "toilet",
    "clock",
}


# --------------------------------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------
def main():
    img_out = OUT_ROOT / "images"
    txt_out = OUT_ROOT / "texts"
    img_out.mkdir(parents=True, exist_ok=True)
    txt_out.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load annotations (train + val)
    # --------------------------------------------------
    cat_id_to_name = {}
    img_to_cat_names = defaultdict(list)

    for ann_file in ANN_FILES:
        data = load_json(ann_file)

        for cat in data["categories"]:
            cat_id_to_name[cat["id"]] = cat["name"]

        for ann in data["annotations"]:
            img_to_cat_names[ann["image_id"]].append(
                cat_id_to_name[ann["category_id"]]
            )

    # --------------------------------------------------
    # Filter: EXACTLY ONE CATEGORY (not one instance)
    # --------------------------------------------------
    valid_images = {}
    class_counter = Counter()

    for img_id, cat_names in img_to_cat_names.items():
        unique_cats = set(cat_names)

        if len(unique_cats) != 1:
            continue

        cat_name = next(iter(unique_cats))
        if cat_name in COCO_CROSS_CATEGORIES:
            valid_images[img_id] = cat_name
            class_counter[cat_name] += 1

    # --------------------------------------------------
    # VERBOSE distribution
    # --------------------------------------------------
    print("\n[✓] COCO-cross class distribution (train+val):")
    for cls in sorted(COCO_CROSS_CATEGORIES):
        print(f"  {cls:<12} : {class_counter.get(cls, 0)}")

    print(f"\n[✓] Total samples: {sum(class_counter.values())}")

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
    # Copy images and save captions
    # --------------------------------------------------
    copied = 0
    for img_id, cls in valid_images.items():
        img_name = f"{img_id:012d}.jpg"

        src = None
        for d in IMAGE_DIRS:
            p = d / img_name
            if p.exists():
                src = p
                break
        if src is None:
            continue

        (img_out / cls).mkdir(exist_ok=True)
        (txt_out / cls).mkdir(exist_ok=True)

        shutil.copy(src, img_out / cls / img_name)

        with open(txt_out / cls / f"{img_id:012d}.txt", "w") as f:
            for c in img_to_captions.get(img_id, []):
                f.write(c.strip() + "\n")

        copied += 1

    print(f"\n[✓] COCO-cross built: {copied} samples")


# --------------------------------------------------
if __name__ == "__main__":
    main()
