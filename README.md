# multimodal-clustering-benchmarks

## Overview

**multimodal-clustering-benchmarks** is a data preparation repository for multimodal (image–text) clustering research.  
The repository provides **reproducible scripts** to download image–caption benchmarks following protocols described in the literature.

The focus is on datasets where **images and their captions are treated as multiple views of the same instance**, and where **ground-truth labels** are available for clustering evaluation.

This repository **does not distribute datasets**.  
Instead, it provides scripts that allow users to download datasets from official or license-compliant sources and construct standardized benchmark subsets locally.

---

## Supported Datasets

The repository currently supports:

- **Pascal Sentences**
- **Flickr30K** (downloaded via HuggingFace datasets with license enforcement)
- **COCO-cross** (subset of MS-COCO with 10 supercategories)
- **COCO-all** (subset of MS-COCO with ≥100 images per category)

All datasets are prepared in a raw format consisting of:
- image files
- corresponding caption files (one text file per image)
- consistent image–text alignment via shared IDs

---

## Repository Structure

```text
multimodal-clustering-benchmarks/
├── download_pascal_sentences.py
├── download_flickr30k.py
├── build_coco_cross.py
├── build_coco_all.py
└── README.md
```

> **Note**  
> Only scripts are included in the repository.  
> Datasets are downloaded and constructed locally by the user.

---

## MS-COCO Setup (COCO-cross and COCO-all)

COCO-cross and COCO-all are constructed subsets of MS-COCO following published multimodal clustering protocols.

### Step 1 — Download MS-COCO 2017

Download the official MS-COCO 2017 files using a browser or any download manager:

- **Train images**  
  http://images.cocodataset.org/zips/train2017.zip

- **Validation images**  
  http://images.cocodataset.org/zips/val2017.zip

- **Annotations**  
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip

After extracting the archives, place them under a local `coco/` directory with the following structure:

```text
coco/
├── train2017/
├── val2017/
└── annotations/
```

### Step 2 — Build COCO-cross

COCO-cross retains images that contain objects from **exactly one COCO supercategory** and assigns each image a single label.

Run:
```bash
python build_coco_cross.py
```

Output:

```text
coco_cross/
└── raw/
    ├── images/
    └── texts/
```

- ~7.4k images  
- 10 supercategories  
- 5 captions per image  

### Step 3 — Build COCO-all

COCO-cross retains images that contain objects from **exactly one COCO supercategory** and assigns each image a single label.

Run:
```bash
python build_coco_all.py
```

Output:

```text
coco_all/
└── raw/
    ├── images/
    └── texts/
```

- ~23k images 
- 43 categories 
- 5 captions per image  

