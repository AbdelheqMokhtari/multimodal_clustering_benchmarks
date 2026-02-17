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
├── build_RGB_D_nyu_depth.py (To be added soon)
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

## Feature Extraction

Once you have constructed your raw dataset folders (e.g., coco_cross/raw/image/...), you can use the extract_features.py script to extract multimodal embeddings and save them into a standardized MATLAB .mat file for downstream clustering and evaluation.

### What it does

The script iterates through the raw dataset and processes both the image and text views:

* **Image Features (View 1):** Uses a ResNet-50 model (pretrained on ImageNet) with average pooling to extract 2048-dimensional visual features.
    
* **Text Features (View 2):** Uses Gensim's GloVe-Wiki-Gigaword-300 model to process the captions. It tokenizes the text and averages the 300-dimensional word vectors to create a unified 300-dimensional document vector (I am aiming to use doc2vec but i couldn't find any available pretrained doc2vec wiki for now so i'am using Glove).
    
* **Data Alignment & Export:** Matches the paired views, registers the ground truth folder name as an integer class, and saves everything directly into the dataset folder (e.g., `coco_cross/coco_cross.mat`).

The resulting `.mat` file contains four variables:
* `X1`: The image feature matrix (Shape: `N x 2048`)
* `X2`: The text feature matrix (Shape: `N x 300`)
* `Y`: The ground truth label array (Shape: `N x 1`)
* `label_mapping`: A reference array mapping the integer labels in `Y` back to their original string class names.


### How to Run

Execute the script from your terminal, passing the name of the target dataset folder using the `--dataset` argument:

```bash
# Example: Extract features for the COCO-cross dataset
python extract_features.py --dataset coco_cross

# Example: Extract features for the Pascal dataset
python extract_features.py --dataset pascal

# Example: Extract features for the COCO-all dataset
python extract_features.py --dataset coco_all
```