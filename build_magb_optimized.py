import os
import ast
import json
import time
import argparse
import requests
import numpy as np
import pandas as pd

from io import BytesIO
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from scipy.io import savemat
from scipy.sparse import csr_matrix

import torch
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download


# =========================================================
# SETTINGS
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUPPORTED_DATASETS = [
    "Toys",
    "Movies",
    "Grocery"
]

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 64
NUM_DOWNLOAD_WORKERS = 32


# =========================================================
# DOWNLOAD SINGLE IMAGE
# =========================================================

def download_single_image(args):

    idx, url, save_dir = args

    save_path = os.path.join(
        save_dir,
        f"{idx}.jpg"
    )

    if os.path.exists(save_path):

        return True

    try:

        # Parse image URL list
        if isinstance(url, str):

            parsed = ast.literal_eval(url)

            if isinstance(parsed, list) and len(parsed) > 0:

                url = parsed[0]

        response = requests.get(
            url,
            timeout=3
        )

        img = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        img.save(save_path)

        return True

    except:

        return False


# =========================================================
# DOWNLOAD ALL IMAGES
# =========================================================

def download_all_images(image_urls, image_dir):

    print("\nDownloading images...")

    start = time.time()

    os.makedirs(image_dir, exist_ok=True)

    tasks = [

        (idx, url, image_dir)

        for idx, url in enumerate(image_urls)
    ]

    with ThreadPoolExecutor(
        max_workers=NUM_DOWNLOAD_WORKERS
    ) as executor:

        results = list(

            tqdm(
                executor.map(
                    download_single_image,
                    tasks
                ),
                total=len(tasks)
            )
        )

    success = sum(results)
    failed = len(results) - success

    elapsed = time.time() - start

    print(f"\nDownloaded: {success}")
    print(f"Failed    : {failed}")

    return elapsed, failed


# =========================================================
# IMAGE DATASET
# =========================================================

class ImageDataset(Dataset):

    def __init__(self, image_dir, num_images):

        self.image_dir = image_dir
        self.num_images = num_images

        self.transform = T.Compose([

            T.Resize((224, 224)),

            T.ToTensor(),

            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):

        return self.num_images

    def __getitem__(self, idx):

        path = os.path.join(
            self.image_dir,
            f"{idx}.jpg"
        )

        try:

            img = Image.open(path).convert("RGB")

            img = self.transform(img)

            return img

        except:

            return torch.zeros(3, 224, 224)


# =========================================================
# MINILM TEXT EMBEDDINGS
# =========================================================

def build_minilm_embeddings(texts):

    print("\nBuilding MiniLM text embeddings...")

    start = time.time()

    model = SentenceTransformer(
        TEXT_MODEL_NAME,
        device=DEVICE
    )

    X = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed = time.time() - start

    return X.astype(np.float32), elapsed


# =========================================================
# RESNET IMAGE EMBEDDINGS
# =========================================================

def build_resnet_embeddings(image_dir, num_images):

    print("\nBuilding ResNet50 image embeddings...")

    start = time.time()

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    # remove classifier
    model = torch.nn.Sequential(
        *list(model.children())[:-1]
    )

    model.eval()
    model.to(DEVICE)

    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------

    dataset = ImageDataset(
        image_dir,
        num_images
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    features = []

    with torch.no_grad():

        for batch in tqdm(loader):

            batch = batch.to(DEVICE)

            feat = model(batch)

            feat = feat.squeeze(-1).squeeze(-1)

            feat = feat.cpu().numpy()

            features.append(feat)

    X = np.concatenate(features, axis=0)

    elapsed = time.time() - start

    return X.astype(np.float32), elapsed


# =========================================================
# BUILD ORIGINAL MAGB GRAPH
# =========================================================

def build_original_graph(df):

    print("\nBuilding original MAGB graph...")

    start = time.time()

    id_to_idx = {

        pid: i

        for i, pid in enumerate(df["id"])
    }

    edges = set()

    for i, row in tqdm(df.iterrows(), total=len(df)):

        # -------------------------------------------------
        # ALSO BUY
        # -------------------------------------------------

        try:

            also_buy = ast.literal_eval(
                str(row["also_buy"])
            )

            if isinstance(also_buy, list):

                for item in also_buy:

                    if item in id_to_idx:

                        j = id_to_idx[item]

                        edges.add((i, j))

        except:
            pass

        # -------------------------------------------------
        # ALSO VIEW
        # -------------------------------------------------

        try:

            also_view = ast.literal_eval(
                str(row["also_view"])
            )

            if isinstance(also_view, list):

                for item in also_view:

                    if item in id_to_idx:

                        j = id_to_idx[item]

                        edges.add((i, j))

        except:
            pass

    edges = np.array(list(edges))

    rows = edges[:, 0]
    cols = edges[:, 1]

    values = np.ones(len(rows))

    A = csr_matrix(
        (values, (rows, cols)),
        shape=(len(df), len(df))
    )

    # symmetrize graph
    A = A + A.T
    A[A > 0] = 1

    elapsed = time.time() - start

    print("\nGraph built.")
    print("Nodes:", len(df))
    print("Edges:", A.nnz)

    return A, elapsed


# =========================================================
# MAIN
# =========================================================

def main(dataset_name):

    total_start = time.time()

    print("=" * 60)
    print(f"Processing dataset: {dataset_name}")
    print("=" * 60)

    # =====================================================
    # DIRECTORIES
    # =====================================================

    output_dir = os.path.join(
        "MAGB",
        dataset_name
    )

    image_dir = os.path.join(
        output_dir,
        "images"
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # =====================================================
    # DOWNLOAD CSV
    # =====================================================

    print("\nDownloading CSV metadata...")

    csv_start = time.time()

    csv_file = hf_hub_download(
        repo_id="Sherirto/MAGB",
        repo_type="dataset",
        filename=f"{dataset_name}/{dataset_name}.csv"
    )

    csv_time = time.time() - csv_start

    print(csv_file)

    # =====================================================
    # LOAD CSV
    # =====================================================

    df = pd.read_csv(csv_file)

    print("\nDataset shape:")
    print(df.shape)

    # =====================================================
    # TEXTS
    # =====================================================

    texts = df["text"].astype(str).tolist()

    # =====================================================
    # LABELS
    # =====================================================

    labels = df["label"].values.astype(np.int32)

    # =====================================================
    # IMAGE URLS
    # =====================================================

    image_urls = df["imageURLHighRes"].tolist()

    # =====================================================
    # DOWNLOAD IMAGES
    # =====================================================

    image_download_time, failed_images = download_all_images(
        image_urls,
        image_dir
    )

    # =====================================================
    # TEXT EMBEDDINGS
    # =====================================================

    X_text, text_time = build_minilm_embeddings(
        texts
    )

    # =====================================================
    # IMAGE EMBEDDINGS
    # =====================================================

    X_image, image_time = build_resnet_embeddings(
        image_dir,
        len(df)
    )

    # =====================================================
    # GRAPH
    # =====================================================

    A_behavior, graph_time = build_original_graph(
        df
    )

    # =====================================================
    # SAVE COMPLETE MATLAB FILE
    # =====================================================

    print("\nSaving MATLAB file...")

    mat_path = os.path.join(
        output_dir,
        f"{dataset_name}_Complete.mat"
    )

    savemat(

        mat_path,

        {
            "X_text": X_text,

            "X_image": X_image,

            "A_behavior": A_behavior,

            "labels": labels
        }
    )

    # =====================================================
    # SAVE SEPARATE FILES
    # =====================================================

    savemat(

        os.path.join(
            output_dir,
            f"{dataset_name}_Text.mat"
        ),

        {
            "X_text": X_text,
            "A_behavior": A_behavior,
            "labels": labels
        }
    )

    savemat(

        os.path.join(
            output_dir,
            f"{dataset_name}_Image.mat"
        ),

        {
            "X_image": X_image,
            "A_behavior": A_behavior,
            "labels": labels
        }
    )

    # =====================================================
    # METADATA
    # =====================================================

    total_time = time.time() - total_start

    metadata = {

        "dataset": dataset_name,

        "creation_time": datetime.now().isoformat(),

        "device": DEVICE,

        "num_instances": int(len(df)),

        "num_classes": int(
            len(np.unique(labels))
        ),

        "graph": {

            "source": [
                "also_buy",
                "also_view"
            ],

            "shape": list(A_behavior.shape),

            "edges": int(A_behavior.nnz)
        },

        "embeddings": {

            "text_model": TEXT_MODEL_NAME,

            "image_model": "ResNet50",

            "text_shape": list(
                X_text.shape
            ),

            "image_shape": list(
                X_image.shape
            )
        },

        "failed_image_downloads": int(
            failed_images
        ),

        "timing_seconds": {

            "csv_download": float(csv_time),

            "image_download": float(
                image_download_time
            ),

            "text_embeddings": float(
                text_time
            ),

            "image_embeddings": float(
                image_time
            ),

            "graph_build": float(
                graph_time
            ),

            "total": float(total_time)
        }
    }

    metadata_path = os.path.join(
        output_dir,
        f"{dataset_name}_metadata.json"
    )

    with open(metadata_path, "w") as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # =====================================================
    # SUMMARY
    # =====================================================

    print("\nSUCCESS")
    print("=" * 60)

    print("\nSaved files:")
    print(mat_path)

    print("\nShapes:")
    print("X_text      :", X_text.shape)
    print("X_image     :", X_image.shape)
    print("A_behavior  :", A_behavior.shape)
    print("labels      :", labels.shape)

    print("\nTotal Time:")
    print(f"{total_time:.2f} seconds")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="Toys",
        choices=SUPPORTED_DATASETS
    )

    args = parser.parse_args()

    main(args.dataset)