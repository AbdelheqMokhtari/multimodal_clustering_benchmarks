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

from scipy.io import savemat
from scipy.sparse import csr_matrix

import torch
import torchvision.transforms as T
import torchvision.models as models

from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download


# =========================================================
# SETTINGS
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUPPORTED_DATASETS = [
    "Toys",
    "Movies",
    "Grocery",
    "Reddit-S",
    "Reddit-M"
]

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

IMAGE_DIM = 2048


# =========================================================
# DOWNLOAD IMAGE
# =========================================================

def download_image(url):

    try:

        response = requests.get(
            url,
            timeout=10
        )

        img = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        return img

    except:

        return None


# =========================================================
# BUILD TEXT EMBEDDINGS
# =========================================================

def build_text_embeddings(texts):

    print("\nBuilding text embeddings (MiniLM)...")

    start = time.time()

    model = SentenceTransformer(
        TEXT_MODEL_NAME,
        device=DEVICE
    )

    X = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    elapsed = time.time() - start

    return X.astype(np.float32), elapsed


# =========================================================
# BUILD IMAGE EMBEDDINGS
# =========================================================

def build_image_embeddings(image_urls):

    print("\nBuilding image embeddings (ResNet50)...")

    start = time.time()

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    # remove classification head
    model = torch.nn.Sequential(
        *list(model.children())[:-1]
    )

    model.eval()
    model.to(DEVICE)

    transform = T.Compose([

        T.Resize((224, 224)),

        T.ToTensor(),

        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    features = []

    failed_images = 0

    for url in tqdm(image_urls):

        try:

            # Parse stringified list
            if isinstance(url, str):

                parsed = ast.literal_eval(url)

                if isinstance(parsed, list) and len(parsed) > 0:

                    url = parsed[0]

            img = download_image(url)

            if img is None:

                features.append(
                    np.zeros(IMAGE_DIM)
                )

                failed_images += 1

                continue

            x = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():

                feat = model(x)

            feat = feat.squeeze().cpu().numpy()

            features.append(feat)

        except:

            features.append(
                np.zeros(IMAGE_DIM)
            )

            failed_images += 1

    elapsed = time.time() - start

    return np.array(features).astype(np.float32), elapsed, failed_images


# =========================================================
# BUILD ORIGINAL MAGB GRAPH
# =========================================================

def build_original_graph(df):

    print("\nBuilding ORIGINAL MAGB graph...")
    print("Using also_buy + also_view relations")

    start = time.time()

    # -----------------------------------------------------
    # Product ID -> node index
    # -----------------------------------------------------

    id_to_idx = {

        pid: i for i, pid in enumerate(df["id"])
    }

    # -----------------------------------------------------
    # Edge set
    # -----------------------------------------------------

    edges = set()

    for i, row in tqdm(df.iterrows(), total=len(df)):

        # =================================================
        # also_buy
        # =================================================

        try:

            also_buy = ast.literal_eval(
                str(row["also_buy"])
            )

            if isinstance(also_buy, list):

                for item in also_buy:

                    if item in id_to_idx:

                        j = id_to_idx[item]

                        edges.add((i, j))
                        edges.add((j, i))

        except:
            pass

        # =================================================
        # also_view
        # =================================================

        try:

            also_view = ast.literal_eval(
                str(row["also_view"])
            )

            if isinstance(also_view, list):

                for item in also_view:

                    if item in id_to_idx:

                        j = id_to_idx[item]

                        edges.add((i, j))
                        edges.add((j, i))

        except:
            pass

    # -----------------------------------------------------
    # Convert to sparse adjacency matrix
    # -----------------------------------------------------

    edges = np.array(list(edges))

    rows = edges[:, 0]
    cols = edges[:, 1]

    values = np.ones(len(rows))

    A = csr_matrix(
        (values, (rows, cols)),
        shape=(len(df), len(df))
    )

    elapsed = time.time() - start

    print("\nGraph completed.")
    print("Nodes :", len(df))
    print("Edges :", A.nnz)

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
    # OUTPUT DIRECTORY
    # =====================================================

    output_dir = os.path.join(
        "MAGB",
        dataset_name
    )

    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # DOWNLOAD CSV
    # =====================================================

    print("\nDownloading dataset metadata...")

    download_start = time.time()

    csv_file = hf_hub_download(
        repo_id="Sherirto/MAGB",
        repo_type="dataset",
        filename=f"{dataset_name}/{dataset_name}.csv"
    )

    download_time = time.time() - download_start

    print("\nCSV downloaded:")
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
    # TEXT EMBEDDINGS
    # =====================================================

    X_text, text_time = build_text_embeddings(
        texts
    )

    # =====================================================
    # IMAGE EMBEDDINGS
    # =====================================================

    X_image, image_time, failed_images = build_image_embeddings(
        image_urls
    )

    # =====================================================
    # ORIGINAL GRAPH
    # =====================================================

    A_behavior, graph_time = build_original_graph(
        df
    )

    # =====================================================
    # SAVE MATLAB
    # =====================================================

    mat_path = os.path.join(
        output_dir,
        f"{dataset_name}.mat"
    )

    print("\nSaving MATLAB file...")

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
    # METADATA JSON
    # =====================================================

    total_time = time.time() - total_start

    metadata = {

        "dataset_name": dataset_name,

        "creation_time": datetime.now().isoformat(),

        "device_used": DEVICE,

        "number_of_instances": int(len(df)),

        "text_embedding_shape": list(X_text.shape),

        "image_embedding_shape": list(X_image.shape),

        "adjacency_matrix_shape": list(A_behavior.shape),

        "number_of_graph_edges": int(A_behavior.nnz),

        "number_of_classes": int(
            len(np.unique(labels))
        ),

        "failed_image_downloads": int(
            failed_images
        ),

        "embedding_models": {

            "text_model": TEXT_MODEL_NAME,

            "image_model": "ResNet50"
        },

        "graph_construction": {

            "type": "Original MAGB graph",

            "sources": [
                "also_buy",
                "also_view"
            ]
        },

        "timing_seconds": {

            "download": float(download_time),

            "text_embedding": float(text_time),

            "image_embedding": float(image_time),

            "graph_construction": float(graph_time),

            "total": float(total_time)
        }
    }

    json_path = os.path.join(
        output_dir,
        f"{dataset_name}_metadata.json"
    )

    with open(json_path, "w") as f:

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
    print(json_path)

    print("\nShapes:")
    print("X_text      :", X_text.shape)
    print("X_image     :", X_image.shape)
    print("A_behavior  :", A_behavior.shape)
    print("labels      :", labels.shape)

    print("\nTiming:")
    print(f"Download        : {download_time:.2f} sec")
    print(f"Text embeddings : {text_time:.2f} sec")
    print(f"Image embeddings: {image_time:.2f} sec")
    print(f"Graph build     : {graph_time:.2f} sec")
    print(f"Total           : {total_time:.2f} sec")


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