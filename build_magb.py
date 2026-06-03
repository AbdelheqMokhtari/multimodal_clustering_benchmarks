import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from transformers import CLIPProcessor, CLIPModel
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
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

BATCH_SIZE = 64
NUM_DOWNLOAD_WORKERS = 32


# =========================================================
# DOWNLOAD IMAGE
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

        if isinstance(url, str):

            parsed = ast.literal_eval(url)

            if isinstance(parsed, list) and len(parsed) > 0:

                url = parsed[0]

        response = requests.get(
            url,
            timeout=5
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

            pil_img = Image.open(path).convert("RGB")

            tensor_img = self.transform(pil_img)

            return pil_img, tensor_img

        except:

            blank = Image.new(
                "RGB",
                (224, 224)
            )

            blank_tensor = torch.zeros(3, 224, 224)

            return blank, blank_tensor


# =========================================================
# CUSTOM COLLATE
# =========================================================

def custom_collate(batch):

    pil_imgs = [item[0] for item in batch]

    tensor_imgs = torch.stack(
        [item[1] for item in batch]
    )

    return pil_imgs, tensor_imgs


# =========================================================
# MINILM TEXT EMBEDDINGS
# =========================================================

def build_minilm_embeddings(texts):

    print("\nBuilding MiniLM embeddings...")

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
# CLIP TEXT EMBEDDINGS
# =========================================================

def build_clip_text_embeddings(texts):

    print("\nBuilding CLIP text embeddings...")

    start = time.time()

    model = CLIPModel.from_pretrained(
        CLIP_MODEL_NAME
    ).to(DEVICE)

    processor = CLIPProcessor.from_pretrained(
        CLIP_MODEL_NAME
    )

    model.eval()

    features = []

    with torch.no_grad():

        for i in tqdm(range(0, len(texts), BATCH_SIZE)):

            batch = texts[i:i+BATCH_SIZE]

            inputs = processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            inputs = {
                k: v.to(DEVICE)
                for k, v in inputs.items()
            }

            feat = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

            feat = feat.detach().cpu().numpy()

            features.append(feat)

    X = np.concatenate(features, axis=0)

    elapsed = time.time() - start

    return X.astype(np.float32), elapsed


# =========================================================
# IMAGE EMBEDDINGS
# =========================================================

def build_image_embeddings(image_dir, num_images):

    print("\nBuilding image embeddings...")

    start = time.time()

    # =====================================================
    # RESNET
    # =====================================================

    resnet = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    resnet = torch.nn.Sequential(
        *list(resnet.children())[:-1]
    )

    resnet.eval()
    resnet.to(DEVICE)

    # =====================================================
    # CLIP
    # =====================================================

    clip_model = CLIPModel.from_pretrained(
        CLIP_MODEL_NAME
    ).to(DEVICE)

    clip_processor = CLIPProcessor.from_pretrained(
        CLIP_MODEL_NAME
    )

    clip_model.eval()

    # =====================================================
    # DATASET
    # =====================================================

    dataset = ImageDataset(
        image_dir,
        num_images
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate
    )

    # =====================================================
    # STORAGE
    # =====================================================

    resnet_features = []
    clip_features = []

    with torch.no_grad():

        for pil_imgs, imgs in tqdm(loader):

            # -------------------------------------------------
            # RESNET
            # -------------------------------------------------

            imgs = imgs.to(DEVICE)

            feat_resnet = resnet(imgs)

            feat_resnet = feat_resnet.squeeze(-1).squeeze(-1)

            feat_resnet = feat_resnet.cpu().numpy()

            resnet_features.append(feat_resnet)

            # -------------------------------------------------
            # CLIP
            # -------------------------------------------------

            clip_inputs = clip_processor(
                images=pil_imgs,
                return_tensors="pt"
            )

            clip_inputs = {
                k: v.to(DEVICE)
                for k, v in clip_inputs.items()
            }

            feat_clip = clip_model.get_image_features(
                pixel_values=clip_inputs["pixel_values"]
            )

            feat_clip = feat_clip.detach().cpu().numpy()

            clip_features.append(feat_clip)

    X_resnet = np.concatenate(
        resnet_features,
        axis=0
    )

    X_clip = np.concatenate(
        clip_features,
        axis=0
    )

    elapsed = time.time() - start

    return (
        X_resnet.astype(np.float32),
        X_clip.astype(np.float32),
        elapsed
    )


# =========================================================
# BUILD GRAPH
# =========================================================

def build_original_graph(df):

    print("\nBuilding MAGB graph...")

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

    # symmetrize
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

    print("\nDownloading CSV...")

    csv_file = hf_hub_download(
        repo_id="Sherirto/MAGB",
        repo_type="dataset",
        filename=f"{dataset_name}/{dataset_name}.csv"
    )

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
    # IMAGES
    # =====================================================

    image_urls = df["imageURLHighRes"].tolist()

    download_all_images(
        image_urls,
        image_dir
    )

    # =====================================================
    # TEXT FEATURES
    # =====================================================

    X_text_minilm, _ = build_minilm_embeddings(
        texts
    )

    X_text_clip, _ = build_clip_text_embeddings(
        texts
    )

    # =====================================================
    # IMAGE FEATURES
    # =====================================================

    (
        X_image_resnet,
        X_image_clip,
        _

    ) = build_image_embeddings(
        image_dir,
        len(df)
    )

    # =====================================================
    # GRAPH
    # =====================================================

    A_behavior, _ = build_original_graph(df)

    # =====================================================
    # SAVE
    # =====================================================

    print("\nSaving MATLAB file...")

    mat_path = os.path.join(
        output_dir,
        f"{dataset_name}_Complete.mat"
    )

    savemat(

        mat_path,

        {
            "X_text_minilm": X_text_minilm,

            "X_text_clip": X_text_clip,

            "X_image_resnet": X_image_resnet,

            "X_image_clip": X_image_clip,

            "A_behavior": A_behavior,

            "labels": labels
        }
    )

    # =====================================================
    # METADATA
    # =====================================================

    metadata = {

        "dataset": dataset_name,

        "num_instances": int(len(df)),

        "num_classes": int(
            len(np.unique(labels))
        ),

        "text_minilm_shape": list(
            X_text_minilm.shape
        ),

        "text_clip_shape": list(
            X_text_clip.shape
        ),

        "image_resnet_shape": list(
            X_image_resnet.shape
        ),

        "image_clip_shape": list(
            X_image_clip.shape
        ),

        "graph_shape": list(
            A_behavior.shape
        ),

        "graph_edges": int(
            A_behavior.nnz
        ),

        "device": DEVICE,

        "created": datetime.now().isoformat()
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

    total_time = time.time() - total_start

    print("\nSUCCESS")
    print("=" * 60)

    print("\nSaved:")
    print(mat_path)

    print("\nShapes:")
    print("MiniLM Text :", X_text_minilm.shape)
    print("CLIP Text   :", X_text_clip.shape)
    print("ResNet Img  :", X_image_resnet.shape)
    print("CLIP Img    :", X_image_clip.shape)
    print("Graph       :", A_behavior.shape)
    print("Labels      :", labels.shape)

    print(f"\nTotal Time: {total_time:.2f} sec")


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