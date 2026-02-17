import os
import glob
import argparse
import numpy as np
import scipy.io as sio

# Image processing imports
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

# Text processing imports
import gensim.downloader as api
from gensim.utils import simple_preprocess

def main(dataset_name):
    print(f"--- Processing Dataset: {dataset_name} ---")
    
    # Paths
    base_dir = dataset_name
    image_dir = os.path.join(base_dir, 'raw', 'images')
    text_dir = os.path.join(base_dir, 'raw', 'texts')
    output_mat_path = os.path.join(base_dir, f'{dataset_name}.mat')

    if not os.path.exists(image_dir) or not os.path.exists(text_dir):
        raise FileNotFoundError(f"Ensure that the '{image_dir}' and '{text_dir}' folders exist.")

    # 1. Load ResNet-50 Model (ImageNet pretrained, pooling='avg' gives 2048-D)
    print("Loading ResNet-50 model...")
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # 2. Load 300-D Wikipedia Text Model (Auto-downloads ~376MB on first run)
    print("Loading 300-D Wikipedia Text Model (This may take a minute on the first run)...")
    try:
        text_model = api.load("glove-wiki-gigaword-300")
    except Exception as e:
        print(f"Error loading text model: {e}")
        print("Please check your internet connection so Gensim can download the model.")
        return

    # Data containers
    X1_images = []
    X2_texts = []
    Y_labels = []
    
    # Label mapping dictionary
    class_to_id = {}
    current_class_id = 1 

    # 3. Process Files recursively
    # Searches through all subfolders inside raw/image/ (e.g. raw/image/dog/dog_01.jpg)
    image_paths = glob.glob(os.path.join(image_dir, '**', '*.*'), recursive=True)
    image_paths = [p for p in image_paths if os.path.isfile(p)]
    
    print(f"Found {len(image_paths)} images. Starting extraction...")
    
    for count, img_path in enumerate(image_paths, 1):
        if count % 100 == 0:
            print(f"Processed {count}/{len(image_paths)} files...")

        # Extract class name from the parent folder (e.g., "dog")
        class_name = os.path.basename(os.path.dirname(img_path))
        
        filename = os.path.basename(img_path)
        name_only = os.path.splitext(filename)[0]
        
        # Construct path to the corresponding text file: raw/text/class_name/file.txt
        txt_path = os.path.join(text_dir, class_name, f"{name_only}.txt")
        
        if not os.path.exists(txt_path):
            print(f"Warning: No matching text file found at {txt_path}. Skipping.")
            continue

        # -- Register Class Label --
        if class_name not in class_to_id:
            class_to_id[class_name] = current_class_id
            current_class_id += 1
        label_id = class_to_id[class_name]

        # -- Extract Image Features (ResNet-50) --
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            # Using model(x) instead of predict(x) is much faster for single images in loops
            img_feature = resnet_model(x, training=False) 
            X1_images.append(img_feature[0].numpy())
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

        # -- Extract Text Features (300-D Wiki Model) --
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
                
            # Tokenize and lowercase the text
            tokenized_text = simple_preprocess(text_content)
            
            # Keep only words that exist in our Wikipedia vocabulary
            valid_words = [word for word in tokenized_text if word in text_model]
            
            if valid_words:
                # Average the 300-D word vectors to create a single 300-D document vector
                txt_feature = np.mean(text_model[valid_words], axis=0)
            else:
                # Fallback if the text file is empty or contains no recognizable words
                txt_feature = np.zeros(300) 
                
            X2_texts.append(txt_feature)
            
        except Exception as e:
            print(f"Error processing text {txt_path}: {e}")
            # Ensure we don't desync our images and labels if text fails
            X1_images.pop() 
            continue

        # -- Store Label --
        Y_labels.append(label_id)

    # 4. Convert lists to NumPy matrices
    X1_matrix = np.array(X1_images) # Shape: (N, 2048)
    X2_matrix = np.array(X2_texts)  # Shape: (N, 300)
    Y_matrix = np.array(Y_labels).reshape(-1, 1) # Shape: (N, 1)

    # Prepare class mapping for MATLAB (Array of [id, class_name])
    id_to_class = {v: k for k, v in class_to_id.items()}
    mapping_array = np.array([ [k, v] for k, v in id_to_class.items() ], dtype=object)

    # 5. Save to .mat file
    print(f"\nSaving features to {output_mat_path}...")
    sio.savemat(output_mat_path, {
        'X1': X1_matrix, 
        'X2': X2_matrix, 
        'Y': Y_matrix,
        'label_mapping': mapping_array
    })
    
    print(f"✅ Successfully saved {X1_matrix.shape[0]} matched samples!")
    print(f"   Matrix X1 (Images) : {X1_matrix.shape}")
    print(f"   Matrix X2 (Text)   : {X2_matrix.shape}")
    print(f"   Matrix Y  (Labels) : {Y_matrix.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ResNet50 and 300-D Wikipedia Text features")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset folder (e.g., coco_cross)")
    
    args = parser.parse_args()
    main(args.dataset)