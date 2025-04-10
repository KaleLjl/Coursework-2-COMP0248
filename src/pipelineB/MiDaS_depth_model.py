"""
This script 
  - loads the MiDaS depth estimation model; 
  - processes images to generate and store depth estimations;
  - splits the depth data into training and testing sets.

References:
  - MiDas: https://pytorch.org/hub/intelisl_midas_v2/
  - MiDas: https://github.com/isl-org/MiDaS

Requirements:
  - Download MiDas model from its official repository and place it in the `src/pipelineB/MiDaS` directory.
  - Change the path configurations if needed.
"""
# ========== Imports ================
import os
import cv2
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

# ========== User Control ===========
# set True to generate depth estimations for the corresponding dataset
PROCESS_TRAIN = False  # MIT
PROCESS_TEST1 = False # Harvard
PROCESS_TEST2 = True # RealSense


# ========== MiDaS Loading ==========
#Select the model type
model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

# Load the MiDaS model
# midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_repo_path = "./src/pipelineB/MiDaS"
midas = torch.hub.load(
    repo_or_dir=midas_repo_path,
    model=model_type,
    source="local",
    trust_repo=True
)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)

# Set the model to evaluation mode
midas.eval()

# Load transforms to resize and normalize the image for large or small model
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transforms = torch.hub.load(
    repo_or_dir=midas_repo_path,
    model="transforms",
    source="local",
    trust_repo=True
)

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# ========== Path Config ========
dataset_paths = {
    "Training_Data": "data/CW2-Dataset/data/mit_*",
    "Test_Data_1": "data/CW2-Dataset/data/harvard_*",
    "Test_Data_2": "data/ucl_dataset/image" 
}
output_base_path = "data/pipelineB_data"

# ========== Utils ==============
def normalize_depth(depth):
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_MiDaS_on_image(img_path):
    # Load the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply transforms
    input_batch = transform(img).to(device)

    # Predict and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    normalized_output = normalize_depth(output)

    return normalized_output

# ========== Main Process ==============
# -- Process Training and Test1 Data ----------
if PROCESS_TRAIN or PROCESS_TEST1:
    for split in ["Training_Data", "Test_Data_1"]:
        if not (split == "Training_Data" and PROCESS_TRAIN) and not (split == "Test_Data_1" and PROCESS_TEST1):
            continue

        print(f"\n[INFO] Processing {split}...")

        dataset_dirs = sorted(glob(dataset_paths[split]))
        for dataset_dir in dataset_dirs:
            folder_name = os.path.basename(dataset_dir)
            subfolders = sorted(glob(os.path.join(dataset_dir, "*")))

            for subfolder in subfolders:
                image_dir = os.path.join(subfolder, "image")
                image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
                if not image_paths:
                    continue

                output_dir = os.path.join(output_base_path, split, folder_name)
                ensure_dir(output_dir)

                for img_path in tqdm(image_paths, desc=f"{split}/{folder_name}"):
                    filename = os.path.splitext(os.path.basename(img_path))[0]
                    depth_norm = run_MiDaS_on_image(img_path)

                    # Save
                    np.save(os.path.join(output_dir, f"{filename}.npy"), depth_norm)
                    depth_vis = (depth_norm * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(output_dir, f"{filename}.png"), depth_vis)

# ---------- Test2 ----------
if PROCESS_TEST2:
    print("\n[INFO] Processing Test_Data_2 (UCL dataset)...")
    image_paths = sorted(glob(os.path.join(dataset_paths["Test_Data_2"], "*.jpg")))
    output_dir = os.path.join(output_base_path, "Test_Data_2", "ucl_dataset")
    ensure_dir(output_dir)

    for img_path in tqdm(image_paths, desc="Test_Data_2/ucl_dataset"):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        depth_norm = run_MiDaS_on_image(img_path)

        np.save(os.path.join(output_dir, f"{filename}.npy"), depth_norm)
        depth_vis = (depth_norm * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{filename}.png"), depth_vis)

print("\n Processing complete.")
