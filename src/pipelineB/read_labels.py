"""
This script 
  - reads the labels from the dat file for the given dataset
  - deals with the missing label cases
  - saves the labels for each folder in a csv file

References:
  - Provided materials: read_labels.py
"""
# ========== Imports ================
import os
import pickle
import glob
import csv

# ========== User Control ===========
# set True to generate the labels for the dataset
PROCESS_TRAIN = False
PROCESS_TEST1 = False
PROCESS_TEST2 = True

# ========= Outline Cases ===========
MISSING_LABEL_LIST = {
    "76-1studyroom2/0002111-000070763319",
    "mit_32_d507/0004646-000155745519",
    "harvard_c11/0000006-000000187873",
    "mit_lab_hj/0001106-000044777376",
    "mit_lab_hj/0001326-000053659116"
}

FULL_NEGATIVE_SEQUENCES = {
    "mit_gym_z_squash",
    "harvard_tea_2"
}

# ========== Path Config ============
dataset_base_path = "data/CW2-Dataset/data/"
output_base_path = "data/pipelineB_data/"

# ========== Utils ==================
def generate_labels_for_sequence(sequence_path, output_folder, folder_key):
    label_path = os.path.join(sequence_path, "labels/tabletop_labels.dat")
    image_path = os.path.join(sequence_path, "image/")
    
    if not os.path.exists(label_path):
        print(f"[INFO] No label file in {sequence_path}, assuming full-negative sequence.")
        image_files = sorted(glob.glob(os.path.join(image_path, "*.jpg")))
        label_rows = []
        for img_path in image_files:
            filename = os.path.splitext(os.path.basename(img_path))[0]
            label_rows.append([filename, 0])
    else:
        with open(label_path, 'rb') as f:
            labels = pickle.load(f)

        image_files = sorted(glob.glob(os.path.join(image_path, "*.jpg")))
        label_rows = []

        for img_file, polygon_list in zip(image_files, labels):
            filename = os.path.splitext(os.path.basename(img_file))[0]
            rel_path_id = os.path.join(folder_key, filename)
            if rel_path_id in MISSING_LABEL_LIST:
                continue
            label = 1 if len(polygon_list) > 0 else 0
            label_rows.append([filename, label])

    # Write CSV to pipelineB folder
    output_path = os.path.join(output_folder, "labels.csv")
    os.makedirs(output_folder, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(label_rows)
    print(f"[DONE] Labels saved to {output_path}")

# ========== Main Process ===========
for mode, flag in zip(["Training_Data", "Test_Data_1"], [PROCESS_TRAIN, PROCESS_TEST1]):
    if not flag:
        continue

    if mode == "Training_Data":
        folders = sorted(glob.glob(os.path.join(dataset_base_path, "mit_*")))
    elif mode == "Test_Data_1":
        folders = sorted(glob.glob(os.path.join(dataset_base_path, "harvard_*")))
    
    for folder in folders:
        top_folder = os.path.basename(folder)
        subfolders = sorted(glob.glob(os.path.join(folder, "*")))
        for subfolder in subfolders:
            sequence_name = os.path.basename(subfolder)
            folder_key = f"{top_folder}/{sequence_name}"
            output_folder = os.path.join(output_base_path, mode, top_folder)
            generate_labels_for_sequence(subfolder, output_folder, folder_key)

# ---------- Test2 ----------
if PROCESS_TEST2:
    print("[INFO] Processing Test_Data_2 (UCL Dataset)")

    label_txt_path = "data/ucl_dataset/labels/ucl_labels.txt"
    image_dir = "data/ucl_dataset/image"
    output_folder = os.path.join(output_base_path, "Test_Data_2", "ucl_dataset")
    os.makedirs(output_folder, exist_ok=True)

    # Read image files (ensure matching only existing .jpg)
    available_images = {
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(image_dir, "*.jpg"))
    }

    label_rows = []
    with open(label_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            filename, label = parts
            if filename in available_images:
                label_rows.append([filename, int(label)])
            else:
                print(f"[WARN] Skipping {filename}: not found in image folder.")

    # Write CSV
    output_path = os.path.join(output_folder, "labels.csv")
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(label_rows)

    print(f"[DONE] UCL labels saved to {output_path}")
