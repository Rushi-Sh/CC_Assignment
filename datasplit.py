import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
dataset_path = "CC_dataset/PlantVillage_for_object_detection/Dataset"
images_path = os.path.join(dataset_path, "images")
labels_path = os.path.join(dataset_path, "labels")
output_path = "DatasetNew"

# Get all image filenames
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg'))]

# Train-Test-Validation Split (70%, 20%, 10%)
train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.33, random_state=42)

# Helper function to move files
def move_files(file_list, split_type):
    for file_name in file_list:
        # Move images
        shutil.move(os.path.join(images_path, file_name), f"{output_path}/{split_type}/images/{file_name}")

        # Move corresponding labels
        label_file = file_name.rsplit(".", 1)[0] + ".txt"
        if os.path.exists(os.path.join(labels_path, label_file)):
            shutil.move(os.path.join(labels_path, label_file), f"{output_path}/{split_type}/labels/{label_file}")

# Create parent folder and subfolders
os.makedirs(output_path, exist_ok=True)
for split in ["train", "val", "test"]:
    os.makedirs(f"{output_path}/{split}/images", exist_ok=True)
    os.makedirs(f"{output_path}/{split}/labels", exist_ok=True)

# Move files to respective directories
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("Dataset successfully divided into 'train_test_valid' folder with train, val, and test splits!")
