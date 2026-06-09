import os
import random
from PIL import Image
from tqdm import tqdm

# ================= Configuration Paths =================
SRC_WRONG = r"D:\Desktop\CigYOLO\data\Wrong"
SRC_RIGHT = r"D:\Desktop\CigYOLO\data\Right"
DATASET_ROOT = r"D:\Desktop\CigYOLO\dataset"
# ========================================================

def prepare_yolo_dataset_three_way(src_dir, category_name, train_ratio=0.7, val_ratio=0.1):
    """
    Read BMP files from source folder, convert to JPG, and split into train/val/test sets
    with ratio 7:1:2
    """
    if not os.path.exists(src_dir):
        print(f"Source folder not found: {src_dir}, skipping this category.")
        return

    files = [f for f in os.listdir(src_dir) if f.lower().endswith('.bmp')]
    random.shuffle(files)

    total_count = len(files)
    train_end = int(total_count * train_ratio)
    val_end = train_end + int(total_count * val_ratio)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    train_target_dir = os.path.join(DATASET_ROOT, "train", category_name)
    val_target_dir = os.path.join(DATASET_ROOT, "val", category_name)
    test_target_dir = os.path.join(DATASET_ROOT, "test", category_name)

    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(val_target_dir, exist_ok=True)
    os.makedirs(test_target_dir, exist_ok=True)

    print(f"\nProcessing category [{category_name}]: {total_count} images...")
    print(f" └─ Split: Train {len(train_files)} | Val {len(val_files)} | Test {len(test_files)}")

    def convert_and_save(file_list, target_path, desc_name):
        if not file_list:
            return
        for file in tqdm(file_list, desc=f" [{desc_name}]"):
            src_file_path = os.path.join(src_dir, file)
            file_name_without_ext = os.path.splitext(file)[0]
            dest_file_path = os.path.join(target_path, f"{file_name_without_ext}.jpg")

            try:
                with Image.open(src_file_path) as img:
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    img.save(dest_file_path, 'JPEG', quality=95)
            except Exception as e:
                print(f"Failed to convert {file}: {e}")

    convert_and_save(train_files, train_target_dir, "Train")
    convert_and_save(val_files, val_target_dir, "Val")
    convert_and_save(test_files, test_target_dir, "Test")

if __name__ == "__main__":
    random.seed(42)
    prepare_yolo_dataset_three_way(SRC_WRONG, "Wrong", train_ratio=0.7, val_ratio=0.1)
    prepare_yolo_dataset_three_way(SRC_RIGHT, "Right", train_ratio=0.7, val_ratio=0.1)
    print("\nDataset preparation and 7:1:2 split completed!")