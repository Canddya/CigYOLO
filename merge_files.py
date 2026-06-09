import os
import shutil

base_path = r"D:\Desktop\CigYOLO\data"
wrong_target = os.path.join(base_path, "Wrong")
right_target = os.path.join(base_path, "Right")

os.makedirs(wrong_target, exist_ok=True)
os.makedirs(right_target, exist_ok=True)

def merge_files():
    success_count = 0

    for root, dirs, files in os.walk(base_path):
        if root.startswith(wrong_target) or root.startswith(right_target):
            continue

        folder_name = os.path.basename(root)

        if folder_name == "错误":
            target_dir = wrong_target
        elif folder_name == "良好":
            target_dir = right_target
        else:
            continue

        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(target_dir, file)

            if os.path.exists(dest_file_path):
                parent_name = os.path.basename(os.path.dirname(root))
                name, ext = os.path.splitext(file)
                new_file_name = f"{name}_{parent_name}{ext}"
                dest_file_path = os.path.join(target_dir, new_file_name)

            shutil.copy2(src_file_path, dest_file_path)
            success_count += 1

    print(f"Merge completed! Successfully merged {success_count} files.")

if __name__ == "__main__":
    merge_files()