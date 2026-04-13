import os
import shutil
from collections import defaultdict

master_folder = r"C:/Users/ASUS/Desktop/New folder/eda_output/L2_district_targets"
destination_folder = r"C:/Users/ASUS/Desktop/New folder/eda_output/overall"
target_filename = "health_target_dashboard.png"  # exact file name

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

counter = 1

for folder in os.listdir(master_folder):
    subfolder_path = os.path.join(master_folder, folder)

    if os.path.isdir(subfolder_path):
        source_file = os.path.join(subfolder_path, target_filename)

        if os.path.exists(source_file):
            # Rename to avoid overwrite
            name, ext = os.path.splitext(target_filename)
            new_name = f"{name}_{counter}{ext}"
            dest_file = os.path.join(destination_folder, new_name)

            shutil.move(source_file, dest_file)
            print(f"Moved: {source_file} → {dest_file}")

            counter += 1