import os
import shutil

root_folder = "your/root/folder/path"

for dirpath, dirnames, filenames in os.walk(root_folder):
    if dirpath == root_folder:
        continue

    for filename in filenames:
        if filename.lower().endswith(".pdf"):
            source_path = os.path.join(dirpath, filename)

            parent_folder_name = os.path.basename(dirpath)

            new_filename = f"{parent_folder_name}_{filename}"
            destination_path = os.path.join(root_folder, new_filename)

            if os.path.exists(destination_path):
                base, ext = os.path.splitext(new_filename)
                counter = 1
                while os.path.exists(destination_path):
                    new_filename = f"{base}_{counter}{ext}"
                    destination_path = os.path.join(root_folder, new_filename)
                    counter += 1

            shutil.move(source_path, destination_path)
            print(f"Movido: {source_path} -> {destination_path}")

for dirpath, dirnames, filenames in os.walk(root_folder, topdown=False):
    if dirpath == root_folder:
        continue

    if not os.listdir(dirpath):
        os.rmdir(dirpath)
        print(f"Eliminada carpeta vacía: {dirpath}")
