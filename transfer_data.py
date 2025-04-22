import tkinter as tk
from tkinter import filedialog
import shutil
import os

def select_folder(title):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_selected = filedialog.askdirectory(title=title)
    return folder_selected

def transfer_data(source_folder, destination_folder, exclude_extensions=None):
    source_folder_name = os.path.basename(source_folder.rstrip(os.sep))
    destination_folder = os.path.join(destination_folder, source_folder_name)
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for root, dirs, files in os.walk(source_folder):
        relative_path = os.path.relpath(root, source_folder)
        dest_dir = os.path.join(destination_folder, relative_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for file in files:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(dest_dir, file)
            if exclude_extensions and 'images' in root and any(source_path.endswith(ext) for ext in exclude_extensions):
                continue
            shutil.copy2(source_path, destination_path)
    print(f"data transferred from {source_folder} to {destination_folder}, excluding: {exclude_extensions}")


if __name__ == "__main__":
    source_folder = select_folder("Select Data Source Folder")
    raw_data_folder = select_folder("Select Raw Data Destination Folder")
    processed_data_folder = select_folder("Select Processed Data Destination Folder")
    print(f"transfering data from {source_folder} to {processed_data_folder}")
    transfer_data(source_folder=source_folder, destination_folder=processed_data_folder, exclude_extensions=[".png"])
    print(f"transfering data from {source_folder} to {raw_data_folder}")
    transfer_data(source_folder=source_folder, destination_folder=raw_data_folder)
    print(f"Data transfer completed. You can savely delete {source_folder} now.")