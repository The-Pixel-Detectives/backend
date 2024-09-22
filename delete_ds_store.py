import os

def delete_ds_store_files(directory):
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is .DS_Store
            if file == '.DS_Store':
                file_path = os.path.join(root, file)
                try:
                    # Delete the .DS_Store file
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    # Replace 'your_directory_path' with the path to the directory you want to clean up
    directory_to_clean = '../data/qdrant'
    delete_ds_store_files(directory_to_clean)
