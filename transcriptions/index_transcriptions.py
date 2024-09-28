import os
import json
from search_transcriptions import index_transcription

# Base directory containing all video folders (transcription folders)
# Updated base directory pointing to the transcriptions folder in Google Drive
base_transcription_dir = "/Users/vothanhhoanganh/Programming/Volumes/"

# Fixed group ID (the first part of the folder name, e.g., "L21")
group_id = "L21"  # Assuming "L21" is the constant part for all folders

def index_transcriptions():
    # Loop through each folder in the base transcription directory
    for folder_name in os.listdir(base_transcription_dir):
        folder_path = os.path.join(base_transcription_dir, folder_name)
        
        # Check if the folder contains transcription files
        if os.path.isdir(folder_path) and folder_name.startswith(group_id):
            # Extract video_id from folder name (e.g., "L21_V001")
            video_id = folder_name
            
            # Process each transcription file in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    filepath = os.path.join(folder_path, filename)

                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.read()

                        # Extract frame_idx from filename
                        frame_idx = filename.split('.')[0]
                        print(f"Processing file: {filename} for video_id: {video_id}, frame_idx: {frame_idx}")

                        # Save the transcription and metadata to Elasticsearch
                        index_transcription(filename, content, video_id, group_id, frame_idx)

                        # Save the metadata as JSON for reference
                        output_dir = f"output/metadata/{group_id}"
                        os.makedirs(output_dir, exist_ok=True)
                        with open(os.path.join(output_dir, f"{filename}_metadata.json"), 'w') as json_file:
                            json.dump({
                                "group_id": group_id,
                                "video_id": video_id,
                                "frame_idx": filename.split('.')[0],
                                "transcription_segments": content.split("\n")
                            }, json_file, indent=4)

                        print(f"Indexed and saved metadata for {filename}")

if __name__ == "__main__":
    index_transcriptions()
