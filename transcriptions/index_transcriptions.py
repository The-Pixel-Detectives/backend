import os
import cv2
import json
from tqdm import tqdm
from search_transcriptions import index_transcription

# Base directory containing all video folders (transcription folders)
# Updated base directory pointing to the transcriptions folder in Google Drive
base_transcription_dir = "/Volumes/Data/AI/Challenge/data/transcriptions/"
video_root_dir = "/Volumes/T7/AIC/data/videos/"

def index_transcriptions():
    # Loop through each folder in the base transcription directory
    group_list = os.listdir(base_transcription_dir)
    group_list.sort()
    group_list = ["Videos_L17"]
    for group in group_list:
        group_dir = os.path.join(base_transcription_dir, group)
        if not os.path.isdir(group_dir):
            continue

        group_id = group.split("_")[-1]
        video_list = os.listdir(group_dir)
        video_list.sort()
        print(group_id)
        for video_id in tqdm(video_list):
            folder_path = os.path.join(group_dir, video_id)
            # Check if the folder contains transcription files
            if not os.path.isdir(folder_path):
                continue

            video_path = os.path.join(video_root_dir, group, "video", video_id + ".mp4")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            # print(video_id, fps)
            # Process each transcription file in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    filepath = os.path.join(folder_path, filename)

                    if "(1)" in filename:
                        os.remove(filepath)
                        continue
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.read()

                        # Extract frame_idx from filename
                        frame_idx = filename.split('.')[0]
                        # print(f"Processing file: {filename} for video_id: {video_id}, frame_idx: {frame_idx}")

                        # Save the transcription and metadata to Elasticsearch
                        index_transcription(filename, content, video_id, group_id, frame_idx, fps)

                        # Save the metadata as JSON for reference
                        output_dir = f"../data/metadata/Videos_{group_id}/{video_id}"
                        os.makedirs(output_dir, exist_ok=True)
                        with open(os.path.join(output_dir, f"{filename}_metadata.json"), 'w') as json_file:
                            json.dump({
                                "group_id": group_id,
                                "video_id": video_id,
                                "frame_idx": int(frame_idx),
                                "fps": fps,
                                "transcription_segments": content.split("\n")
                            }, json_file, indent=4)

                        # print(f"Indexed and saved metadata for {filename}")

if __name__ == "__main__":
    index_transcriptions()
