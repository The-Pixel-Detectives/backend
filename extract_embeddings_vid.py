import os
import cv2
import numpy as np
import json  # Import json to save metadata as JSON
from model import Clip4Clip
from tqdm import tqdm

# Initialize Clip4Clip model
clip_model = Clip4Clip()

def process_video(video_path, output_base_dir, segment_duration=5, fps=25):
    """
    Process a video to extract frames and compute embeddings using CLIP4CLIP.

    Args:
        video_path (str): Path to the input video file.
        output_base_dir (str): Base directory to save extracted embeddings.
        segment_duration (int): Duration of each segment in seconds (default: 5).
        fps (int): Frames per second of the video (default: 25).
    """
    # Get the video ID from the video path (filename without extension)
    video_id = os.path.basename(video_path).split('.')[0]

    # Create a specific output directory for this video
    output_dir = os.path.join(output_base_dir, video_id)
    os.makedirs(output_dir, exist_ok=True)

    # Capture video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = list(range(0, fps * segment_duration, 5))  # Select frames 1, 5, 10, etc.
    group_id = 0

    while True:
        # Calculate start and end frame for the current segment
        start_frame = group_id * (segment_duration - 1) * fps
        end_frame = start_frame + segment_duration * fps

        if start_frame >= total_frames:
            break

        embeddings = []
        metadata = []

        for idx in frame_indices:
            frame_number = start_frame + idx
            if frame_number >= total_frames:
                break

            # Set the video position to the frame number
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                continue

            # Extract embeddings using the model
            embedding = clip_model.extract_embedding(frame)
            embeddings.append(embedding)
            metadata.append({
                "group_id": group_id,
                "video_id": video_id,
                "frame_index": frame_number,
                "fps": fps
            })

            # Clear frame from memory
            del frame

        # Save embeddings for the segment in the specific video folder
        np.save(os.path.join(output_dir, f"{video_id}_group{group_id}_embeddings.npy"), np.array(embeddings))

        # Save metadata for the segment as JSON
        with open(os.path.join(output_dir, f"{video_id}_group{group_id}_metadata.json"), 'w') as json_file:
            json.dump(metadata, json_file, indent=4)

        group_id += 1

    cap.release()

if __name__ == "__main__":
    video_path = "video/L02_V001.mp4"  # Path to your video file
    output_base_dir = "../data/clip4clip_embeddings"  # Base directory to save output

    # Process the video
    process_video(video_path, output_base_dir)
