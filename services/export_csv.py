import csv
import os
import cv2
from config import settings
import json


with open("./fps_map.json", "r") as f:
    fps_map = json.load(f)


# Function to get the FPS of a video
def get_video_fps(video_path):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_capture.release()
    return fps

# Function to calculate frame indices based on start and end times
def generate_frame_indices(video_id, start_time, end_time):
    fps = fps_map[video_id]
    indices = [t for t in range(int(start_time * fps), int(end_time * fps), 5)]

    # Reorder the indices to middle → left → right
    mid = len(indices) // 2
    result = [indices[mid]]

    left = mid - 1
    right = mid + 1

    while left >= 0 or right < len(indices):
        if left >= 0:
            result.append(indices[left])
            left -= 1
        if right < len(indices):
            result.append(indices[right])
            right += 1

    return result[:100]  # Limiting to 100 indices

# Function to export frame indices to CSV
def export_to_csv(video_id, frame_indices, filename, qa):
    filepath = os.path.join('exports', f"{filename}.csv")
    os.makedirs('exports', exist_ok=True)

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)

        if qa == "-1":
            # Write 2 columns (Video ID, Frame Index) - no need to include header
            for index in frame_indices:
                writer.writerow([video_id, index])
        else:
            # Write 3 columns (Video ID, Frame Index, QA) - no need to include header
            for index in frame_indices:
                writer.writerow([video_id, index, qa])

    return filepath
