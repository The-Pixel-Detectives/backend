import cv2
import json
from tqdm import tqdm
from glob import glob


video_list = glob("/Volumes/CSZoneT7/AIC/data/videos/*/*/*.mp4")
print(len(video_list))

fps_map = {}
for video_path in tqdm(video_list):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_id = video_path.split("/")[-1].replace(".mp4", "")
    fps_map[video_id] = fps

with open("./fps_map.json", "w") as f:
    json.dump(fps_map, f)
