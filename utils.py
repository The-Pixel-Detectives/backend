import os
from config import settings
from PIL import Image
import numpy as np
from io import BytesIO


def get_video_path(group_id, video_id):
    return os.path.join(settings.VIDEO_DIR, f"Videos_{group_id}", "video", f"{video_id}.mp4")


def get_keyframe_path(group_id, video_id, idx):
    return os.path.join(settings.KEYFRAME_DIR, f"Keyframes_{group_id}", "keyframes", video_id, f"{idx:03d}.jpg")


def get_sketch_img_path(id):
    return os.path.join(settings.SKETCH_IMG_DIR, f"{id}.png")


def load_image_into_numpy_array(data):
    img = Image.open(BytesIO(data))
    return np.array(img)

