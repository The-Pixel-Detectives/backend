import os
from config import settings
from PIL import Image
import numpy as np
import math
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


def visualize_images(imgs, lim=5):
    num_row = math.ceil(len(imgs) / lim)
    rows = []
    blank_img = np.zeros_like(imgs[0])
    for _ in range(num_row):
        row = imgs[:lim]
        if len(row) < lim:
            row = row + [blank_img] * (lim - len(row))
        row = np.concatenate(row, axis=1)
        rows.append(row)

        if len(imgs) > lim:
            imgs = imgs[lim:]

    concat_img = np.concatenate(rows, axis=0)
    return concat_img
