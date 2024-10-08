import os
import cv2
import numpy as np
from tqdm import tqdm
from model import Clip4Clip, JinaCLIP, SBIRModel


# needed input dimensions for the CNN
input_dir = "/Volumes/T7/AIC/data/keyframes"
output_dir = "/Volumes/T7/AIC/data/jina_embeddings"
# output_dir = "../data/sbir_embeddings"
os.makedirs(output_dir, exist_ok=True)

# img2vec = Clip4Clip()
img2vec = JinaCLIP()
# img2vec = SBIRModel()

group_list = os.listdir(input_dir)
for group in group_list:
    # group_dir = os.path.join(input_dir, group, "keyframes")
    group_dir = os.path.join(input_dir, group)
    if not os.path.isdir(group_dir):
        continue

    video_list = os.listdir(group_dir)
    for video in video_list:
        video_dir = os.path.join(group_dir, video)
        if not os.path.isdir(video_dir):
            continue

        out_path = os.path.join(output_dir, group, video)
        os.makedirs(out_path, exist_ok=True)

        img_list = os.listdir(video_dir)
        img_list = [f for f in img_list if ".jpeg" in f and f[0] != '.']

        print(group, video)
        for img_name in tqdm(img_list):
            filepath = os.path.join(video_dir, img_name)
            image = cv2.imread(filepath)
            x = img2vec.extract_embedding(image)
            np.save(os.path.join(out_path, img_name + ".npy"), x)
