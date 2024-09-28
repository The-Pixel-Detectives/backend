import os
import cv2
# import pandas as pd
from qdrant_client import QdrantClient
import numpy as np
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
from uuid import uuid4


CLIP4CLIP_INDEX = "clip4clip_embeddings"
JINA_INDEX = "jina_embeddings"
SBIR_INDEX = "sbir_embeddings"

collection_name = JINA_INDEX
# collection_name = SBIR_INDEX

if collection_name == CLIP4CLIP_INDEX:
    vector_dimension = 512
elif collection_name == JINA_INDEX:
    vector_dimension = 768
elif collection_name == SBIR_INDEX:
    vector_dimension = 768

# input_dir = "./data/clip4clip_embeddings/"
input_dir = "/Volumes/T7/AIC/data/embeddings/"
video_root_dir = "/Volumes/T7/AIC/data/videos/"
keyframe_dir = "/Volumes/T7/AIC/data/keyframes/"
# input_dir = "../data/sbir_embeddings/"
# frame_map_dir = "../data/map-keyframes/"


client = QdrantClient(host="localhost", port=6333)

if not client.collection_exists(collection_name):
    if collection_name == SBIR_INDEX:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dimension, distance=Distance.EUCLID),
        )
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE),
        )

group_list = os.listdir(input_dir)
group_list.sort()
# group_list = group_list[5:]
batch_size = 200
for group in tqdm(group_list):
    group_dir = os.path.join(input_dir, group)
    if not os.path.isdir(group_dir):
        continue

    video_list = os.listdir(group_dir)
    video_list = [v for v in video_list if v[0] != "."]
    for video in video_list:
        video_dir = os.path.join(group_dir, video)
        if not os.path.isdir(video_dir):
            print("Skip", video_dir)
            continue

        # frame_df = pd.read_csv(os.path.join(frame_map_dir, f"{video}.csv"))
        video_path = os.path.join(video_root_dir, group, "video", video + ".mp4")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(video, fps)

        keyframe_list = os.listdir(os.path.join(keyframe_dir, group, video))
        keyframe_list = [f for f in keyframe_list if ".jpeg" in f and f[0] != '.']

        embedding_list = os.listdir(video_dir)
        embedding_list = [f for f in embedding_list if ".npy" in f and f[0] != '.']

        if len(embedding_list) != len(keyframe_list):
            print("Different lengths for embeddings and keyframes", video, len(embedding_list), len(keyframe_list))

        points = []
        for file in embedding_list:
            embedding = np.load(os.path.join(video_dir, file))
            keyframe = int(file.replace(".npy", "").replace(".jpeg", "").strip())
            # item = frame_df[frame_df["n"] == keyframe].iloc[0]
            # frame_idx = item["frame_idx"]
            # fps = item["fps"]
            frame_idx = keyframe
            points.append(PointStruct(
                id=str(uuid4()),
                vector=embedding.tolist(),
                payload={
                    "group": group.split("_")[-1],
                    "video": video,
                    "keyframe": keyframe,
                    "frame_idx": frame_idx,
                    "fps": fps,
                }
            ))

            if len(points) > batch_size:
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                points = []

        if len(points) > 0:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
