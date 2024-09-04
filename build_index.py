import os
import pandas as pd
from qdrant_client import QdrantClient
import numpy as np
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
from uuid import uuid4


CLIP4CLIP_INDEX = "clip4clip_embeddings"
JINA_INDEX = "jina_embeddings"
SBIR_INDEX = "sbir_embeddings"

# collection_name = JINA_INDEX
collection_name = SBIR_INDEX

if collection_name == CLIP4CLIP_INDEX:
    vector_dimension = 512
elif collection_name == JINA_INDEX:
    vector_dimension = 768
elif collection_name == SBIR_INDEX:
    vector_dimension = 768

# input_dir = "./data/clip4clip_embeddings/"
# input_dir = "../data/jina_embeddings/"
input_dir = "../data/sbir_embeddings/"
frame_map_dir = "../data/map-keyframes/"

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
for group in tqdm(group_list):
    group_dir = os.path.join(input_dir, group)
    if not os.path.isdir(group_dir):
        continue

    video_list = os.listdir(group_dir)
    for video in video_list:
        video_dir = os.path.join(group_dir, video)
        if not os.path.isdir(video_dir):
            continue

        frame_df = pd.read_csv(os.path.join(frame_map_dir, f"{video}.csv"))

        embedding_list = os.listdir(video_dir)
        embedding_list = [f for f in embedding_list if ".npy" in f]
        points = []
        for file in embedding_list:
            embedding = np.load(os.path.join(video_dir, file))
            keyframe = int(file.replace(".npy", "").replace(".jpg", "").strip())
            item = frame_df[frame_df["n"] == keyframe].iloc[0]
            frame_idx = item["frame_idx"]
            fps = item["fps"]
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

        client.upsert(
            collection_name=collection_name,
            points=points
        )
