import os
from qdrant_client import QdrantClient
import numpy as np
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
from uuid import uuid4


CLIP4CLIP_INDEX = "clip4clip_embeddings"
JINA_INDEX = "jina_embeddings"

collection_name = JINA_INDEX

if collection_name == CLIP4CLIP_INDEX:
    vector_dimension = 512
elif collection_name == JINA_INDEX:
    vector_dimension = 768

# input_dir = "./data/clip4clip_embeddings/"
input_dir = "../data/jina_embeddings/"

client = QdrantClient(host="localhost", port=6333)

if not client.collection_exists(collection_name):
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

        embedding_list = os.listdir(video_dir)
        embedding_list = [f for f in embedding_list if ".npy" in f]
        points = []
        for file in embedding_list:
            embedding = np.load(os.path.join(video_dir, file))
            points.append(PointStruct(
                id=str(uuid4()),
                vector=embedding.tolist(),
                payload={
                    "group": group,
                    "video": video,
                    "keyframe": file.replace(".npy", "").replace(".jpg", "").strip()
                }
            ))

        client.upsert(
            collection_name=collection_name,
            points=points
        )
