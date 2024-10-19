import os
import cv2
# import pandas as pd
from qdrant_client import QdrantClient, models
import numpy as np
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
from uuid import uuid4


CLIP4CLIP_INDEX = "clip4clip_embeddings"
JINA_INDEX = "jina_embeddings"
SIGLIP_INDEX = "siglip_embeddings_opt"
SBIR_INDEX = "sbir_embeddings"

collection_name = SIGLIP_INDEX
# collection_name = SBIR_INDEX

if collection_name == CLIP4CLIP_INDEX:
    vector_dimension = 512
elif collection_name == JINA_INDEX:
    vector_dimension = 768
elif collection_name == SBIR_INDEX:
    vector_dimension = 768
elif collection_name == SIGLIP_INDEX:
    vector_dimension = 1152

# input_dir = "./data/clip4clip_embeddings/"
input_dir = "/Volumes/CSZoneT7/AIC/data/siglip_embeddings2"
video_root_dir = "/Volumes/CSZoneT7/AIC/data/videos/"
keyframe_dir = "/Volumes/CSZoneT7/AIC/data/keyframes/"
# input_dir = "../data/sbir_embeddings/"
# frame_map_dir = "../data/map-keyframes/"


client = QdrantClient(host="localhost", port=6333, timeout=60)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE, on_disk=True),
    shard_number=6,
    hnsw_config=models.HnswConfigDiff(on_disk=True),
    init_from=models.InitFrom(collection="siglip_embeddings")
)

