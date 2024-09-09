import os
import json  # Import json to handle metadata files
from qdrant_client import QdrantClient
import numpy as np
from qdrant_client.models import VectorParams, Distance, PointStruct
from tqdm import tqdm
from uuid import uuid4

RESNET_INDEX = "resnet_embeddings"
HIST_INDEX = "hist_embeddings"
CLIP4CLIP_INDEX = "clip4clip_embeddings"

collection_name = CLIP4CLIP_INDEX

vector_dimension = 512
if collection_name == RESNET_INDEX:
    input_dir = "./data/resnet_embeddings/"
elif collection_name == HIST_INDEX:
    input_dir = "./data/hist_embeddings/"
    vector_dimension = 1440
elif collection_name == CLIP4CLIP_INDEX:
    input_dir = "./data/clip4clip_embeddings/"

client = QdrantClient(host="localhost", port=6333, timeout=60.0)

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE),
    )

video_list = os.listdir(input_dir)
batch_size = 100  # Define the batch size
points = []  # Initialize the points list

def normalize_embeddings(embeddings):
    """
    Normalize the embeddings and compute the mean embedding.
    
    Args:
        embeddings (numpy.ndarray): Array of embeddings.
    
    Returns:
        numpy.ndarray: The mean normalized embedding.
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    
    # Compute the mean of the normalized embeddings
    mean_embedding = np.mean(normalized_embeddings, axis=0)
    
    # Normalize the mean embedding
    mean_embedding /= np.linalg.norm(mean_embedding)
    
    return mean_embedding

for video in tqdm(video_list):
    video_path = os.path.join(input_dir, video)
    if not os.path.isdir(video_path):
        continue

    embedding_list = os.listdir(video_path)
    embedding_list = [f for f in embedding_list if "_embeddings.npy" in f]  # Only select embedding files

    for file in embedding_list:
        embedding_path = os.path.join(video_path, file)
        metadata_path = embedding_path.replace('_embeddings.npy', '_metadata.json')  # Corresponding metadata file path

        # Load the embedding data
        embedding = np.load(embedding_path)

        # Normalize and compute the mean embedding
        mean_embedding = normalize_embeddings(embedding)

        # Check the embedding vector dimension
        if mean_embedding.shape[0] != vector_dimension:
            print(f"Skipping {file}: Embedding dimension mismatch. Expected {vector_dimension}, got {mean_embedding.shape[0]}")
            continue
        
        # Initialize the payload with basic video information
        payload = {
            "video": video,
            "keyframe": file.replace(".npy", "").replace(".jpg", "").strip()
        }

        # Check if the corresponding metadata file exists
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as meta_file:
                metadata = json.load(meta_file)
                
                # Ensure metadata is a dictionary
                if isinstance(metadata, dict):
                    payload.update(metadata)
                else:
                    print(f"Unexpected metadata format in {metadata_path}, skipping.")
                    continue
        else:
            print(f"Metadata file not found for {file}, skipping metadata inclusion.")

        # Create a point with the mean embedding and enriched payload
        points.append(PointStruct(
            id=str(uuid4()),
            vector=mean_embedding.tolist(),
            payload=payload
        ))

        # If the batch size is reached, upsert points to the collection
        if len(points) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            points = []  # Clear the points list after upserting

# Upsert any remaining points after processing all files
if points:  # Only upsert if there are remaining valid points
    client.upsert(
        collection_name=collection_name,
        points=points
    )
