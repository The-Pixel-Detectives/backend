import os
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance
from tqdm import tqdm
from uuid import uuid4
from config import settings  # Import settings as before

# Initialize OpenAI API key
openai.api_key = settings.OPENAI_API_KEY

# Define Qdrant collections and vector dimensions
TRANSCRIPTION_INDEX = "transcription_embeddings"
CLIP4CLIP_INDEX = "clip4clip_embeddings"
JINA_INDEX = "jina_embeddings"
SBIR_INDEX = "sbir_embeddings"

# Select the collection name
collection_name = TRANSCRIPTION_INDEX  # Or select any other index if needed

# Set vector dimensions based on the collection type
if collection_name == CLIP4CLIP_INDEX:
    vector_dimension = 512
elif collection_name == JINA_INDEX or collection_name == SBIR_INDEX:
    vector_dimension = 768
elif collection_name == TRANSCRIPTION_INDEX:
    # Transcription-specific embedding dimension (for OpenAI `text-embedding-ada-002`)
    vector_dimension = 1536

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Create collection if it doesn't exist
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE),
    )

# Transcription directory
transcription_dir = settings.TRANSCRIPTION_DIR  # Use from config, or replace with direct path
transcription_files = os.listdir(transcription_dir)

# Function to generate embeddings from transcription text
def generate_embedding(text: str) -> list:
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

# Set batch size
batch_size = 200  # Increase the batch size for efficiency if needed
points = []

# Process and insert transcriptions in batches
for filename in tqdm(transcription_files, desc="Processing Transcriptions"):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(transcription_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        transcription_content = f.read()

    # Generate embedding for the transcription
    embedding = generate_embedding(transcription_content)

    # Create a point with metadata and embedding
    point = PointStruct(
        id=str(uuid4()),
        vector=embedding,
        payload={
            "filename": filename,
            "content": transcription_content
        }
    )
    points.append(point)

    # Insert points in batches to Qdrant
    if len(points) >= batch_size:
        client.upsert(collection_name=collection_name, points=points)
        points = []

# Insert any remaining points
if points:
    client.upsert(collection_name=collection_name, points=points)

print(f"Successfully indexed {len(transcription_files)} transcriptions into the '{collection_name}' collection.")
