from qdrant_client import QdrantClient, models


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

client.delete(
    collection_name=collection_name,
    points_selector=models.FilterSelector(
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="group",
                    match=models.MatchValue(value="L15"),
                ),
            ],
        )
    ),
)
