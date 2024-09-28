from elasticsearch import Elasticsearch

# Connect to Elasticsearch
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Index name
index_name = 'transcriptions'

# Define the updated index mapping
mapping = {
    "mappings": {
        "properties": {
            "filename": {"type": "text"},
            "content": {"type": "text"},
            "video_id": {"type": "keyword"},
            "group_id": {"type": "keyword"},
            "frame_idx": {"type": "keyword"}
        }
    }
}

# Create the index if it doesn't exist
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")


# # Delete the existing index if it exists
# if es.indices.exists(index=index_name):
#     es.indices.delete(index=index_name)
#     print(f"Index '{index_name}' deleted.")

# # Create the index with the updated mapping
# es.indices.create(index=index_name, body=mapping)
# print(f"Index '{index_name}' created with updated mappings.")