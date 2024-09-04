import os
from glob import glob
from qdrant_client import QdrantClient
import numpy as np
import math
import cv2
from model import Clip4Clip, JinaCLIP, SBIRModel


img_dir = "../data/keyframes"
CLIP4CLIP_INDEX = "clip4clip_embeddings"
JINA_INDEX = "jina_embeddings"
SBIR_INDEX = "sbir_embeddings"

collection_name = SBIR_INDEX

client = QdrantClient(host="localhost", port=6333)

if collection_name == CLIP4CLIP_INDEX:
    img2vec = Clip4Clip()
elif collection_name == JINA_INDEX:
    img2vec = JinaCLIP()
elif collection_name == SBIR_INDEX:
    img2vec = SBIRModel()


while True:
    if collection_name in [CLIP4CLIP_INDEX, JINA_INDEX]:
        text_query = input("Query: ").strip()
        if text_query == "q":
            break

        query_vector = img2vec.extract_text_embedding(text_query)
        query_vector = query_vector.tolist()
    else:
        img_path = input("Img Path: ")
        img_path = img_path.strip()
        if img_path == "q" or img_path == "":
            break

        image = cv2.imread(img_path)
        # query_vector = img2vec.extract_embedding(image).tolist()
        query_vector = img2vec.extract_sketch_embedding(img_path).tolist()
        print(query_vector)

    k = 11
    hits = client.search(
       collection_name=collection_name,
       query_vector=query_vector,
       limit=k
    )

    # visualize result
    vis_dim = 224

    images = []
    if collection_name not in [CLIP4CLIP_INDEX, JINA_INDEX]:
        image = cv2.resize(image, (vis_dim, vis_dim))
        img = cv2.putText(image, "Query", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 1)
        images = [image]

    for point in hits:
        group = point.payload['group']
        video = point.payload['video']
        keyframe = point.payload['keyframe']
        score = point.score
        img_path = os.path.join(img_dir, f"Keyframes_{group}", "keyframes", video, f"{keyframe:03d}.jpg")
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (vis_dim, vis_dim))
        text = f"{video} {keyframe}"
        img = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        images.append(img)

    num_row = math.floor(math.sqrt(k))
    rows = []
    blank_img = np.zeros_like(images[0])
    lim = math.ceil(k / num_row)
    for _ in range(num_row):
        row = images[:lim]
        if len(row) < lim:
            row = row + [blank_img] * (lim - len(row))
        row = np.concatenate(row, axis=1)
        rows.append(row)

        if len(images) > lim:
            images = images[lim:]

    concat_image = np.concatenate(rows, axis=0)
    cv2.imshow("Img", concat_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
