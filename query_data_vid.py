import os
from glob import glob
from qdrant_client import QdrantClient
import numpy as np
import math
import cv2
from model import Img2VecResnet18, ColorDescriptor, Clip4Clip, Jina, CLIP

RESNET_INDEX = "resnet_embeddings"
HIST_INDEX = "hist_embeddings"
CLIP4CLIP_INDEX = "clip4clip_embeddings"
JINA_INDEX = "jina_embeddings"
CLIP_INDEX = "clip_embeddings"

collection_name = CLIP4CLIP_INDEX

client = QdrantClient(host="localhost", port=6333)

if collection_name == RESNET_INDEX:
    img2vec = Img2VecResnet18()
elif collection_name == HIST_INDEX:
    img2vec = ColorDescriptor((8, 12, 3))
elif collection_name == CLIP4CLIP_INDEX:
    img2vec = Clip4Clip()
elif collection_name == JINA_INDEX:
    img2vec = Jina()
elif collection_name == CLIP_INDEX:
    img2vec = CLIP()

video_dir = "./video"  # Directory where videos are stored
img_dir = "./data/keyframes"  # Directory for keyframes

while True:
    if collection_name in {CLIP4CLIP_INDEX, JINA_INDEX, CLIP_INDEX}:
        text_query = input("Query: ").strip()
        if text_query == "q":
            break

        query_vector = img2vec.extract_text_embedding([text_query])
        if isinstance(query_vector[0], list):  # Check if the first element is a list
            query_vector = query_vector[0]  # Flatten the nested list
    else:
        img_path = input("Img Path: ")
        img_path = img_path.strip()
        if img_path == "q" or img_path == "":
            break

        image = cv2.imread(img_path)
        query_vector = img2vec.extract_embedding(image)  # For Jina

    k = 20
    hits = client.search(
       collection_name=collection_name,
       query_vector=query_vector,
       limit=k
    )

    # Visualize result
    vis_dim = 224
    images = []
    
    # Check if we're using CLIP4CLIP or JINA or CLIP Indexes
    if collection_name not in {CLIP4CLIP_INDEX, JINA_INDEX, CLIP_INDEX}:
        if image is not None:
            image = cv2.resize(image, (vis_dim, vis_dim))
            img = cv2.putText(image, "Query", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            images.append(image)
        else:
            print("Query image could not be loaded.")

    for point in hits:
        video = point.payload['video']
        keyframe = point.payload['keyframe']
        frame_indices = point.payload.get('frame_indices', [])
        score = point.score
        video_path = os.path.join(video_dir, video + ".mp4")

        # Load the video and extract the specified frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            continue
        
        # Extract and visualize frames based on frame_indices
        for frame_number in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)  # Set frame position
            res, frame = cap.read()
            if not res:
                print(f"Error reading frame {frame_number} from {video_path}")
                continue

            frame = cv2.resize(frame, (vis_dim, vis_dim))
            text = f"{score:.2f} {video} frame {frame_number}"
            frame = cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            images.append(frame)

        cap.release()  # Release the video file

    if not images:
        print("No images to display.")
        continue

    # Arrange images for display
    num_row = math.floor(math.sqrt(len(images)))
    rows = []
    blank_img = np.zeros_like(images[0])
    lim = math.ceil(len(images) / num_row)
    for _ in range(num_row):
        row = images[:lim]
        if len(row) < lim:
            row.extend([blank_img] * (lim - len(row)))
        row = np.concatenate(row, axis=1)
        rows.append(row)

        if len(images) > lim:
            images = images[lim:]

    concat_image = np.concatenate(rows, axis=0)
    
    # Set the fixed width and height for display
    fixed_width = 1200  # Example fixed width
    fixed_height = 600  # Example fixed height

    # Resize the image to the fixed size
    resized_image = cv2.resize(concat_image, (fixed_width, fixed_height))

    # Display the resized image
    cv2.imshow("Img", resized_image)

    cv2.waitKey(0)
    break 
    cv2.destroyAllWindows()
