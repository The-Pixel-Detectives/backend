import cv2
import numpy as np
import math
import uvicorn
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from schemas import SearchResult, SearchRequest, TranslationRequest, TranslationRespone
from qdrant_client import QdrantClient
from engines import SearchEngine
from services.openai_service import OpenAIService
from utils import load_image_into_numpy_array, get_sketch_img_path, get_keyframe_path, get_video_path, visualize_images
from uuid import uuid4


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


client = QdrantClient(host="localhost", port=6333)
search_engine = SearchEngine(client)


@app.get("/")
async def homepage():
    return {"message": "Welcome to The Pixel Detectives."}


@app.get("/health")
async def check_health():
    return {"message": "I am good. Thank you."}


@app.get("/thumbnail")
async def get_video_thumbnail(group_id: str, video_id: str, frame_indices: str, is_keyframe: bool):
    '''
    Return one image containing all requested frames
    '''
    frame_indices = frame_indices.strip().split(",")
    frame_indices = list(map(int, frame_indices))
    imgs = []
    if is_keyframe:
        for idx in frame_indices:
            img_path = get_keyframe_path(group_id, video_id, idx)
            img = cv2.imread(img_path)
            imgs.append(img)
    else:
        video_path = get_video_path(group_id, video_id)
        for idx in frame_indices:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set frame position
            res, img = cap.read()
            img = cv2.resize(img, (256, 144))
            imgs.append(img)

    concat_img = visualize_images(imgs, lim=5)

    # concat_img = np.concatenate(imgs, axis=1)
    _, buffer = cv2.imencode('.png', concat_img)

    return Response(content=buffer.tobytes(), media_type="image/png")


@app.get("/get-video-preview")
async def get_video_preview(group_id: str, video_id: str, start_index: int, end_index: int, num_skip_frames: int):
    '''
    Return one image containing all requested frames
    '''
    imgs = []
    video_path = get_video_path(group_id, video_id)
    for idx in range(start_index, end_index+1, num_skip_frames):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set frame position
        res, img = cap.read()
        img = cv2.resize(img, (256, 144))
        imgs.append(img)

    concat_img = visualize_images(imgs, lim=5)

    # concat_img = np.concatenate(imgs, axis=1)
    _, buffer = cv2.imencode('.png', concat_img)

    return Response(content=buffer.tobytes(), media_type="image/png")


@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...)
):
    img = load_image_into_numpy_array(await file.read())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_id = f"{uuid4()}"
    img_path = get_sketch_img_path(img_id)
    cv2.imwrite(img_path, img)
    return {
        "id": img_id
    }


@app.post("/search")
async def search_video(
    item: SearchRequest
):
    print(item)
    result = search_engine.search(item)
    print(f"Found {len(result.videos)} videos")
    return result


@app.post("/translate")
async def translate_query(
    item: TranslationRequest
):
    print(item.query)
    response = OpenAIService.translate_query(text=item.query, num_frames=item.num_frames)
    print(response)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
