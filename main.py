import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from schemas import SearchResult, SearchRequest
from engines import TextEngine
from utils import load_image_into_numpy_array, get_sketch_img_path, get_keyframe_path
from uuid import uuid4


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


text_engine = TextEngine()


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
    concat_img = np.concatenate(imgs, axis=1)
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
    result = text_engine.search(item.text_queries, top_k=item.top_k)
    print(f"Found {len(result.videos)} videos")
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
