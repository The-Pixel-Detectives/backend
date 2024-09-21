import cv2
import os
import numpy as np
import math
import threading
import uvicorn
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from schemas import SearchResult, SearchRequest, TranslationRequest, TranslationRespone, OpenVideoRequest
from qdrant_client import QdrantClient
from engines import SearchEngine
from services.openai_service import OpenAIService
from utils import load_image_into_numpy_array, get_sketch_img_path, get_keyframe_path, get_video_path, visualize_images
from uuid import uuid4
from services.export_csv import generate_frame_indices, export_to_csv
from schemas import SearchRequest
from fastapi.exceptions import HTTPException

if os.name == 'posix':  # macOS or Linux
    os.system("alias vlc='/Applications/VLC.app/Contents/MacOS/VLC'")
os.makedirs("exports", exist_ok=True)


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

def vlc_open(video_path, start_time):
    if os.name == 'posix':  # macOS or Linux
        os.system(f"/Applications/VLC.app/Contents/MacOS/VLC --start-time={start_time} --play-and-exit {video_path} &")
    else:
        os.system(f"start vlc --start-time={start_time} --play-and-exit {video_path}")

@app.get("/")
async def homepage():
    return {"message": "Welcome to The Pixel Detectives."}

@app.get("/health")
async def check_health():
    return {"message": "I am good. Thank you."}


@app.get("/get-image")
async def get_single_image(group_id: str, video_id: str, frame_index: str):
    '''
    Return one image containing all requested frames
    '''
    img_path = get_keyframe_path(group_id, video_id, frame_index)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 144))

    _, buffer = cv2.imencode('.png', img)

    return Response(content=buffer.tobytes(), media_type="image/png")


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
            img = cv2.resize(img, (256, 144))
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

    # If video_id is provided, filter the search results by video_id
    # if item.video_id:
    #     result = search_engine.search(item)
    #     filtered_videos = [video for video in result.videos if video.video_id == item.video_id]
    #     result.videos = filtered_videos
    # else:
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

@app.get("/export-csv")
async def export_csv(
    video_id: str, start_time: float, first_frame_end_time: float, end_time: float, filename: str, qa: str
):
    try:
        group_id = video_id.split("_")[0]
        video_path = get_video_path(group_id, video_id)
        print("video_path", video_path)

        # Generate frame index for the first frame (from start_time to first_frame_end_time then get the middle)
        first_frame_indices = generate_frame_indices(video_path, start_time, first_frame_end_time)
        middle_first_frame_index = first_frame_indices[:min(len(first_frame_indices), 10)] # middle frame index of the first range

        frame_indices = generate_frame_indices(video_path, start_time, end_time) # remaining 99 frames
        frame_indices = middle_first_frame_index + frame_indices[:99-len(middle_first_frame_index)]  # concat

        filepath = export_to_csv(video_id, frame_indices, filename, qa)

        return Response(content=open(filepath, "rb").read(), media_type="text/csv", headers={
            "Content-Disposition": f"attachment; filename={filename}.csv"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/open-video")
async def open_video(request: OpenVideoRequest):
    '''
    Open the video file, check the OS, and process frames from start_index to end_index, skipping num_skip_frames in between.
    '''
    print(request)
    video_id = request.video_id
    group_id = video_id.split("_")[0]
    video_path = get_video_path(group_id, request.video_id)
    vlc_open(video_path, request.start_time)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
