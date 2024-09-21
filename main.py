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
import os


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

@app.post("/open-video")
async def open_video(video_id: str, group_id: str, start_index: int, end_index: int, num_skip_frames: int):
    '''
    Open the video file, check the OS, and process frames from start_index to end_index, skipping num_skip_frames in between.
    '''
    video_path = get_video_path(request.group_id, request.video_id)
    imgs = []

    if os.path.exists(video_path):
        try:
            # Check the operating system and log which one is being used
            if os.name == 'posix':  # macOS or Linux
                print("Operating system: macOS or Linux")
            elif os.name == 'nt':  # Windows
                print("Operating system: Windows")

            # Open the video using cv2
            cap = cv2.VideoCapture(video_path)

            # if not cap.isOpened():
            #     return JSONResponse(content={"error": "Failed to open video with cv2"}, status_code=500)

            # Process the specified frame range
            for idx in range(start_index, end_index + 1, num_skip_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the position to the desired frame
                ret, frame = cap.read()

                # if not ret:
                #     return JSONResponse(content={"error": f"Failed to read frame at index {idx}"}, status_code=500)

                # Resize frame to 256x144 (as per your requirement)
                resized_frame = cv2.resize(frame, (256, 144))
                imgs.append(resized_frame)

            cap.release()

            # Display the frames (if required for debugging or visualization)
            for img in imgs:
                cv2.imshow('Frame', img)
                cv2.waitKey(25)  # Display each frame for 25 ms (adjust as needed)

            cv2.destroyAllWindows()

            return Response(content={"status": "Video processed successfully", "video_id": video_id}, status_code=200)

        except Exception as e:
            return Response(content={"error": str(e)}, status_code=500)
    else:
        return Response(content={"error": "Video not found"}, status_code=404)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
