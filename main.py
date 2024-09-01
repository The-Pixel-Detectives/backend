import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from schemas import SearchResult


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def homepage():
    return {"message": "Welcome to The Pixel Detectives."}


@app.get("/health")
async def check_health():
    return {"message": "I am good. Thank you."}


@app.get("/thumbnail")
async def get_video(video_id: str, frame_indices: List[int]):
    '''
    Return one image containing all requested frames
    '''
    result = None  # OpenCV image
    _, buffer = cv2.imencode('.png', result)

    return Response(content=buffer.tobytes(), media_type="image/png")


@app.post("/search")
async def search_video(
    sketches: List[UploadFile] = File(...),
    text_queries: List[str] = []
):
    result = []
    return SearchResult(videos=result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
