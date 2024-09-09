from typing import List, Optional
from pydantic import BaseModel
from fastapi import UploadFile, File


class ImageResult(BaseModel):
    id: str
    video_id: str
    group_id: str
    keyframe: int
    frame_index: int
    score: float
    local_file_path: str  # absolute file path of the video on server
    query: Optional[str] = None
    fps: float


class VideoResult(BaseModel):
    video_id: str
    group_id: str
    keyframes: List[int]
    frame_indices: List[int]
    fps: float
    score: float
    local_file_path: str  # absolute file path of the video on server


class SearchResult(BaseModel):
    videos: List[VideoResult] = []


class SearchRequest(BaseModel):
    image_ids: Optional[list[str]] = []  # list of uploaded image ids
    text_queries: Optional[list[str]] = []
    top_k: int = 10
