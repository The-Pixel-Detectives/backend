from typing import List, Optional
from pydantic import BaseModel
from fastapi import UploadFile, File


class VideoResult(BaseModel):
    video_id: str
    group_id: str
    keyframes: List[int]
    frame_indices: List[int]
    score: float
    local_file_path: str  # absolute file path of the video on server


class SearchResult(BaseModel):
    videos: List[VideoResult] = []


class SearchRequest(BaseModel):
    sketches: Optional[list[str]] = []  # list of uploaded image ids
    text_queries: Optional[list[str]] = []
    top_k: int = 10
