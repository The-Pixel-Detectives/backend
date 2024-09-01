from typing import List
from pydantic import BaseModel


class VideoResult(BaseModel):
    video_id: str
    frame_indices: List[int]
    score: float
    local_file_path: str  # absolute file path of the video on server


class SearchResult(BaseModel):
    videos: List[VideoResult] = []
