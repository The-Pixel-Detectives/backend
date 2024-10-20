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
    timestamps: List[float]
    fps: float
    score: float
    local_file_path: str  # absolute file path of the video on server
    display_keyframe: Optional[bool] = True


class SearchResult(BaseModel):
    videos: List[VideoResult] = []


class SearchRequest(BaseModel):
    image_ids: Optional[list[str]] = []  # list of uploaded image ids
    text_queries: Optional[list[str]] = []
    top_k: int = 20
    use_keyword_search: bool = False
    fuzzy: bool = False
    vietnamese_query: Optional[str]

class TranslationRequest(BaseModel):
    query: str
    num_frames: int


class TranslationRespone(BaseModel):
    sentences: list[str]


class VariationResponse(BaseModel):
    variations: list[str]


class OpenVideoRequest(BaseModel):
    video_id: str
    group_id: str
    start_time: float  # seconds
