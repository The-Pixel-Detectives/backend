from pydantic import BaseModel
from typing import Optional

# Schema for the search request - main /search
class SearchRequest(BaseModel):
    query: str
    num_frames: int
    video_id: Optional[str] = None  # Optional video_id to filter by video ID
