from typing import List
from schemas import SearchResult, VideoResult
from glob import glob
from qdrant_client import QdrantClient
from model import JinaCLIP
from config import settings
from utils import get_video_path


class TextEngine:
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = settings.JINA_INDEX
        self.model = JinaCLIP()

    def search(self, queries: List[str], top_k: int = 10) -> SearchResult:
        if len(queries) == 0:
            return self.search_single(query=queries[0], top_k=top_k)

        if len(queries) == 1:
            return self.search_single(queries[0])

        return self.search_multiple(queries)

    def search_single(self, query: str, top_k: int = 10) -> SearchResult:
        # extract text embeddings
        query_vector = self.model.extract_text_embedding(query)
        query_vector = query_vector.tolist()

        # vector search
        hits = self.client.search(
           collection_name=self.collection_name,
           query_vector=query_vector,
           limit=top_k * 3
        )

        # extract metadata from result
        video_dict = {}
        for point in hits:
            group = point.payload['group']
            video = point.payload['video']
            keyframe = point.payload['keyframe']
            frame_idx = point.payload['frame_idx']
            score = point.score
            if video not in video_dict:
                video_dict[video] = {
                    "group": group,
                    "keyframes": [keyframe],
                    "frame_indices": [frame_idx],
                    "scores": [score]
                }
            else:
                video_dict[video]["keyframes"].append(keyframe)
                video_dict[video]["frame_indices"].append(frame_idx)
                video_dict[video]["scores"].append(score)

        result = []
        for video, item in video_dict.items():
            score = sum(item["scores"])
            frame_indices = item["frame_indices"]
            frame_indices.sort()
            keyframes = item["keyframes"]
            keyframes.sort()
            result.append(VideoResult(
                video_id=video,
                group_id=item["group"],
                frame_indices=frame_indices,
                keyframes=keyframes,
                score=score,
                local_file_path=get_video_path(item["group"], video)
            ))
        result.sort(key=lambda x: x.score, reverse=True)
        result = result[:min(len(result), top_k)]

        return SearchResult(videos=result)

    def search_multiple(
        self,
        queries: List[str]
    ) -> SearchResult:
        pass
