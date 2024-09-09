from typing import List
from schemas import SearchResult, ImageResult
from utils import get_video_path


class BaseEngine:
    def search(self, queries: List[str], top_k: int = 10) -> List[ImageResult]:
        if len(queries) == 0:
            return []

        if len(queries) == 1:
            return self.search_single(queries[0], top_k=top_k)

        return self.search_multiple(queries, top_k=top_k)

    def extract_embedding(self, query) -> List[float]:
        raise NotImplementedError()

    def search_single(self, query: str, top_k: int = 10) -> List[ImageResult]:
        query_vector = self.extract_embedding(query)

        # vector search
        hits = self.client.search(
           collection_name=self.collection_name,
           query_vector=query_vector,
           limit=top_k
        )

        # extract metadata from result
        result = []
        for point in hits:
            result.append(ImageResult(
                id=str(point.id),
                video_id=point.payload["video"],
                group_id=point.payload["group"],
                frame_index=point.payload['frame_idx'],
                keyframe=point.payload['keyframe'],
                # fps=point.payload['fps'],
                fps=25,
                score=point.score,
                local_file_path=get_video_path(point.payload["group"], point.payload["video"]),
                query=query
            ))

        result.sort(key=lambda x: x.score, reverse=True)
        result = result[:min(len(result), top_k)]

        return result

    def search_multiple(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> SearchResult:
        results = []
        for query in queries:
            results.append(self.search_single(
                query=query,
                top_k=top_k * 5
            ).videos)
