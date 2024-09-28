from elasticsearch import Elasticsearch
from typing import List
from schemas import ImageResult

class KeywordEngine:
    def __init__(self, elasticsearch_client: Elasticsearch, index_name: str = "transcriptions"):
        self.es = elasticsearch_client
        self.index_name = index_name

    def search(self, queries: List[str], top_k: int = 20, fuzzy: bool = False, fuzz_threshold: int = 70):
        """
        Perform a regular keyword search with optional fuzzy logic.
        """
        results = []
        for query in queries:
            query_results = self.search_elasticsearch(query, top_k=top_k, fuzzy=fuzzy, fuzz_threshold=fuzz_threshold)
            results.extend(query_results)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def search_elasticsearch(self, keyword: str, top_k: int, fuzzy: bool, fuzz_threshold: int):
        """
        Search Elasticsearch using the provided keyword. If fuzzy is enabled, perform a fuzzy search.
        """
        if fuzzy:
            query = {
                "query": {
                    "match": {
                        "content": {
                            "query": keyword,
                            "fuzziness": "AUTO",
                            # "operator":  "and"
                        }
                    }
                }
            }
        else:
            query = {
                "query": {
                    "match": {
                        "content": keyword
                    }
                }
            }

        top_k = 20
        # Elasticsearch search request
        response = self.es.search(index=self.index_name, body=query, size=top_k)

        # Parse the results from Elasticsearch
        results = []
        for hit in response['hits']['hits']:
            source = hit["_source"]
            score = hit["_score"]

            # Handle missing fields by providing default values
            keyframe = source.get("keyframe", 0)  # Default to 0 if missing
            frame_index = source.get("frame_idx", 0)  # Default to 0 if missing
            local_file_path = source.get("local_file_path", "")  # Default to an empty string if missing

            # Append the result with default values
            results.append(
                ImageResult(
                    id=hit["_id"],
                    video_id=source.get("video_id", ""),  # Default to empty string if missing
                    group_id=source.get("group_id", ""),  # Default to empty string if missing
                    frame_index=frame_index,
                    keyframe=keyframe,
                    local_file_path=local_file_path,  # Set to empty string if not available
                    score=score,
                    fps=25,  # Default FPS, update if necessary
                    content=source.get("content", "")  # Default to empty string if missing
                )
            )


        return results

    def search_vietnamese(self, queries: List[str], top_k: int = 20, fuzzy: bool = False, fuzz_threshold: int = 70):
        """
        Perform a Vietnamese keyword search with optional fuzzy logic.
        """
        results = []
        print(f"Performing Vietnamese keyword search for queries: {queries}, Fuzzy: {fuzzy}")
        for query in queries:
            query_results = self.search_elasticsearch(query, top_k=top_k, fuzzy=fuzzy, fuzz_threshold=fuzz_threshold)
            results.extend(query_results)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

