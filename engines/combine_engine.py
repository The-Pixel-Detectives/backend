import numpy as np
from typing import List
from schemas import SearchResult
from .image_engine import ImageEngine
from .text_engine import TextEngine
from schemas import SearchRequest, VideoResult, ImageResult
from .keyword_engine import KeywordEngine
from elasticsearch import Elasticsearch
from services.openai_service import OpenAIService


class SearchEngine:
    def __init__(self, client):
        self.image_engine = ImageEngine(client)
        self.text_engine = TextEngine(client)

        es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
        self.keyword_engine = KeywordEngine(es)

        self.weights = {
            "text": 0.9,
            "image": 0.1,
            "keyword": 0.8
        }
        self.text_keep_threshold = 0.11
        self.image_keep_threshold = 0.45
        self.frame_index_purnish_w = 1.0
        self.search_top_k_factor = 20
        self.keyword_keep_threshold = 0.5

    def search(self, item: SearchRequest) -> SearchResult:
        top_k = item.top_k
        text_queries = [x.strip() for x in item.text_queries if x.strip() != ""]
        image_queries = [x.strip() for x in item.image_ids if x.strip() != ""]

        is_multiple_search = len(text_queries) > 1

        # search image by text
        text_results = []
        for query in text_queries:
            # print("Original", query)
            # variations = OpenAIService.create_variations(query, num_variations=3).variations
            # query = [query] + variations
            # print("Variations", variations)
            text_results += self.text_engine.search(
                queries=[query],
                top_k=top_k * self.search_top_k_factor
            )

        # search by keyword
        if item.use_keyword_search and item.vietnamese_query:
            print("Vietnamese keyword searching")
            keyword_results = []
            keyword_results += self.keyword_engine.search_vietnamese(
                queries=[item.vietnamese_query],
                top_k=top_k * self.search_top_k_factor,
                fuzzy = item.fuzzy
            )
        else:
            keyword_results = []

        text_results.extend(keyword_results)

        image_dict = {}
        for item in text_results:
            id = item.id
            if item.score < self.text_keep_threshold:
                continue
            item.score = item.score * self.weights["text"]
            image_dict[id] = item

        # search image by image
        if not is_multiple_search:
            image_results = self.image_engine.search(
                queries=image_queries,
                top_k=top_k * self.search_top_k_factor
            )

            for item in image_results:
                id = item.id
                if item.score < self.image_keep_threshold:
                    continue
                item.score = item.score * self.weights["image"]
                if id not in image_dict:
                    image_dict[id] = item
                else:
                    image_dict[id].score += item.score


        video_dict = {}
        for item in image_dict.values():
            id = item.video_id
            if id not in video_dict:
                video_dict[id] = [item]
            else:
                video_dict[id].append(item)

        if is_multiple_search:
            return self.handle_multple_text_queries(video_dict, text_queries, top_k=top_k)

        result = []
        for video in video_dict.values():
            video.sort(key=lambda x: x.score, reverse=True)
            # video = video[:min(len(video), 5)]

            keyframes = [x.keyframe for x in video]
            keyframes.sort()

            frame_indices = [x.frame_index for x in video]
            frame_indices.sort()

            scores = [x.score for x in video]

            first_item = video[0]
            item = VideoResult(
                video_id=first_item.video_id,
                group_id=first_item.group_id,
                fps=first_item.fps,
                keyframes=keyframes,
                frame_indices=frame_indices,
                timestamps=[v/first_item.fps for v in frame_indices],
                score=max(scores),
                local_file_path=first_item.local_file_path,
                display_keyframe=True

            )

            result.append(item)

        result.sort(key=lambda x: x.score, reverse=True)
        # result = result[:min(len(result), top_k)]

        return SearchResult(videos=result)

    def handle_multple_text_queries(
        self,
        video_dict: dict,
        queries: List[str],
        top_k: int
    ):
        result = []
        for video in video_dict.values():
            if len(video) == 0:
                continue

            video.sort(key=lambda x: x.frame_index)
            fps = video[0].fps
            window_len = fps * 60
            # split into chunks of 30s
            chunk = []
            for item in video:
                if len(chunk) == 0:
                    chunk.append(item)
                    continue

                if item.frame_index - chunk[0].frame_index > window_len:
                    segment = self.find_video_segment(chunk, queries)
                    if segment is not None:
                        result.append(segment)

                    min_index = item.frame_index - window_len
                    chunk = [x for x in chunk if x.frame_index > min_index]
                chunk.append(item)

            if len(chunk) > 0:
                segment = self.find_video_segment(chunk, queries)
                if segment is not None:
                    result.append(segment)

        result.sort(key=lambda x: x.score, reverse=True)
        # result = result[:min(len(result), top_k)]

        return SearchResult(videos=result)

    def find_video_segment(self, video: List[ImageResult], queries: List[str]):
        query_pos_map = {}
        data = []
        cnt = 0
        for query in queries:
            query_pos_map[query] = cnt
            data.append([])
            cnt += 1

        keyframe_map = {}
        for i, item in enumerate(video):
            data[query_pos_map[item.query]].append(item)
            keyframe_map[item.frame_index] = item.keyframe

        indices = []
        score = 0
        prev_min_index = 0
        for row in data:
            row = [x for x in row if x.frame_index > prev_min_index]
            if len(row) == 0:
                continue

            row.sort(key=lambda x: x.score, reverse=True)
            row = row[:min(5, len(row))]
            indices.extend([x.frame_index for x in row])
            score += max([x.score for x in row])
            prev_min_index = min([x.frame_index for x in row])

        indices = list(set(indices))
        indices.sort()

        keyframes = [keyframe_map[x] for x in indices]

        return VideoResult(
            video_id=video[0].video_id,
            group_id=video[0].group_id,
            fps=video[0].fps,
            keyframes=keyframes,
            frame_indices=indices,
            timestamps=[v/video[0].fps for v in indices],
            score=score,
            local_file_path=video[0].local_file_path,
            display_keyframe=True,
        )
