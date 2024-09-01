import numpy as np
from typing import List
from schemas import SearchResult


class SketchEngine:
    def __init__(self):
        pass

    def search(self, queries: List[np.ndarray]) -> SearchResult:
        if len(queries) == 0:
            return SearchResult()

        if len(queries) == 1:
            return self.search_single(queries[0])

        return self.search_multiple(queries)

    def search_single(self, query: np.ndarray) -> SearchResult:
        pass

    def search_multiple(
        self,
        queries: List[np.ndarray]
    ) -> SearchResult:
        pass
