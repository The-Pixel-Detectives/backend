from typing import List
from schemas import SearchResult


class TextEngine:
    def __init__(self):
        pass

    def search(self, queries: List[str]) -> SearchResult:
        if len(queries) == 0:
            return SearchResult()

        if len(queries) == 1:
            return self.search_single(queries[0])

        return self.search_multiple(queries)

    def search_single(self, query: str) -> SearchResult:
        pass

    def search_multiple(
        self,
        queries: List[str]
    ) -> SearchResult:
        pass
