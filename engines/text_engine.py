from typing import List
from model import jina_model
from config import settings
from .base import BaseEngine


class TextEngine(BaseEngine):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.collection_name = settings.JINA_INDEX
        self.model = jina_model

    def extract_embedding(self, query) -> List[float]:
        # extract text embeddings
        query_vector = self.model.extract_text_embedding(query)
        return query_vector.tolist()
