from typing import List
import numpy as np
from model import embedding_model
from config import settings
from .base import BaseEngine


class TextEngine(BaseEngine):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.collection_name = settings.SIGLIP_INDEX
        self.model = embedding_model

    def extract_embedding(self, query) -> List[float]:
        # extract text embeddings
        if type(query) is str:
            query_vector = self.model.extract_text_embedding(query)
        else:
            vectors = []
            for text in query:
                query_vector = self.model.extract_text_embedding(text)
                vectors.append(query_vector)
            stacked_vector = np.stack(vectors)
            print(stacked_vector, stacked_vector.shape)
            query_vector = np.mean(stacked_vector, axis=0)

        return query_vector.tolist()
