from typing import List
import cv2
from model import jina_model
from config import settings
from .base import BaseEngine
from utils import get_sketch_img_path


class ImageEngine(BaseEngine):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.collection_name = settings.JINA_INDEX
        self.model = jina_model

    def extract_embedding(self, query) -> List[float]:
        # extract text embeddings
        img = cv2.imread(get_sketch_img_path(query))
        query_vector = self.model.extract_embedding(img)
        return query_vector.tolist()
