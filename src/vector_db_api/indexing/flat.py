from typing import Dict, List, Tuple
from uuid import UUID

from vector_db_api.indexing.base import BaseIndex
from vector_db_api.indexing.utils import get_similarity_function

# TODO: check algorithm and complexity
class FlatIndex(BaseIndex):
    def __init__(self) -> None:
        self.vectors: Dict[UUID, List[float]] = {}
    
    def add(self, chunk_id: UUID, vec: List[float]) -> None:
        self.vectors[chunk_id] = vec
    
    def update(self, chunk_id: UUID, vec: List[float]) -> None:
        self.vectors[chunk_id] = vec
    
    def remove(self, chunk_id: UUID) -> None:
        self.vectors.pop(chunk_id, None)
    
    def search(self, query: List[float], k: int = 10, metric: str = "cosine") -> List[Tuple[UUID, float]]:
        similarity_func = get_similarity_function(metric)
        similarities = [(chunk_id, similarity_func(query, vec)) for chunk_id, vec in self.vectors.items()]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def rebuild(self, items: List[Tuple[UUID, List[float]]]) -> None:
        self.vectors = {chunk_id: vec for chunk_id, vec in items}
    