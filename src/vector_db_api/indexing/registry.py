from __future__ import annotations
from typing import Dict, Optional, Iterable, Tuple, List
from uuid import UUID
import threading

from vector_db_api.indexing.base import BaseIndex
from vector_db_api.indexing.flat import FlatIndex
from vector_db_api.indexing.lsh import LSHIndex
from vector_db_api.indexing.ivf import IVFIndex

from vector_db_api.models.indexing import IndexType
from vector_db_api.models.entities import Library

class IndexRegistry:
    def __init__(self) -> None:
        self.registry: Dict[UUID, BaseIndex] = {}
        self.lock = threading.Lock()

    def get_or_create(self, lib: Library) -> BaseIndex:
        with self.lock:
            index = self.registry.get(lib.id)
            if index is not None:
                return index
            index = self.create_index(lib.index_config, lib.embedding_dim)
            self.registry[lib.id] = index
            return index
    
    def get(self, lib_id: UUID) -> Optional[BaseIndex]:
        # No deep copy: callers treat this as read-only
        with self.lock:
            return self.registry.get(lib_id)

    def swap(self, lib_id: UUID, new_index: BaseIndex) -> None:
        with self.lock:
            self.registry[lib_id] = new_index
        
    def remove(self, lib_id: UUID) -> None:
        with self.lock:
            self.registry.pop(lib_id, None)

    # helper function to create an index based on the index type
    def create_index(self, index_config: IndexType, embedding_dim: int) -> BaseIndex:
        if index_config.type == "flat":
            return FlatIndex()
        
        if index_config.type == "lsh":
            L = getattr(index_config, "lsh_num_tables", 8) or 8
            H = getattr(index_config, "lsh_hyperplanes_per_table", 16) or 16
            return LSHIndex(dim=embedding_dim, num_tables=L, hyperplanes_per_table=H)

        if index_config.type == "ivf":
            K = getattr(index_config, "ivf_num_centroids", 64) or 64
            NP = getattr(index_config, "ivf_nprobe", 4) or 4
            return IVFIndex(dim=embedding_dim, num_centroids=K, nprobe=NP)

        raise ValueError(f"Unsupported index type: {index_config.type}")