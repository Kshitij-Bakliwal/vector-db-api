from __future__ import annotations

import math, random, threading
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from vector_db_api.indexing.base import BaseIndex
from vector_db_api.indexing.utils import dot, normalize, get_similarity_function

LSH_OVERSAMPLE: int = 6 # verify up to K * LSH_OVERSAMPLE candidates
LSH_MAX_CANDIDATES: Optional[int] = None

class LSHTable:
    def __init__(self, dim: int, H: int, rng: random.Random) -> None:
        self.dim = dim
        self.H = H
        self.hyperplanes: List[List[float]] = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(H)]
        self.buckets: Dict[int, Set[UUID]] = {}
    
    def signature(self, vec: List[float]) -> int:
        sig = 0
        for i, hp in enumerate(self.hyperplanes):
            if dot(vec, hp) >= 0.0:
                sig |= (1 << i)
        return sig
    
    def add(self, chunk_id: UUID, vec: List[float]) -> None:
        sig = self.signature(vec)
        self.buckets.setdefault(sig, set()).add(chunk_id)
    
    def remove(self, chunk_id: UUID, vec: List[float]) -> None:
        sig = self.signature(vec)
        bucket = self.buckets.get(sig)
        if not bucket:
            return
        bucket.discard(chunk_id)
        if not bucket:
            self.buckets.pop(sig, None)


# TODO: understand working of LSH index
class LSHIndex(BaseIndex):
    """
    Random-hyperplane LSH for cosine + exact rerank.
    Public parameters: num_tables (L), hyperplanes_per_table (H)
    Other behavior controlled by constants above.
    """
    def __init__(self, dim: int, num_tables: int = 8, hyperplanes_per_table: int = 16, *, seed: Optional[int] = None) -> None:
        self.dim = dim
        self.L = max(1, int(num_tables))
        self.H = max(1, int(hyperplanes_per_table))

        self.rng = random.Random(seed)
        self.tables: List[LSHTable] = [LSHTable(dim, self.H, self.rng) for _ in range(self.L)]
        self.vecs: Dict[UUID, List[float]] = {}     # normalized vectors
        self.lock = threading.RLock()
    
    def add(self, chunk_id: UUID, vec: List[float]) -> None:
        norm_vec = normalize(vec)
        with self.lock:
            ov = self.vecs.get(chunk_id)
            if ov is not None:
                # Remove old vector from all tables
                for table in self.tables:
                    table.remove(chunk_id, ov)
                # Remove from vecs if new vector is None (zero vector)
                if norm_vec is None:
                    self.vecs.pop(chunk_id, None)
                    return
            
            # Add new vector if it's valid
            if norm_vec is not None:
                self.vecs[chunk_id] = norm_vec
                for table in self.tables:
                    table.add(chunk_id, norm_vec)
    
    def remove(self, chunk_id: UUID) -> None:
        with self.lock:
            ov = self.vecs.pop(chunk_id, None)
            if ov is not None:
                for table in self.tables:
                    table.remove(chunk_id, ov)
    
    def search(self, query: List[float], k: int = 10, metric: str = "cosine") -> List[Tuple[UUID, float]]:
        norm_query = normalize(query)
        if norm_query is None:
            return []
        with self.lock:
            candidates: Set[UUID] = set()
            for table in self.tables:
                b = table.buckets.get(table.signature(norm_query))
                if b is not None:
                    candidates.update(b)
            target = LSH_OVERSAMPLE * max(1, k)
            if LSH_MAX_CANDIDATES is not None:
                target = min(target, LSH_MAX_CANDIDATES)
            
            cand_list = list(candidates)[:target] if len(candidates) > target else list(candidates)
            vecs = self.vecs.copy()
        
        # If LSH doesn't find enough candidates, fall back to searching all vectors
        if len(cand_list) < k and len(vecs) > len(cand_list):
            # Add more candidates from all vectors to ensure we have enough to choose from
            all_candidates = list(vecs.keys())
            # Add candidates that weren't already included
            for cand in all_candidates:
                if cand not in candidates:
                    cand_list.append(cand)
                    if len(cand_list) >= k * 2:  # Get 2x more candidates for better results
                        break
        
        # Get the appropriate similarity function
        similarity_func = get_similarity_function(metric)
        
        # For non-normalized metrics, use original query; for cosine, use normalized
        query_for_similarity = norm_query if metric == "cosine" else query
        scores = [(cand, similarity_func(query_for_similarity, vec)) for cand in cand_list if (vec := vecs.get(cand)) is not None]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def rebuild(self, items: List[Tuple[UUID, List[float]]]) -> None:
        with self.lock:
            self.vecs.clear()
            for table in self.tables:
                table.buckets.clear()
            for cid, vec in items:
                norm_vec = normalize(vec)
                if norm_vec is not None:
                    self.vecs[cid] = norm_vec
                    for table in self.tables:
                        table.add(cid, norm_vec)
                
        