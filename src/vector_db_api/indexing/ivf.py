from __future__ import annotations

import math, random, threading
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from vector_db_api.indexing.base import BaseIndex
from vector_db_api.indexing.utils import dot, normalize, argmax_idx, get_similarity_function

IVF_KMEAN_ITERS: int = 20
IVF_MAX_CANDIDATES: Optional[int] = None

class IVFIndex(BaseIndex):
    """
    Inverted File index (cosine): k-means centroids + per-centroid posting lists, with exact rerank.

    Public params:
      - num_centroids (k): number of clusters
      - nprobe: how many nearest centroids to scan at query time

    Storage:
      - self._centroids: List[List[float]]  (normalized)
      - self._lists: Dict[int, Set[UUID]]   (centroid_id -> chunk ids)
      - self._vecs: Dict[UUID, List[float]] (normalized vectors)
      - self._assign: Dict[UUID, int]       (chunk_id -> centroid_id)
    """
    def __init__(self, dim: int, num_centroids: int, nprobe: int = 4, *, seed: Optional[int] = None) -> None:
        self.dim = dim
        self.k = max(1, int(num_centroids))
        self.nprobe = max(1, int(nprobe))

        self.rng = random.Random(seed)
        self.centroids: List[List[float]] = []
        self.lists: Dict[int, Set[UUID]] = {}
        self.vecs: Dict[UUID, List[float]] = {}
        self.assign: Dict[UUID, int] = {}

        self.lock = threading.RLock()
    
    def add(self, chunk_id: UUID, vec: List[float]) -> None:
        norm_vec = normalize(vec)
        if norm_vec is None:
            return
        with self.lock:
            self.vecs[chunk_id] = norm_vec
            if self.centroids:
                cid = self._nearest_centroid(norm_vec)
                self.assign[chunk_id] = cid
                self.lists.setdefault(cid, set()).add(chunk_id)

    def update(self, chunk_id: UUID, vec: List[float]) -> None:
        norm_vec = normalize(vec)
        with self.lock:
            old_norm_vec = self.vecs.get(chunk_id)
            old_cid = self.assign.get(chunk_id)

            if old_cid is not None and old_norm_vec is not None:
                b = self.lists.get(old_cid)
                if b is not None:
                    b.discard(chunk_id)
                    if not b:
                        self.lists.pop(old_cid, None)
            
            if norm_vec is None:
                self.vecs.pop(chunk_id, None)
                self.assign.pop(chunk_id, None)
                return
            
            self.vecs[chunk_id] = norm_vec
            if self.centroids:
                new_cid = self._nearest_centroid(norm_vec)
                self.assign[chunk_id] = new_cid
                self.lists.setdefault(new_cid, set()).add(chunk_id)
            else:
                self.assign.pop(chunk_id, None)
    
    def remove(self, chunk_id: UUID) -> None:
        with self.lock:
            self.vecs.pop(chunk_id, None)
            cid = self.assign.pop(chunk_id, None)
            if cid is not None:
                b = self.lists.get(cid)
                if b is not None:
                    b.discard(chunk_id)
                    if not b:
                        self.lists.pop(cid, None)
    
    # ----------- query and rebuild -----------
    def search(self, query: List[float], k: int = 10, metric: str = "cosine") -> List[Tuple[UUID, float]]:
        norm_query = normalize(query)

        if norm_query is None:
            return []

        with self.lock:
            if not self.centroids:
                vecs = self.vecs.copy()
                # score outside of lock
                pass
            else:
                closest_centroids = [dot(norm_query, c) for c in self.centroids]

                nprobe = min(self.nprobe, len(closest_centroids))
                top_idx = sorted(range(len(closest_centroids)), key=lambda i: closest_centroids[i], reverse=True)[:nprobe]

                cands = set()
                for i in top_idx:
                    lst = self.lists.get(i)
                    if lst:
                        cands.update(lst)
                
                target = max(1, k)
                if IVF_MAX_CANDIDATES is not None:
                    target = min(target * 10, IVF_MAX_CANDIDATES)   # small oversampling factor added intentionally
                cand_list = list(cands)[:target] if len(cands) > target else list(cands)
                vecs = self.vecs.copy()
        
        # Get the appropriate similarity function
        similarity_func = get_similarity_function(metric)
        
        if not self.centroids:
            # For non-normalized metrics, use original query; for cosine, use normalized
            query_for_similarity = norm_query if metric == "cosine" else query
            scores = [(cand, similarity_func(query_for_similarity, vec)) for cand, vec in vecs.items()]
        else:
            # For non-normalized metrics, use original query; for cosine, use normalized
            query_for_similarity = norm_query if metric == "cosine" else query
            scores = [(cand, similarity_func(query_for_similarity, vecs[cand])) for cand in cand_list if cand in vecs]

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    def rebuild(self, items: List[Tuple[UUID, List[float]]]) -> None:
        vec_items: List[Tuple[UUID, List[float]]] = []
        for cid, vec in items:
            norm_vec = normalize(vec)
            if norm_vec is not None:
                vec_items.append((cid, norm_vec))
        
        with self.lock:
            self.vecs = {cid: norm_vec for cid, norm_vec in vec_items}
            n = len(vec_items)
            k = min(self.k, n) if n > 0 else 0

            self.centroids = []
            self.lists.clear()
            self.assign.clear()

            if k == 0:
                return

            # Initialize centroids
            sample_ids = [cid for cid, _ in vec_items]
            self.rng.shuffle(sample_ids)
            init_ids = sample_ids[:k]
            id_to_vec = dict(vec_items)
            centroids = [id_to_vec[cid] for cid in init_ids]

            # Run k-means iterations
            for _ in range(IVF_KMEAN_ITERS):
                buckets: List[List[List[float]]] = [[] for _ in range(k)]
                for _, vec in vec_items:
                    sims = [dot(vec, c) for c in centroids]
                    closest_idx = argmax_idx(sims)
                    buckets[closest_idx].append(vec)
                
                new_centroids = []
                for b in buckets:
                    if not b:
                        # re-seed empty centroid
                        new_centroids.append(id_to_vec[self.rng.choice(sample_ids)])
                    else:
                        acc = [0.0] * self.dim
                        for vec in b:
                            for i, xi in enumerate(vec):
                                acc[i] += xi
                        norm_vec = normalize(acc)
                        new_centroids.append(norm_vec if norm_vec is not None else id_to_vec[self.rng.choice(sample_ids)])
                
                centroids = new_centroids
            
            # Assign final centroids to self.centroids
            self.centroids = centroids
            
            # Assign vectors to centroids
            for cid, vec in vec_items:
                closest_idx = self._nearest_centroid(vec, centroids)
                self.assign[cid] = closest_idx
                self.lists.setdefault(closest_idx, set()).add(cid)
        
    # ----------- helpers -----------
    def _nearest_centroid(self, vec: List[float], centroids: Optional[List[List[float]]] = None) -> int:
        C = centroids if centroids is not None else self.centroids
        if not C:
            return 0
        best_i, best_s = 0, dot(vec, C[0])
        for i in range(1, len(C)):
            s = dot(vec, C[i])
            if s > best_s:
                best_i, best_s = i, s
        return best_i
    

    