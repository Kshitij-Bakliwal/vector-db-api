from typing import List, Dict, Optional, Tuple
from uuid import UUID

from vector_db_api.repos.libraries import LibraryRepo
from vector_db_api.repos.chunks import ChunkRepo
from vector_db_api.concurrency.locks import LibraryLockRegistry
from vector_db_api.indexing.registry import IndexRegistry
from vector_db_api.services.exceptions import NotFoundError, ValidationError

class SearchService:
    def __init__(self, libs: LibraryRepo, chunks: ChunkRepo, locks: LibraryLockRegistry, indexes: IndexRegistry) -> None:
        self.libs = libs
        self.chunks = chunks

        self.locks = locks
        self.indexes = indexes
    
    def query(self, lib_id: UUID, query_embedding: List[float], k: int = 10, metric: str = "cosine", filters: Optional[Dict] = None) -> List[Dict]:
        lib = self.libs.get(lib_id)
        if not lib:
            raise NotFoundError(f"Library with id {lib_id} not found.")

        if len(query_embedding) != lib.embedding_dim: 
            raise ValidationError(f"Embedding dim mismatch: got {len(query_embedding)}, expected {lib.embedding_dim}.")
        
        # Get index under read lock and then release to search without blocking
        lock = self.locks.lock_for_library(lib_id)
        lock.acquire_read()

        try:
            index = self.indexes.get(lib_id)
            if index is None:
                return []
        finally:
            lock.release_read()
        
        hits: List[Tuple[UUID, float]] = index.search(query_embedding, k, metric)

        # Apply filters if provided
        doc_ids = set(filters.get("doc_ids", [])) if filters and filters.get("doc_ids") else None
        tags = set(filters.get("tags", [])) if filters and filters.get("tags") else None
        author = filters.get("author") if filters and filters.get("author") else None
        created_after = filters.get("created_after") if filters and filters.get("created_after") else None

        results = []
        for chunk_id, score in hits:
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue
            
            # Apply chunk-level filters
            if doc_ids is not None and chunk.document_id not in doc_ids:
                continue
                
            if tags is not None and not any(tag in chunk.metadata.tags for tag in tags):
                continue
                
            if author is not None and chunk.metadata.author != author:
                continue
                
            if created_after is not None and chunk.created_at <= created_after:
                continue
            
            results.append({
                "chunk_id": chunk_id,
                "document_id": chunk.document_id,
                "score": score,
                "text": chunk.text,
                "position": chunk.position,
                "metadata": chunk.metadata.model_dump(),
                "created_at": chunk.created_at,
                "updated_at": chunk.updated_at
            })
        
        return results
            