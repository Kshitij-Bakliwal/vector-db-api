from typing import Dict, List, Optional, Iterable
from uuid import UUID
from datetime import datetime
from vector_db_api.models.entities import Chunk

class ChunkRepo:
    def __init__(self) -> None:
        self.chunks: Dict[UUID, Chunk] = {}
        self.chunks_by_document: Dict[UUID, List[UUID]] = {}
        self.chunks_by_library: Dict[UUID, List[UUID]] = {}
    
    def add(self, chunk: Chunk) -> None:
        now = datetime.utcnow()
        chunk.created_at = now
        chunk.updated_at = now
        self.chunks[chunk.id] = chunk
        self.chunks_by_document.setdefault(chunk.document_id, []).append(chunk.id)
        self.chunks_by_library.setdefault(chunk.library_id, []).append(chunk.id)
    
    def add_bulk(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            self.add(chunk)
    
    def get(self, chunk_id: UUID) -> Optional[Chunk]:
        chunk = self.chunks.get(chunk_id)
        return chunk.model_copy(deep=True) if chunk else None
    
    def list_by_document(self, doc_id: UUID, limit: int = 100, offset: int = 0) -> List[Chunk]:
        chunks = self.chunks_by_document.get(doc_id, [])
        selected_chunks = chunks[offset: offset + limit]
        return [self.chunks[chunk_id].model_copy(deep=True) for chunk_id in selected_chunks]
    
    def list_by_library(self, lib_id: UUID, limit: int = 100, offset: int = 0) -> List[Chunk]:
        chunks = self.chunks_by_library.get(lib_id, [])
        selected_chunks = chunks[offset: offset + limit]
        return [self.chunks[chunk_id].model_copy(deep=True) for chunk_id in selected_chunks]
    
    def update_on_version(self, chunk: Chunk, expected_version: int) -> bool:
        current_chunk = self.chunks.get(chunk.id)

        if current_chunk is None or current_chunk.version != expected_version:
            return False
        
        # remove from library and document if they have changed
        if current_chunk.document_id != chunk.document_id:
            old_chunks = self.chunks_by_document.get(current_chunk.document_id, [])
            if chunk.id in old_chunks:
                old_chunks.remove(chunk.id)
            self.chunks_by_document.setdefault(chunk.document_id, []).append(chunk.id)
        
        if current_chunk.library_id != chunk.library_id:
            old_chunks = self.chunks_by_library.get(current_chunk.library_id, [])
            if chunk.id in old_chunks:
                old_chunks.remove(chunk.id)
            self.chunks_by_library.setdefault(chunk.library_id, []).append(chunk.id)
        
        chunk.version = expected_version + 1
        chunk.updated_at = datetime.utcnow()
        self.chunks[chunk.id] = chunk
        return True
    
    def delete(self, chunk_id: UUID) -> bool:
        chunk = self.chunks.pop(chunk_id, None)
        if not chunk:
            return False

        chunks_in_doc = self.chunks_by_document.get(chunk.document_id, [])
        if chunk_id in chunks_in_doc:
            chunks_in_doc.remove(chunk_id)

        chunks_in_lib = self.chunks_by_library.get(chunk.library_id, [])
        if chunk_id in chunks_in_lib:
            chunks_in_lib.remove(chunk_id)

        return True

    def delete_by_document(self, doc_id: UUID) -> int:
        chunk_ids = list(self.chunks_by_document.get(doc_id, []))
        count = 0
        for chunk_id in chunk_ids:
            self.delete(chunk_id)
            count += 1
        return count