from typing import List, Optional
from uuid import UUID
from datetime import datetime

from vector_db_api.models.entities import Chunk
from vector_db_api.repos.libraries import LibraryRepo
from vector_db_api.repos.documents import DocumentRepo
from vector_db_api.repos.chunks import ChunkRepo
from vector_db_api.concurrency.locks import LibraryLockRegistry
from vector_db_api.indexing.registry import IndexRegistry

from vector_db_api.services.exceptions import NotFoundError, ConflictError, ValidationError

class ChunkService:
    def __init__(self, libs: LibraryRepo, docs: DocumentRepo, chunks: ChunkRepo, locks: LibraryLockRegistry, indexes: IndexRegistry) -> None:
        self.libs = libs
        self.docs = docs
        self.chunks = chunks

        self.locks = locks
        self.indexes = indexes
    
    def upsert(self, chunk: Chunk) -> Chunk:
        lib = self.libs.get(chunk.library_id)
        if not lib:
            raise NotFoundError(f'Library with id {chunk.library_id} not found.')
        
        doc = self.docs.get(chunk.document_id)
        if not doc or doc.library_id != chunk.library_id:
            raise ValidationError(f'Document with id {chunk.document_id} not found or not in library.')
        
        if chunk.embedding is not None and len(chunk.embedding) != lib.embedding_dim:
            raise ValidationError(f'Embedding dim mismatch: got {len(chunk.embedding)}, expected {lib.embedding_dim}.')
        
        lock = self.locks.lock_for_library(chunk.library_id)
        lock.acquire_write()

        try:
            existing_chunk = self.chunks.get(chunk.id)
            index = self.indexes.get_or_create(lib)

            if existing_chunk is not None:
                expected_version = existing_chunk.version
                chunk.updated_at = datetime.utcnow()
                chunk.version = existing_chunk.version

                if not self.chunks.update_on_version(chunk, expected_version):
                    raise ConflictError(f'Chunk with id {chunk.id} modified concurrently during upsert.')
                
                if chunk.id not in doc.chunk_ids:
                    doc.chunk_ids.append(chunk.id)
                    if not self.docs.update_on_version(doc, doc.version):
                        raise ConflictError(f'Document with id {doc.id} modified concurrently during upsert.')
                
                if existing_chunk.embedding is not None:
                    index.update(chunk.id, chunk.embedding)
            
            else:
                now = datetime.utcnow()
                chunk.created_at = now
                chunk.updated_at = now
                self.chunks.add(chunk)

                doc.chunk_ids.append(chunk.id)
                doc.updated_at = now
                if not self.docs.update_on_version(doc, doc.version):
                    raise ConflictError(f'Document with id {doc.id} modified concurrently during upsert.')
                
                if chunk.embedding is not None:
                    index.add(chunk.id, chunk.embedding)
            
            return chunk
        finally:
            lock.release_write()
    
    def bulk_upsert(self, lib_id: UUID, doc_id: UUID, chunks: List[Chunk]) -> List[Chunk]:
        lib = self.libs.get(lib_id)
        if not lib:
            raise NotFoundError(f'Library with id {lib_id} not found.')
        
        doc = self.docs.get(doc_id)
        if not doc or doc.library_id != lib_id:
            raise ValidationError(f'Document with id {doc_id} not found or not in library.')
        
        embedding_dim = lib.embedding_dim

        lock = self.locks.lock_for_library(lib_id)
        lock.acquire_write()

        try:
            index = self.indexes.get_or_create(lib)
            created_chunks = []

            for chunk in chunks:
                chunk.library_id = lib_id
                chunk.document_id = doc_id
                
                if chunk.embedding is not None and len(chunk.embedding) != embedding_dim:
                    raise ValidationError(f'Embedding dim mismatch: got {len(chunk.embedding)}, expected {embedding_dim}.')
                
                now = datetime.utcnow()
                chunk.created_at = now
                chunk.updated_at = now
                self.chunks.add(chunk)
                doc.chunk_ids.append(chunk.id)
                if chunk.embedding is not None:
                    index.add(chunk.id, chunk.embedding)
                created_chunks.append(chunk)
            
            doc.updated_at = now
            if not self.docs.update_on_version(doc, doc.version):
                raise ConflictError(f'Document with id {doc.id} modified concurrently during bulk upsert.')
            
            return created_chunks
        finally:
            lock.release_write()
    
    def delete(self, lib_id: UUID, chunk_id: UUID) -> None:
        chunk = self.chunks.get(chunk_id)
        if not chunk or chunk.library_id != lib_id:
            return
        
        doc = self.docs.get(chunk.document_id)

        lock = self.locks.lock_for_library(lib_id)
        lock.acquire_write()

        try:
            index = self.indexes.get(lib_id)
            if index is not None and chunk.embedding is not None:
                index.remove(chunk.id)
            self.chunks.delete(chunk_id)
            if doc is not None and chunk_id in doc.chunk_ids:
                doc.chunk_ids.remove(chunk_id)
            doc.updated_at = datetime.utcnow()
            if not self.docs.update_on_version(doc, doc.version):
                raise ConflictError(f'Document with id {doc.id} modified concurrently during delete.')
        finally:
            lock.release_write()
